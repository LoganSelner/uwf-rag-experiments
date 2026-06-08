"""Agent pipeline — single-agent ReAct (D1) + multi-agent supervisor (D2).

One reasoning LLM drives a reason→act(tool)→observe loop over a roster of
tools, deciding for itself when it has enough evidence to answer. The
mechanism is **native tool/function calling** (not text-protocol parsing):
each iteration calls ``generator.generate(messages, tools=specs)``; if the
model emits tool calls they are executed and their results fed back via the
neutral message schema, otherwise the model's text is the final answer. The
loop runs up to ``agent.max_iterations`` steps; if the budget is exhausted a
final answer is forced with ``tools=None``.

The loop is intentionally monolithic (small private methods, no pluggable
reasoning-strategy abstraction) — there is exactly one mechanism today. A
future text-protocol strategy, if a non-tool-calling backend ever needs one,
would lift cleanly out of the ``run`` loop / ``_execute_tool_calls``.

**Multi-agent (``agent.mode == "multi"``)** is the *same* loop: a supervisor is
just an ``AgentPipeline`` whose tools are :class:`SubAgentTool`s wrapping
specialist ``AgentPipeline``s (agents-as-tools). The mode only changes
construction (``from_config`` → ``_build_supervisor`` vs ``_build_single``);
``run`` never branches on it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from ragbench.components._tool_protocol import dump_arguments, synthesize_call_id
from ragbench.components.base import BaseGenerator, BaseMemory, BaseTool
from ragbench.components.build import BuildContext
from ragbench.components.generators import build_generator
from ragbench.components.tools import SubAgentTool, tool_to_spec
from ragbench.core.config import (
    AgentConfig,
    AgentSpecConfig,
    ExperimentConfig,
    MemoryConfig,
    ToolConfig,
)
from ragbench.core.registry import registry
from ragbench.core.types import (
    AgentStep,
    GenerationResult,
    Message,
    RetrievedChunk,
    ToolCall,
    ToolSpec,
)

if TYPE_CHECKING:
    from ragbench.pipeline.indexing import IndexArtifact

logger = logging.getLogger(__name__)

# Per-step tool output is truncated in the AgentStep trace (the full text still
# rides in the message history the model sees) to keep result JSONL compact.
_STEP_OUTPUT_PREVIEW = 500


class AgentPipeline:
    """Single-agent ReAct pipeline (tool-calling loop)."""

    def __init__(
        self,
        reasoning_generator: BaseGenerator,
        tools: list[BaseTool],
        tool_specs: list[ToolSpec],
        max_iterations: int,
        top_k_final: int,
        system_prompt: str = "",
        memory: BaseMemory | None = None,
    ) -> None:
        self._generator = reasoning_generator
        self._tools = tools
        self._tools_by_name = {t.name: t for t in tools}
        self._tool_specs = tool_specs
        self._max_iterations = max_iterations
        self._top_k_final = top_k_final
        self._system_prompt = system_prompt
        # Conversation memory is kept as a registered, honest stub. Multi-turn
        # evaluation threads history through ``run(query, history)`` — the
        # evaluator owns conversation state as the single source of truth — not
        # through this object, which would only matter for a stateful serving
        # surface this harness doesn't have.
        self._memory = memory

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        index_artifact: IndexArtifact,
    ) -> AgentPipeline:
        """Build a single agent (``mode='single'``) or supervisor (``'multi'``).

        Takes the full ``ExperimentConfig`` (not just ``config.agent``) so the
        RAG tool — and each multi-agent specialist — can reuse ``config.query``'s
        retrieval + rerank stack over the same cached index. The single/multi
        branch is the only mode-dependent code; the ``run`` loop is identical for
        both (a supervisor is just an agent whose tools are sub-agents).
        """
        agent = config.agent

        # The build context carries the index + the query stack (which the RAG
        # tool reuses) + the generator factory, plus the registry. Each
        # multi-agent specialist is built by recursing through it.
        ctx = BuildContext(
            registry=registry,
            index=index_artifact,
            query=config.query,
            make_generator=build_generator,
        )
        top_k_final = config.query.retrieval.top_k_final

        if agent.mode == "multi":
            return cls._build_supervisor(agent, ctx, top_k_final)
        return cls._build_single(agent, ctx, top_k_final)

    @classmethod
    def _build_single(
        cls,
        agent: AgentConfig,
        ctx: BuildContext,
        top_k_final: int,
    ) -> AgentPipeline:
        """Construct a single-agent ReAct pipeline over a registry-built roster.

        Shared by the top-level ``mode='single'`` path and by each specialist a
        supervisor builds — a specialist is just a single agent over a
        source-scoped query stack.
        """
        generator: BaseGenerator = ctx.make_generator(agent.llm)

        tools: list[BaseTool] = []
        for entry in agent.tools:
            tool_cls = registry.get("tool", entry.type)
            tools.append(tool_cls.build(entry, ctx))
        tool_specs = [tool_to_spec(t) for t in tools]

        return cls(
            reasoning_generator=generator,
            tools=tools,
            tool_specs=tool_specs,
            max_iterations=agent.max_iterations,
            top_k_final=top_k_final,
            system_prompt=agent.system_prompt,
            memory=cls._build_memory(agent),
        )

    @classmethod
    def _build_supervisor(
        cls,
        agent: AgentConfig,
        ctx: BuildContext,
        top_k_final: int,
    ) -> AgentPipeline:
        """Construct a multi-agent supervisor (agents-as-tools).

        The supervisor runs the *same* ReAct loop as a single agent, but its
        roster is one :class:`SubAgentTool` per specialist. Each specialist is
        built through the same context (recursion), scoped to its own source
        filter; the supervisor advertises it by name/description so the
        supervisor LLM routes on the specialist's description.
        """
        supervisor_gen: BaseGenerator = ctx.make_generator(agent.llm)

        tools: list[BaseTool] = []
        for spec in agent.agents:
            specialist = cls._build_specialist(spec, agent, ctx, top_k_final)
            tools.append(
                SubAgentTool(
                    agent=specialist,
                    name=spec.name,
                    description=spec.description,
                )
            )
        tool_specs = [tool_to_spec(t) for t in tools]

        return cls(
            reasoning_generator=supervisor_gen,
            tools=tools,
            tool_specs=tool_specs,
            max_iterations=agent.max_iterations,
            top_k_final=top_k_final,
            system_prompt=agent.system_prompt,
            memory=cls._build_memory(agent),
        )

    @classmethod
    def _build_specialist(
        cls,
        spec: AgentSpecConfig,
        supervisor: AgentConfig,
        ctx: BuildContext,
        top_k_final: int,
    ) -> AgentPipeline:
        """Build one specialist as a single agent over a source-scoped stack.

        The specialist's ``filters`` are merged over the base query stack's
        retrieval filters (scoping it to e.g. one source); its retrieval method
        and top_k budget are inherited from the base stack so the budget stays
        equal across specialists and the linear/single baselines. ``llm`` falls
        back to the supervisor's when the specialist sets no provider; the tool
        roster defaults to a single ``rag`` tool over the scoped stack.
        """
        llm = spec.llm if spec.llm.provider else supervisor.llm

        spec_ctx = ctx
        if spec.filters and ctx.query is not None:
            scoped_retrieval = ctx.query.retrieval.model_copy(
                update={"filters": {**ctx.query.retrieval.filters, **spec.filters}}
            )
            scoped_query = ctx.query.model_copy(update={"retrieval": scoped_retrieval})
            spec_ctx = dataclasses.replace(ctx, query=scoped_query)

        tools_cfg = spec.tools or [ToolConfig(name=f"{spec.name}_search", type="rag")]
        spec_agent = AgentConfig(
            mode="single",
            max_iterations=supervisor.max_iterations,
            system_prompt=spec.system_prompt,
            llm=llm,
            memory=MemoryConfig(type="none"),
            tools=tools_cfg,
        )
        return cls._build_single(spec_agent, spec_ctx, top_k_final)

    @staticmethod
    def _build_memory(agent: AgentConfig) -> BaseMemory:
        """Build the (currently single-turn-inert) conversation memory."""
        mem_cls = registry.get("memory", agent.memory.type or "none")
        return mem_cls(config={"window_size": agent.memory.window_size})

    def run(
        self,
        query: str,
        history: list[Message] | None = None,
        injected_contexts: list[str] | None = None,
    ) -> GenerationResult:
        """Run the ReAct loop for a single query.

        ``history`` (optional) seeds prior conversation turns between the system
        prompt and the current user turn, for multi-turn evaluation. The
        evaluator supplies it; single-turn callers leave it ``None``.
        ``injected_contexts`` is rejected: the knowledge-conflict probe is
        linear-only (an agent controls its own retrieval), and the validator
        enforces conflict ⇒ ``pipeline_mode 'linear'`` — this guard catches
        programmatic misuse.
        """
        if injected_contexts:
            raise ValueError(
                "injected_contexts (the knowledge-conflict probe) is linear-only "
                "and not supported in agent mode."
            )
        messages: list[Message] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        steps: list[AgentStep] = []
        collected: list[RetrievedChunk] = []
        num_tool_calls = 0
        num_retrieval_calls = 0
        answer = ""
        iterations = 0
        forced = False

        for i in range(self._max_iterations):
            iterations = i + 1
            result = self._generator.generate(messages, tools=self._tool_specs)

            if not result.tool_calls:
                answer = result.answer
                steps.append(AgentStep(step_number=i, action="final"))
                break

            calls = self._assign_ids(result.tool_calls, step=i)
            messages.append(self._assistant_turn(result.answer, calls))
            tool_calls, retrieval_calls = self._execute_tool_calls(
                calls, messages, steps, collected, step=i
            )
            num_tool_calls += tool_calls
            num_retrieval_calls += retrieval_calls
        else:
            # Budget exhausted without a tool-free turn → force a text answer
            # (tools=None guarantees the model can't keep calling tools).
            forced = True
            final = self._generator.generate(messages, tools=None)
            answer = final.answer
            steps.append(
                AgentStep(step_number=self._max_iterations, action="final_forced")
            )

        aggregated = self._aggregate_chunks(collected)
        logger.info(
            "Agent finished in %d iteration(s): %d tool call(s), %d chunk(s)",
            iterations,
            num_tool_calls,
            len(aggregated),
        )
        return GenerationResult(
            query=query,
            answer=answer,
            retrieved_chunks=aggregated,
            metadata={
                "mode": "agent",
                "iterations": iterations,
                "forced_final": forced,
                "num_tool_calls": num_tool_calls,
                "num_retrieval_calls": num_retrieval_calls,
                "retrieved_chunk_count": len(aggregated),
                "steps": [dataclasses.asdict(s) for s in steps],
            },
        )

    def query(
        self,
        question: str,
        history: list[Message] | None = None,
        injected_contexts: list[str] | None = None,
    ) -> GenerationResult:
        """Satisfy the Queryable protocol — delegates to run()."""
        return self.run(question, history, injected_contexts)

    # --- loop internals ---------------------------------------------------

    def _execute_tool_calls(
        self,
        calls: list[ToolCall],
        messages: list[Message],
        steps: list[AgentStep],
        collected: list[RetrievedChunk],
        *,
        step: int,
    ) -> tuple[int, int]:
        """Execute each requested call, append observations, accumulate chunks.

        Returns ``(num_tool_calls, num_retrieval_calls)`` for this turn. A
        missing tool or a tool failure becomes a recoverable observation so the
        model can re-plan rather than the whole query dying.
        """
        num_tool_calls = 0
        num_retrieval_calls = 0
        for call in calls:
            num_tool_calls += 1
            query_arg = str(call.arguments.get("query", ""))
            tool = self._tools_by_name.get(call.name)
            if tool is None:
                observation = (
                    f"Unknown tool '{call.name}'. Available tools: "
                    f"{sorted(self._tools_by_name)}."
                )
                success = False
            else:
                tr = tool.execute(query_arg)
                observation = tr.content
                success = tr.success
                if tr.retrieved_chunks:
                    collected.extend(tr.retrieved_chunks)
                    num_retrieval_calls += 1
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )
            steps.append(
                AgentStep(
                    step_number=step,
                    action="tool_call",
                    tool_name=call.name,
                    tool_input=query_arg,
                    tool_output=observation[:_STEP_OUTPUT_PREVIEW],
                    metadata={"success": success},
                )
            )
        return num_tool_calls, num_retrieval_calls

    @staticmethod
    def _assign_ids(tool_calls: list[ToolCall], *, step: int) -> list[ToolCall]:
        """Synthesize ids for providers that omitted them (Ollama, Gemini).

        Correlates each assistant tool call with its result turn; providers
        that supply real ids (OpenAI, EdenAI) keep them.
        """
        return [
            ToolCall(
                id=tc.id or synthesize_call_id(step, idx),
                name=tc.name,
                arguments=tc.arguments,
            )
            for idx, tc in enumerate(tool_calls)
        ]

    @staticmethod
    def _assistant_turn(content: str, calls: list[ToolCall]) -> Message:
        """Re-serialize an assistant tool-calling turn into the neutral schema."""
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": dump_arguments(c.arguments),
                    },
                }
                for c in calls
            ],
        }

    def _aggregate_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Union retrieved chunks across all tool calls for evaluation.

        Dedups by ``chunk_id`` (keeping the highest score), orders by score,
        and caps at ``top_k_final`` — the same budget the linear pipeline
        reports, so agent and linear context-precision denominators stay
        comparable. Empty (the agent never retrieved) is a valid, informative
        signal; the evaluator simply scores no context for that query.
        """
        best: dict[str, RetrievedChunk] = {}
        for rc in chunks:
            cid = rc.chunk.chunk_id
            if cid not in best or rc.score > best[cid].score:
                best[cid] = rc
        ranked = sorted(best.values(), key=lambda rc: rc.score, reverse=True)
        return ranked[: self._top_k_final]
