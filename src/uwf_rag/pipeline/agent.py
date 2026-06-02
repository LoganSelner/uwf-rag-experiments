"""Agent pipeline — single-agent ReAct (Phase D1).

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

Multi-agent (supervisor) mode is Phase D2; the validator rejects
``agent.mode == "multi"`` for now.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from uwf_rag.components._tool_protocol import dump_arguments, synthesize_call_id
from uwf_rag.components.base import BaseGenerator, BaseMemory, BaseTool
from uwf_rag.components.build import BuildContext
from uwf_rag.components.generators import build_generator
from uwf_rag.components.tools import tool_to_spec
from uwf_rag.core.config import ExperimentConfig
from uwf_rag.core.registry import registry
from uwf_rag.core.types import (
    AgentStep,
    GenerationResult,
    Message,
    RetrievedChunk,
    ToolCall,
    ToolSpec,
)

if TYPE_CHECKING:
    from uwf_rag.pipeline.indexing import IndexArtifact

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
        # Reserved for multi-turn evaluation (Phase E); the single-query loop
        # below does not thread history yet.
        self._memory = memory

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        index_artifact: IndexArtifact,
    ) -> AgentPipeline:
        """Build the reasoning LLM, the tool roster, and the loop budget.

        Takes the full ``ExperimentConfig`` (not just ``config.agent``) so the
        RAG tool can reuse ``config.query``'s retrieval + rerank stack.
        """
        agent = config.agent

        # The build context carries the index + the query stack (which the RAG
        # tool reuses) + the generator factory, plus the registry for any
        # recursive construction (the Phase D2 supervisor's sub-agents).
        ctx = BuildContext(
            registry=registry,
            index=index_artifact,
            query=config.query,
            make_generator=build_generator,
        )

        # Reasoning generator — built from agent.llm via the shared factory
        # (sub_provider / base_url ride through agent.llm.params).
        generator: BaseGenerator = ctx.make_generator(agent.llm)

        # Tool roster — each tool builds itself from its entry + the context.
        tools: list[BaseTool] = []
        for entry in agent.tools:
            tool_cls = registry.get("tool", entry.type)
            tools.append(tool_cls.build(entry, ctx))
        tool_specs = [tool_to_spec(t) for t in tools]

        mem_cls = registry.get("memory", agent.memory.type or "none")
        memory: BaseMemory = mem_cls(config={"window_size": agent.memory.window_size})

        return cls(
            reasoning_generator=generator,
            tools=tools,
            tool_specs=tool_specs,
            max_iterations=agent.max_iterations,
            top_k_final=config.query.retrieval.top_k_final,
            system_prompt=agent.system_prompt,
            memory=memory,
        )

    def run(self, query: str) -> GenerationResult:
        """Run the ReAct loop for a single query."""
        messages: list[Message] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
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

    def query(self, question: str) -> GenerationResult:
        """Satisfy the Queryable protocol — delegates to run()."""
        return self.run(question)

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
