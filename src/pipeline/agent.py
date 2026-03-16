"""Agent pipeline stub.

This module establishes the AgentPipeline class location and interface.
Implementation will be added in a future branch (Phase 4 of the
architecture plan). See ARCHITECTURE_PLAN.md Sections 4.2 and 8.3.

The AgentPipeline will support:
- Single-agent mode (v2): One LLM in a ReAct loop choosing tools
- Multi-agent mode (v3): Supervisor LLM routing to specialized agents

Both modes produce the same GenerationResult type as QueryPipeline.
"""

from __future__ import annotations

from core.config import AgentConfig
from core.types import GenerationResult, IndexArtifact


class AgentPipeline:
    """Agent-based pipeline (single-agent or multi-agent).

    NOT YET IMPLEMENTED. This stub establishes the file location
    and class interface. See ARCHITECTURE_PLAN.md Section 8.3.
    """

    def __init__(
        self,
        config: AgentConfig,
        index_artifact: IndexArtifact,
    ) -> None:
        raise NotImplementedError(
            "AgentPipeline is not yet implemented. "
            "This is planned for Phase 4. See ARCHITECTURE_PLAN.md."
        )

    def run(self, query: str) -> GenerationResult:
        """Run a query through the agent pipeline."""
        raise NotImplementedError("AgentPipeline.run() is not yet implemented.")
