"""
Configuration for Vanilla MCTS agents.

Uses composition: embeds MCTSConfig rather than duplicating fields.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

from src.agents.config import AgentConfig
from src.algorithms.vanilla_mcts.mcts import MCTSConfig


@dataclass
class VanillaMCTSAgentConfig(AgentConfig):
    """
    Configuration for Vanilla MCTS agents using composition.

    Embeds MCTSConfig to avoid field duplication. Adding a new parameter
    to MCTSConfig automatically makes it available here.
    """

    mcts: MCTSConfig
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to nested dict structure for YAML."""
        return {
            "mcts": asdict(self.mcts),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VanillaMCTSAgentConfig:
        """Deserialize from nested dict structure."""
        mcts_data = data.get("mcts", {})
        mcts = MCTSConfig(**mcts_data)
        device = data.get("device", "cpu")
        return cls(mcts=mcts, device=device)
