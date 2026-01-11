"""
Configuration for Vanilla MCTS agents.

Contains all parameters needed to reconstruct a Vanilla MCTS agent from checkpoint.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

from src.agents.config import AgentConfig


@dataclass
class VanillaMCTSAgentConfig(AgentConfig):
    """
    Configuration for Vanilla MCTS agents.

    Contains:
    - MCTS hyperparameters
    - Device configuration (kept for consistency)

    NO model information needed (pure algorithmic approach).
    """

    # MCTS config
    num_sims: int
    c_exploration: float
    max_rollout_depth: int | None
    rollout_seed: int | None
    illegal_action_penalty: float

    # Device (kept for consistency, but not really used)
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for saving."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VanillaMCTSAgentConfig:
        """Deserialize config from dict."""
        return cls(**data)
