"""
Configuration for AlphaZero agents.

Contains all parameters needed to reconstruct an AlphaZero agent from checkpoint.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

from src.agents.config import AgentConfig


@dataclass
class AlphaZeroAgentConfig(AgentConfig):
    """
    Configuration for AlphaZero MCTS agents.

    Contains:
    - Model information (class name and constructor kwargs)
    - MCTS hyperparameters
    - Device configuration
    """

    # Model info
    model_class: str
    model_kwargs: Dict[str, Any]

    # MCTS config
    num_sims: int
    c_puct: float
    dirichlet_alpha: float
    dirichlet_eps: float
    illegal_action_penalty: float

    # Device
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for saving."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AlphaZeroAgentConfig:
        """Deserialize config from dict."""
        return cls(**data)
