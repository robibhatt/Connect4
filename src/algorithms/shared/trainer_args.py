"""
Shared TrainerArgs configuration.

Single source of truth for trainer arguments used by MCGS, vanilla_mcts, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TrainerArgs:
    """Arguments for algorithm trainers."""

    num_test_games: int = 10
    device: str = "cpu"
    random_seed: int | None = None
    verbose: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for YAML/JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainerArgs:
        """Deserialize from dict."""
        return cls(**data)
