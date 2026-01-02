"""
Checkpointable agent interface.

Provides save/load functionality for trainable agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch

from src.games.core.game import Game
from src.agents.agent import Agent


class CheckpointableAgent(ABC):
    """
    Mixin interface for agents that can be saved/loaded from checkpoints.

    Only trainable agents need to implement this interface.
    Simple agents like RandomAgent don't need checkpointing.
    """

    @abstractmethod
    def to_checkpoint(self, save_dir: Path) -> None:
        """
        Save agent to checkpoint directory.

        Typically saves model weights to model.pt.
        Agent metadata (game, config, etc.) is handled by checkpoint_utils.

        Args:
            save_dir: Directory to save checkpoint files
        """
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        game: Game,
        device: Optional[str | torch.device] = None
    ) -> Agent:
        """
        Load agent from checkpoint directory.

        Reconstructs agent from saved files (model.pt, agent.yaml, etc.)

        Args:
            checkpoint_dir: Path to saved agent directory
            game: Game instance for the agent
            device: Optional device override for model

        Returns:
            Loaded agent ready to play
        """
        pass
