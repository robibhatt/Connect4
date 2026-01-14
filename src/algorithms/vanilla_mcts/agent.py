"""Generic Vanilla MCTS agent that works with any game."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import yaml

from src.games.core.game import Game, State
from src.agents.agent import Agent
from src.agents.checkpointable import CheckpointableAgent
from src.algorithms.vanilla_mcts.mcts import VanillaMCTS, MCTSConfig


class VanillaMCTSAgent(Agent, CheckpointableAgent):
    """
    Generic Vanilla MCTS agent that works with any game.

    Wraps a VanillaMCTS instance for move selection.
    Can be saved to and loaded from checkpoints.

    Key difference from AlphaZeroAgent: No model.pt file needed!
    """

    def __init__(self, game: Game, mcts: VanillaMCTS):
        """
        Initialize agent with game and MCTS.

        Args:
            game: Game instance
            mcts: Pre-configured VanillaMCTS instance
        """
        super().__init__(game)
        self.mcts = mcts

    def act(self, s: State) -> int:
        """
        Choose action via MCTS search.

        Args:
            s: Current game state

        Returns:
            Action index
        """
        return self.mcts.play_move(s)

    def start(self) -> None:
        """Clear MCTS tree at game start."""
        self.mcts.clear()

    def to_checkpoint(self, save_dir: Path) -> None:
        """
        Save agent to checkpoint directory.

        For Vanilla MCTS, there's no model to save!
        Agent metadata (game, config) is handled by checkpoint_utils.

        This method intentionally does nothing - no model.pt file created.

        Args:
            save_dir: Directory to save checkpoint files
        """
        # No model weights to save - this is intentional!
        # Only agent.yaml will be saved (by checkpoint_utils)
        pass

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        game: Game,
        device: Optional[str] = None
    ) -> VanillaMCTSAgent:
        """
        Load agent from checkpoint directory.

        Args:
            checkpoint_dir: Path to saved agent
            game: Game instance
            device: Optional device (ignored for Vanilla MCTS, kept for API consistency)

        Returns:
            Loaded agent ready to play

        Raises:
            FileNotFoundError: If checkpoint files not found
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load agent config
        with (checkpoint_dir / "agent.yaml").open('r') as f:
            agent_yaml = yaml.safe_load(f)

        # Extract MCTS configuration
        mcts_core_dict = agent_yaml['mcts']

        # Create MCTS config
        mcts_cfg = MCTSConfig(**mcts_core_dict)

        # Create MCTS instance
        mcts = VanillaMCTS(
            game=game,
            cfg=mcts_cfg
        )

        return cls(game=game, mcts=mcts)
