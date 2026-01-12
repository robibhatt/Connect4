"""Generic MCGS agent that works with any game."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import yaml

from src.games.core.game import Game, State
from src.agents.agent import Agent
from src.agents.checkpointable import CheckpointableAgent
from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig


class MCGSAgent(Agent, CheckpointableAgent):
    """
    Generic MCGS agent that works with any game.

    Wraps a MCGS instance for move selection.
    Can be saved to and loaded from checkpoints.

    Key difference from AlphaZeroAgent: No model.pt file needed!
    """

    def __init__(self, game: Game, mcgs: MCGS):
        """
        Initialize agent with game and MCGS.

        Args:
            game: Game instance
            mcgs: Pre-configured MCGS instance
        """
        super().__init__(game)
        self.mcgs = mcgs

    def act(self, s: State) -> int:
        """
        Choose action via MCGS search.

        Args:
            s: Current game state

        Returns:
            Action index
        """
        return self.mcgs.play_move(s)

    def start(self) -> None:
        """Clear MCGS tree at game start."""
        self.mcgs.clear()

    def to_checkpoint(self, save_dir: Path) -> None:
        """
        Save agent to checkpoint directory.

        For MCGS, there's no model to save!
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
    ) -> MCGSAgent:
        """
        Load agent from checkpoint directory.

        Args:
            checkpoint_dir: Path to saved agent
            game: Game instance
            device: Optional device (ignored for MCGS, kept for API consistency)

        Returns:
            Loaded agent ready to play

        Raises:
            FileNotFoundError: If checkpoint files not found
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load agent config
        with (checkpoint_dir / "agent.yaml").open('r') as f:
            agent_yaml = yaml.safe_load(f)

        # Extract MCGS configuration (nested structure from composed config)
        agent_config_dict = agent_yaml['mcgs']
        mcgs_core_dict = agent_config_dict['mcgs']

        # Create MCGS config
        mcgs_cfg = MCGSCoreConfig(**mcgs_core_dict)

        # Create MCGS instance
        mcgs = MCGS(
            game=game,
            cfg=mcgs_cfg
        )

        return cls(game=game, mcgs=mcgs)
