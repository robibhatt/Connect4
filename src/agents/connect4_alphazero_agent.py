"""Connect4-specific AlphaZero MCTS agent."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from src.games.core.game import Game, State
from src.agents.agent import Agent
from src.agents.checkpointable import CheckpointableAgent
from src.algorithms.alphazero import MCTS, MCTSConfig
from src.models.registry import ModelRegistry


class Connect4AlphaZeroAgent(Agent, CheckpointableAgent):
    """
    AlphaZero MCTS agent for Connect4.

    Wraps an MCTS instance that uses a neural network for move selection.
    Can be saved to and loaded from checkpoints.
    """

    def __init__(self, game: Game, mcts: MCTS):
        """
        Initialize agent with game and MCTS.

        Args:
            game: Connect4 game instance
            mcts: Pre-configured MCTS instance with model
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

        Saves model weights to model.pt.
        Agent metadata (game, config, etc.) is handled by checkpoint_utils.

        Args:
            save_dir: Directory to save checkpoint files
        """
        model_path = save_dir / "model.pt"
        torch.save(self.mcts.model.state_dict(), model_path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        game: Game,
        device: Optional[str | torch.device] = None
    ) -> Connect4AlphaZeroAgent:
        """
        Load agent from checkpoint directory.

        Args:
            checkpoint_dir: Path to saved agent
            game: Connect4 game instance
            device: Optional device override for model

        Returns:
            Loaded agent ready to play

        Raises:
            FileNotFoundError: If checkpoint files not found
            KeyError: If model class not registered
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load agent config
        with (checkpoint_dir / "agent.yaml").open('r') as f:
            agent_yaml = yaml.safe_load(f)

        # Extract configurations
        model_config = agent_yaml['model']
        mcts_config_dict = agent_yaml['mcts']
        device_str = device or agent_yaml.get('device', 'cpu')
        device_obj = torch.device(device_str)

        # Load model
        ModelClass = ModelRegistry.get_model(model_config['class'])
        model = ModelClass(**model_config['kwargs'])

        model_path = checkpoint_dir / "model.pt"
        state_dict = torch.load(model_path, map_location=device_obj, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.to(device_obj)
        model.eval()

        # Create MCTS
        mcts_cfg = MCTSConfig(**mcts_config_dict)
        mcts = MCTS(
            game=game,
            model=model,
            device=device_obj,
            cfg=mcts_cfg
        )

        return cls(game=game, mcts=mcts)
