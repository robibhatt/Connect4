"""
Factory functions for creating AlphaZero algorithm components.

Provides clean separation between config and instantiation logic.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from src.games.core.game import Game
from src.algorithms.alphazero.config import AlphaZeroConfig
from src.algorithms.alphazero.trainer import Trainer
from src.algorithms.alphazero.mcts import MCTS
from src.algorithms.alphazero.agent_config import AlphaZeroAgentConfig


def create_alphazero_trainer(
    game: Game,
    model: nn.Module,
    config: AlphaZeroConfig
) -> Trainer:
    """
    Create an AlphaZero Trainer instance from unified config.

    Args:
        game: Game instance
        model: Neural network model
        config: AlphaZeroConfig with all parameters

    Returns:
        Trainer instance ready for training

    Example:
        game = TicTacToe()
        model = TicTacToeMLPNet(hidden=64)
        config = AlphaZeroConfig(
            model_class='TicTacToeMLPNet',
            model_kwargs={'hidden': 64},
            iterations=400,
            num_sims=50,
        )
        trainer = create_alphazero_trainer(game, model, config)
        trainer.run()
    """
    # Extract legacy configs for backward compatibility
    trainer_args = config.to_trainer_args()
    mcts_config = config.to_mcts_config()

    # Create MCTS instance
    device = torch.device(config.device)
    mcts = MCTS(
        game=game,
        model=model,
        device=device,
        cfg=mcts_config
    )

    # Create Trainer instance
    trainer = Trainer(
        game=game,
        model=model,
        mcts=mcts,
        args=trainer_args
    )

    return trainer


def create_alphazero_agent_config(config: AlphaZeroConfig) -> AlphaZeroAgentConfig:
    """
    Create an AlphaZeroAgentConfig from training config.

    Args:
        config: AlphaZeroConfig used during training

    Returns:
        AlphaZeroAgentConfig for saving/loading agent
    """
    return AlphaZeroAgentConfig(
        model_class=config.model_class,
        model_kwargs=config.model_kwargs,
        num_sims=config.num_sims,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_eps=config.dirichlet_eps,
        illegal_action_penalty=config.illegal_action_penalty,
        device=config.device,
    )
