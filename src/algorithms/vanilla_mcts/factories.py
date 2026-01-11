"""
Factory functions for creating Vanilla MCTS algorithm components.

Provides clean separation between config and instantiation logic.
"""

from __future__ import annotations

from src.games.core.game import Game
from src.algorithms.vanilla_mcts.config import VanillaMCTSConfig
from src.algorithms.vanilla_mcts.trainer import Trainer
from src.algorithms.vanilla_mcts.mcts import VanillaMCTS
from src.algorithms.vanilla_mcts.agent_config import VanillaMCTSAgentConfig


def create_vanilla_mcts_trainer(
    game: Game,
    model,  # Kept for API consistency, but ignored
    config: VanillaMCTSConfig
) -> Trainer:
    """
    Create a Vanilla MCTS Trainer instance from unified config.

    Note: model parameter is ignored (kept for API consistency with AlphaZero).

    Args:
        game: Game instance
        model: Ignored (no model needed for Vanilla MCTS)
        config: VanillaMCTSConfig with all parameters

    Returns:
        Trainer instance ready for "training" (validation)

    Example:
        game = TicTacToe()
        config = VanillaMCTSConfig(
            num_sims=1000,
            c_exploration=1.414,
        )
        trainer = create_vanilla_mcts_trainer(game, None, config)
        trainer.run()
    """
    # Extract legacy configs for backward compatibility
    trainer_args = config.to_trainer_args()
    mcts_config = config.to_mcts_config()

    # Create MCTS instance (no model needed!)
    mcts = VanillaMCTS(
        game=game,
        cfg=mcts_config
    )

    # Create Trainer instance
    trainer = Trainer(
        game=game,
        mcts=mcts,
        args=trainer_args
    )

    return trainer


def create_vanilla_mcts_agent_config(config: VanillaMCTSConfig) -> VanillaMCTSAgentConfig:
    """
    Create a VanillaMCTSAgentConfig from training config.

    Args:
        config: VanillaMCTSConfig used during training

    Returns:
        VanillaMCTSAgentConfig for saving/loading agent
    """
    return VanillaMCTSAgentConfig(
        num_sims=config.num_sims,
        c_exploration=config.c_exploration,
        max_rollout_depth=config.max_rollout_depth,
        rollout_seed=config.rollout_seed,
        illegal_action_penalty=config.illegal_action_penalty,
        device=config.device,
    )
