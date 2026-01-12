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
        config: VanillaMCTSConfig with embedded core and trainer configs

    Returns:
        Trainer instance ready for "training" (validation)

    Example:
        game = TicTacToe()
        config = VanillaMCTSConfig(
            core=MCTSConfig(num_sims=1000),
            trainer=TrainerArgs(verbose=True),
        )
        trainer = create_vanilla_mcts_trainer(game, None, config)
        trainer.run()
    """
    # Use embedded configs directly (no extraction needed with composition)
    mcts = VanillaMCTS(
        game=game,
        cfg=config.core
    )

    trainer = Trainer(
        game=game,
        mcts=mcts,
        args=config.trainer
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
    # Pass embedded core config directly (no field-by-field extraction)
    return VanillaMCTSAgentConfig(
        mcts=config.core,
        device=config.trainer.device,
    )
