"""
Factory functions for creating MCGS algorithm components.

Provides clean separation between config and instantiation logic.
"""

from __future__ import annotations

from src.games.core.game import Game
from src.algorithms.mcgs.config import MCGSConfig
from src.algorithms.mcgs.trainer import Trainer
from src.algorithms.mcgs.mcgs import MCGS
from src.algorithms.mcgs.agent_config import MCGSAgentConfig


def create_mcgs_trainer(
    game: Game,
    model,  # Kept for API consistency, but ignored
    config: MCGSConfig
) -> Trainer:
    """
    Create a MCGS Trainer instance from unified config.

    Note: model parameter is ignored (kept for API consistency with AlphaZero).

    Args:
        game: Game instance
        model: Ignored (no model needed for MCGS)
        config: MCGSConfig with embedded core and trainer configs

    Returns:
        Trainer instance ready for "training" (validation)

    Example:
        game = TicTacToe()
        config = MCGSConfig(
            core=MCGSCoreConfig(num_sims=1000),
            trainer=TrainerArgs(verbose=True),
        )
        trainer = create_mcgs_trainer(game, None, config)
        trainer.run()
    """
    # Use embedded configs directly (no extraction needed with composition)
    mcgs = MCGS(
        game=game,
        cfg=config.core
    )

    trainer = Trainer(
        game=game,
        mcgs=mcgs,
        args=config.trainer
    )

    return trainer


def create_mcgs_agent_config(config: MCGSConfig) -> MCGSAgentConfig:
    """
    Create a MCGSAgentConfig from training config.

    Args:
        config: MCGSConfig used during training

    Returns:
        MCGSAgentConfig for saving/loading agent
    """
    # Pass embedded core config directly (no field-by-field extraction)
    return MCGSAgentConfig(
        mcgs=config.core,
        device=config.trainer.device,
    )
