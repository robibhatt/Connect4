"""
MCGS (Monte Carlo Graph Search) algorithm implementation.

This module provides Monte Carlo Graph Search with UCB1 selection
and random rollouts, without neural networks.
"""

from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig
from src.algorithms.mcgs.trainer import Trainer, TrainerArgs
from src.algorithms.mcgs.config import MCGSConfig
from src.algorithms.mcgs.factories import create_mcgs_trainer
from src.algorithms.mcgs.agent import MCGSAgent
from src.algorithms.mcgs.agent_config import MCGSAgentConfig

__all__ = [
    'MCGS',
    'MCGSCoreConfig',
    'Trainer',
    'TrainerArgs',
    'MCGSConfig',
    'create_mcgs_trainer',
    'MCGSAgent',
    'MCGSAgentConfig',
]


def register_algorithm():
    """Register MCGS with the AlgorithmRegistry."""
    from src.algorithms.registry import AlgorithmRegistry, AlgorithmMetadata
    from src.algorithms.mcgs.factories import create_mcgs_agent_config

    AlgorithmRegistry.register(
        'mcgs',
        MCGSConfig,
        create_mcgs_trainer,
        create_mcgs_agent_config,
        AlgorithmMetadata(
            requires_model=False,
            requires_checkpoint=True,
            checkpoint_files=('agent.yaml',)
        )
    )
