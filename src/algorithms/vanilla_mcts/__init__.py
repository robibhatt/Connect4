"""
Vanilla MCTS algorithm implementation.

This module provides classic Monte Carlo Tree Search with UCB1 selection
and random rollouts, without neural networks.
"""

from src.algorithms.vanilla_mcts.mcts import VanillaMCTS, MCTSConfig
from src.algorithms.vanilla_mcts.trainer import Trainer, TrainerArgs
from src.algorithms.vanilla_mcts.config import VanillaMCTSConfig
from src.algorithms.vanilla_mcts.factories import create_vanilla_mcts_trainer
from src.algorithms.vanilla_mcts.agent import VanillaMCTSAgent
from src.algorithms.vanilla_mcts.agent_config import VanillaMCTSAgentConfig

__all__ = [
    'VanillaMCTS',
    'MCTSConfig',
    'Trainer',
    'TrainerArgs',
    'VanillaMCTSConfig',
    'create_vanilla_mcts_trainer',
    'VanillaMCTSAgent',
    'VanillaMCTSAgentConfig',
]


def register_algorithm():
    """Register VanillaMCTS with the AlgorithmRegistry."""
    from src.algorithms.registry import AlgorithmRegistry, AlgorithmMetadata
    from src.algorithms.vanilla_mcts.factories import create_vanilla_mcts_agent_config

    AlgorithmRegistry.register(
        'vanilla_mcts',
        VanillaMCTSConfig,
        create_vanilla_mcts_trainer,
        create_vanilla_mcts_agent_config,
        AlgorithmMetadata(
            requires_model=False,
            requires_checkpoint=True,
            checkpoint_files=('agent.yaml',)
        )
    )
