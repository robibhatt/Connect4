"""
AlphaZero algorithm implementation.

This module provides the complete AlphaZero self-play reinforcement learning
algorithm, including MCTS with neural network guidance, training loop, and
replay buffer for experience storage.
"""

from src.algorithms.alphazero.mcts import MCTS, MCTSConfig
from src.algorithms.alphazero.trainer import Trainer, TrainerArgs
from src.algorithms.alphazero.replay_buffer import ReplayBuffer, AZSample
from src.algorithms.alphazero.config import AlphaZeroConfig
from src.algorithms.alphazero.factories import create_alphazero_trainer
from src.algorithms.alphazero.agent import AlphaZeroAgent
from src.algorithms.alphazero.agent_config import AlphaZeroAgentConfig

__all__ = [
    'MCTS',
    'MCTSConfig',
    'Trainer',
    'TrainerArgs',
    'ReplayBuffer',
    'AZSample',
    'AlphaZeroConfig',
    'create_alphazero_trainer',
    'AlphaZeroAgent',
    'AlphaZeroAgentConfig',
]


def register_algorithm():
    """Register AlphaZero with the AlgorithmRegistry."""
    from src.algorithms.registry import AlgorithmRegistry, AlgorithmMetadata
    from src.algorithms.alphazero.factories import create_alphazero_agent_config

    AlgorithmRegistry.register(
        'alphazero',
        AlphaZeroConfig,
        create_alphazero_trainer,
        create_alphazero_agent_config,
        AlgorithmMetadata(
            requires_model=True,
            requires_checkpoint=True,
            checkpoint_files=('agent.yaml', 'model.pt')
        )
    )
