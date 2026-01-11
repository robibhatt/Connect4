"""
Random agent pseudo-algorithm.

This module provides a "random" algorithm entry for the registry.
Random agents don't require training or checkpoints - they simply
make random valid moves.
"""

from src.agents import RandomAgent

__all__ = ['RandomAgent']


def register_algorithm():
    """Register Random as a pseudo-algorithm with the AlgorithmRegistry."""
    from src.algorithms.registry import AlgorithmRegistry, AlgorithmMetadata

    AlgorithmRegistry.register(
        'random',
        config_class=None,  # No config needed
        trainer_factory=None,  # No training
        agent_config_factory=None,
        metadata=AlgorithmMetadata(
            requires_model=False,
            requires_checkpoint=False,
            checkpoint_files=()
        )
    )
