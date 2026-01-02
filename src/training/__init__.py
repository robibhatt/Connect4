"""
AlphaZero training infrastructure.

This module provides the training loop, replay buffer, and related
data structures for self-play reinforcement learning.
"""

from src.training.trainer import Trainer, TrainerArgs
from src.training.replay_buffer import ReplayBuffer, AZSample

__all__ = ['Trainer', 'TrainerArgs', 'ReplayBuffer', 'AZSample']
