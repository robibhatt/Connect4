"""
Monte Carlo Tree Search (MCTS) implementation.

This module provides AlphaZero-style MCTS with neural network guidance,
including configuration and the core MCTS algorithm.
"""

from src.mcts.alphazero_mcts import MCTS, MCTSConfig

__all__ = ['MCTS', 'MCTSConfig']
