"""
Neural network models for AlphaZero training.

This module provides game-specific neural network architectures
that output policy logits and value predictions.
"""

from src.models.tiktaktoenet import TicTacToeNet

__all__ = ['TicTacToeNet']
