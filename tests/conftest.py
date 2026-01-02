"""Shared pytest fixtures for all tests."""
import pytest
import numpy as np
import torch
from pathlib import Path

from src.games.tictactoe import TicTacToe
from src.games.connect4 import Connect4


# ===== Game Fixtures =====

@pytest.fixture
def tictactoe_game():
    """TicTacToe game instance."""
    return TicTacToe()


@pytest.fixture
def connect4_game():
    """Connect4 game instance."""
    return Connect4()


# ===== Torch Fixtures =====

@pytest.fixture
def device():
    """CPU device for testing."""
    return torch.device('cpu')


@pytest.fixture
def torch_rng():
    """Seeded torch RNG for reproducibility."""
    return torch.Generator().manual_seed(42)


# ===== NumPy Fixtures =====

@pytest.fixture
def numpy_rng():
    """Seeded NumPy RNG for reproducibility."""
    return np.random.default_rng(42)


# ===== Temporary Directory Fixtures =====

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
