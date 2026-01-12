"""Pytest fixtures for agent tests."""

import pytest
import numpy as np

from src.agents.agent import RandomAgent


@pytest.fixture
def random_agent_tictactoe(tictactoe_game, numpy_rng):
    """Deterministic RandomAgent for TicTacToe."""
    return RandomAgent(tictactoe_game, numpy_rng)


@pytest.fixture
def random_agent_connect4(connect4_game, numpy_rng):
    """Deterministic RandomAgent for Connect4."""
    return RandomAgent(connect4_game, numpy_rng)
