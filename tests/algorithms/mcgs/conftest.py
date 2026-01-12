"""Pytest fixtures for MCGS tests."""

import pytest

from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig


@pytest.fixture
def mcgs_config():
    """Minimal MCGS config for fast tests."""
    return MCGSCoreConfig(num_sims=10)


@pytest.fixture
def mcgs_tictactoe(tictactoe_game, mcgs_config):
    """MCGS instance for TicTacToe."""
    return MCGS(game=tictactoe_game, cfg=mcgs_config)
