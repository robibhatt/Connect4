"""Pytest fixtures for agent tests."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock
from pathlib import Path

from src.agents.agent import RandomAgent
from src.agents.alphazero_agent import AlphaZeroAgent
from src.agents.alphazero_agent_config import AlphaZeroAgentConfig
from src.agents.tictactoe_alphazero_agent import TicTacToeAlphaZeroAgent
from src.agents.connect4_alphazero_agent import Connect4AlphaZeroAgent
from src.algorithms.alphazero import MCTS, MCTSConfig
from src.games.tictactoe.models.mlp import TicTacToeMLPNet
from src.games.connect4.models.mlp import Connect4MLPNet


@pytest.fixture
def random_agent_tictactoe(tictactoe_game, numpy_rng):
    """Deterministic RandomAgent for TicTacToe."""
    return RandomAgent(tictactoe_game, numpy_rng)


@pytest.fixture
def random_agent_connect4(connect4_game, numpy_rng):
    """Deterministic RandomAgent for Connect4."""
    return RandomAgent(connect4_game, numpy_rng)


@pytest.fixture
def mock_mcts():
    """Mock MCTS for agent testing."""
    mcts = Mock(spec=['play_move', 'clear', 'run', 'select_action', 'model'])
    mcts.play_move = Mock(return_value=0)
    mcts.clear = Mock()
    mcts.run = Mock(return_value=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    mcts.select_action = Mock(return_value=0)
    mcts.model = Mock(spec=nn.Module)
    return mcts


@pytest.fixture
def sample_agent_config():
    """Sample AlphaZeroAgentConfig for testing."""
    return AlphaZeroAgentConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={'hidden': 8},  # Match the tiny model used in tictactoe_mcts_real
        num_sims=10,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9,
        device='cpu'
    )


@pytest.fixture
def tictactoe_mcts_real(tictactoe_game, device, numpy_rng):
    """Real MCTS instance with small TicTacToeMLPNet for integration tests."""
    # Use tiny model for speed
    model = TicTacToeMLPNet(hidden=8)
    model.to(device)
    model.eval()

    # Use minimal MCTS config for speed
    mcts_cfg = MCTSConfig(
        num_sims=10,  # Very few sims for fast tests
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9
    )

    return MCTS(
        game=tictactoe_game,
        model=model,
        device=device,
        cfg=mcts_cfg,
        rng=numpy_rng
    )


@pytest.fixture
def connect4_mcts_real(connect4_game, device, numpy_rng):
    """Real MCTS instance with small Connect4MLPNet for integration tests."""
    # Use tiny model for speed
    model = Connect4MLPNet(hidden=16)
    model.to(device)
    model.eval()

    # Use minimal MCTS config for speed
    mcts_cfg = MCTSConfig(
        num_sims=10,  # Very few sims for fast tests
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9
    )

    return MCTS(
        game=connect4_game,
        model=model,
        device=device,
        cfg=mcts_cfg,
        rng=numpy_rng
    )


@pytest.fixture
def sample_tictactoe_agent(tictactoe_game, tictactoe_mcts_real):
    """TicTacToeAlphaZeroAgent with real model for checkpoint tests."""
    return TicTacToeAlphaZeroAgent(game=tictactoe_game, mcts=tictactoe_mcts_real)


@pytest.fixture
def sample_connect4_agent(connect4_game, connect4_mcts_real):
    """Connect4AlphaZeroAgent with real model for checkpoint tests."""
    return Connect4AlphaZeroAgent(game=connect4_game, mcts=connect4_mcts_real)


@pytest.fixture
def sample_alphazero_agent_tictactoe(tictactoe_game, tictactoe_mcts_real):
    """Generic AlphaZeroAgent for TicTacToe with real model for tests."""
    return AlphaZeroAgent(game=tictactoe_game, mcts=tictactoe_mcts_real)


@pytest.fixture
def sample_alphazero_agent_connect4(connect4_game, connect4_mcts_real):
    """Generic AlphaZeroAgent for Connect4 with real model for tests."""
    return AlphaZeroAgent(game=connect4_game, mcts=connect4_mcts_real)


@pytest.fixture
def checkpoint_dir_with_tictactoe_agent(tmp_path, sample_tictactoe_agent, sample_agent_config):
    """Pre-saved TicTacToe agent checkpoint."""
    from src.agents.checkpoint_utils import save_agent_checkpoint

    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )
    return save_dir


@pytest.fixture
def checkpoint_dir_with_connect4_agent(tmp_path, sample_connect4_agent):
    """Pre-saved Connect4 agent checkpoint."""
    from src.agents.checkpoint_utils import save_agent_checkpoint
    from src.agents.alphazero_agent_config import AlphaZeroAgentConfig

    config = AlphaZeroAgentConfig(
        model_class='Connect4MLPNet',
        model_kwargs={'hidden': 16},
        num_sims=10,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9,
        device='cpu'
    )

    save_dir = save_agent_checkpoint(
        agent=sample_connect4_agent,
        agent_class_name='Connect4AlphaZeroAgent',
        game_name='connect4',
        config=config,
        root_dir=str(tmp_path)
    )
    return save_dir
