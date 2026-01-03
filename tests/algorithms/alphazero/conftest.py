"""Fixtures for AlphaZero algorithm tests."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock

from src.algorithms.alphazero import AlphaZeroAgent, AlphaZeroAgentConfig, MCTS, MCTSConfig
from src.games.tictactoe.models.mlp import TicTacToeMLPNet
from src.games.connect4.models.mlp import Connect4MLPNet


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
        model_kwargs={'hidden': 8},
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
    model = TicTacToeMLPNet(hidden=8)
    model.to(device)
    model.eval()

    mcts_cfg = MCTSConfig(
        num_sims=10,
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
    model = Connect4MLPNet(hidden=16)
    model.to(device)
    model.eval()

    mcts_cfg = MCTSConfig(
        num_sims=10,
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
def sample_alphazero_agent_tictactoe(tictactoe_game, tictactoe_mcts_real):
    """Generic AlphaZeroAgent for TicTacToe with real model for tests."""
    return AlphaZeroAgent(game=tictactoe_game, mcts=tictactoe_mcts_real)


@pytest.fixture
def sample_alphazero_agent_connect4(connect4_game, connect4_mcts_real):
    """Generic AlphaZeroAgent for Connect4 with real model for tests."""
    return AlphaZeroAgent(game=connect4_game, mcts=connect4_mcts_real)


@pytest.fixture
def checkpoint_dir_with_tictactoe_agent(tmp_path, sample_alphazero_agent_tictactoe, sample_agent_config):
    """Pre-saved TicTacToe agent checkpoint."""
    from src.agents.checkpoint_utils import save_agent_checkpoint

    save_dir = save_agent_checkpoint(
        agent=sample_alphazero_agent_tictactoe,
        agent_class_name='AlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )
    return save_dir


@pytest.fixture
def checkpoint_dir_with_connect4_agent(tmp_path, sample_alphazero_agent_connect4):
    """Pre-saved Connect4 agent checkpoint."""
    from src.agents.checkpoint_utils import save_agent_checkpoint

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
        agent=sample_alphazero_agent_connect4,
        agent_class_name='AlphaZeroAgent',
        game_name='connect4',
        config=config,
        root_dir=str(tmp_path)
    )
    return save_dir
