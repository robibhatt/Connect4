"""
Tests for AlphaZeroConfig.

Verifies:
- Config construction from dict (YAML parsing)
- Field validation
- Extraction of legacy configs (TrainerArgs, MCTSConfig)
- Default values
"""

import pytest

from src.algorithms.alphazero.config import AlphaZeroConfig
from src.algorithms.alphazero import TrainerArgs, MCTSConfig


# ===== Construction Tests =====

def test_create_config_from_dict():
    """Test creating config from YAML-like dict"""
    config_dict = {
        'model_class': 'TicTacToeMLPNet',
        'model_kwargs': {'hidden': 64},
        'device': 'mps',
        'iterations': 400,
        'games_per_iteration': 8,
        'num_sims': 50,
        'c_puct': 1.25,
        'lr': 0.003,
        'batch_size': 64,
    }
    config = AlphaZeroConfig(**config_dict)

    assert config.model_class == 'TicTacToeMLPNet'
    assert config.model_kwargs == {'hidden': 64}
    assert config.iterations == 400
    assert config.num_sims == 50
    assert config.device == 'mps'


def test_config_defaults():
    """Test that minimal config uses defaults from TrainerArgs and MCTSConfig"""
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={}
    )

    # Should have defaults from TrainerArgs
    assert config.iterations == 100  # TrainerArgs default
    assert config.games_per_iteration == 25  # TrainerArgs default
    assert config.lr == 1e-3  # TrainerArgs default

    # Should have defaults from MCTSConfig
    assert config.num_sims == 200  # MCTSConfig default
    assert config.c_puct == 1.5  # MCTSConfig default


# ===== Validation Tests =====

def test_config_requires_model_class():
    """Test that model_class is required"""
    with pytest.raises(TypeError):
        AlphaZeroConfig(iterations=100)  # Missing model_class


def test_config_validates_device():
    """Test device field is accessible"""
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={},
        device='cpu'
    )
    assert config.device == 'cpu'


# ===== Legacy Config Extraction Tests =====

def test_to_trainer_args():
    """Test extracting TrainerArgs from AlphaZeroConfig"""
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={'hidden': 64},
        device='mps',
        iterations=400,
        games_per_iteration=8,
        temp_moves=4,
        tau=1.0,
        deterministic_after_temp=True,
        add_dirichlet_noise=True,
        batch_size=64,
        train_steps_per_iteration=50,
        lr=0.003,
        weight_decay=0.0001,
        value_loss_coef=1.0,
        buffer_capacity=40000,
        clear_mcts_each_game=True,
        # MCTS fields (should NOT be in TrainerArgs)
        num_sims=50,
        c_puct=1.25,
    )

    trainer_args = config.to_trainer_args()

    assert isinstance(trainer_args, TrainerArgs)
    assert trainer_args.iterations == 400
    assert trainer_args.games_per_iteration == 8
    assert trainer_args.device == 'mps'
    assert trainer_args.lr == 0.003
    assert not hasattr(trainer_args, 'num_sims')  # MCTS field should not be included
    assert not hasattr(trainer_args, 'c_puct')  # MCTS field should not be included


def test_to_mcts_config():
    """Test extracting MCTSConfig from AlphaZeroConfig"""
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={},
        num_sims=50,
        c_puct=1.25,
        dirichlet_alpha=0.6,
        dirichlet_eps=0.2,
        illegal_action_penalty=1e9,
        # Trainer fields (should NOT be in MCTSConfig)
        iterations=400,
        lr=0.003,
    )

    mcts_config = config.to_mcts_config()

    assert isinstance(mcts_config, MCTSConfig)
    assert mcts_config.num_sims == 50
    assert mcts_config.c_puct == 1.25
    assert mcts_config.dirichlet_alpha == 0.6
    assert not hasattr(mcts_config, 'iterations')  # Trainer field should not be included
    assert not hasattr(mcts_config, 'lr')  # Trainer field should not be included


# ===== Field Coverage Tests =====

def test_all_trainer_args_fields_present():
    """Ensure all 14 TrainerArgs fields are in AlphaZeroConfig"""
    trainer_fields = [
        'iterations', 'games_per_iteration', 'temp_moves', 'tau',
        'deterministic_after_temp', 'add_dirichlet_noise', 'batch_size',
        'train_steps_per_iteration', 'lr', 'weight_decay', 'value_loss_coef',
        'buffer_capacity', 'device', 'clear_mcts_each_game'
    ]

    # Create config with minimal required fields
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={}
    )

    # All trainer fields should be present (either from kwargs or defaults)
    for field in trainer_fields:
        assert hasattr(config, field), f"Missing field: {field}"


def test_all_mcts_config_fields_present():
    """Ensure all 5 MCTSConfig fields are in AlphaZeroConfig"""
    mcts_fields = [
        'num_sims', 'c_puct', 'dirichlet_alpha',
        'dirichlet_eps', 'illegal_action_penalty'
    ]

    # Create config with minimal required fields
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={}
    )

    # All MCTS fields should be present (either from kwargs or defaults)
    for field in mcts_fields:
        assert hasattr(config, field), f"Missing field: {field}"
