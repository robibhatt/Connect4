"""Tests for AlphaZeroAgentConfig."""

import pytest
from src.agents.alphazero_agent_config import AlphaZeroAgentConfig


# ===== Construction Tests =====

def test_create_minimal_config():
    """Can create config with minimal required fields."""
    config = AlphaZeroAgentConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={},
        num_sims=50,
        c_puct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9
    )
    assert config is not None
    assert config.model_class == 'TicTacToeMLPNet'
    assert config.device == 'cpu'  # default value


def test_config_stores_all_fields():
    """Config should store all provided fields."""
    config = AlphaZeroAgentConfig(
        model_class='Connect4MLPNet',
        model_kwargs={'hidden': 128, 'dropout': 0.1},
        num_sims=100,
        c_puct=1.5,
        dirichlet_alpha=0.5,
        dirichlet_eps=0.3,
        illegal_action_penalty=1e10,
        device='cuda'
    )

    assert config.model_class == 'Connect4MLPNet'
    assert config.model_kwargs == {'hidden': 128, 'dropout': 0.1}
    assert config.num_sims == 100
    assert config.c_puct == 1.5
    assert config.dirichlet_alpha == 0.5
    assert config.dirichlet_eps == 0.3
    assert config.illegal_action_penalty == 1e10
    assert config.device == 'cuda'


def test_config_defaults():
    """Default values are applied correctly."""
    config = AlphaZeroAgentConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={},
        num_sims=50,
        c_puct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9
        # device not provided, should default to 'cpu'
    )

    assert config.device == 'cpu'


# ===== Serialization Tests =====

def test_to_dict_contains_all_fields(sample_agent_config):
    """to_dict() should contain all expected keys."""
    config_dict = sample_agent_config.to_dict()

    expected_keys = {
        'model_class', 'model_kwargs', 'num_sims', 'c_puct',
        'dirichlet_alpha', 'dirichlet_eps', 'illegal_action_penalty', 'device'
    }

    assert set(config_dict.keys()) == expected_keys


def test_from_dict_reconstructs_config(sample_agent_config):
    """from_dict() should reconstruct config from dict."""
    config_dict = sample_agent_config.to_dict()
    reconstructed = AlphaZeroAgentConfig.from_dict(config_dict)

    assert isinstance(reconstructed, AlphaZeroAgentConfig)
    assert reconstructed.model_class == sample_agent_config.model_class
    assert reconstructed.model_kwargs == sample_agent_config.model_kwargs
    assert reconstructed.num_sims == sample_agent_config.num_sims
    assert reconstructed.c_puct == sample_agent_config.c_puct
    assert reconstructed.device == sample_agent_config.device


def test_roundtrip_preserves_values():
    """Config should survive to_dict() -> from_dict() roundtrip."""
    original = AlphaZeroAgentConfig(
        model_class='Connect4MLPNet',
        model_kwargs={'hidden': 256, 'layers': 3},
        num_sims=200,
        c_puct=2.0,
        dirichlet_alpha=0.4,
        dirichlet_eps=0.2,
        illegal_action_penalty=1e8,
        device='mps'
    )

    config_dict = original.to_dict()
    reconstructed = AlphaZeroAgentConfig.from_dict(config_dict)

    # Compare all fields
    assert reconstructed.model_class == original.model_class
    assert reconstructed.model_kwargs == original.model_kwargs
    assert reconstructed.num_sims == original.num_sims
    assert reconstructed.c_puct == original.c_puct
    assert reconstructed.dirichlet_alpha == original.dirichlet_alpha
    assert reconstructed.dirichlet_eps == original.dirichlet_eps
    assert reconstructed.illegal_action_penalty == original.illegal_action_penalty
    assert reconstructed.device == original.device


def test_nested_model_kwargs_preserved():
    """Nested dicts in model_kwargs should survive serialization."""
    config = AlphaZeroAgentConfig(
        model_class='CustomNet',
        model_kwargs={
            'hidden': 64,
            'activation': 'relu',
            'nested': {'key1': 'value1', 'key2': 42}
        },
        num_sims=50,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9
    )

    config_dict = config.to_dict()
    reconstructed = AlphaZeroAgentConfig.from_dict(config_dict)

    assert reconstructed.model_kwargs == config.model_kwargs
    assert reconstructed.model_kwargs['nested'] == {'key1': 'value1', 'key2': 42}


# ===== Edge Cases =====

def test_config_with_empty_model_kwargs():
    """Empty dict is valid for model_kwargs."""
    config = AlphaZeroAgentConfig(
        model_class='SimpleNet',
        model_kwargs={},
        num_sims=50,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        illegal_action_penalty=1e9
    )

    assert config.model_kwargs == {}

    # Should survive roundtrip
    config_dict = config.to_dict()
    reconstructed = AlphaZeroAgentConfig.from_dict(config_dict)
    assert reconstructed.model_kwargs == {}


def test_config_with_various_model_kwargs():
    """Test with various model_kwargs structures."""
    test_cases = [
        {},
        {'hidden': 64},
        {'hidden': 128, 'dropout': 0.1},
        {'layers': [64, 128, 256], 'activation': 'relu', 'batch_norm': True}
    ]

    for kwargs in test_cases:
        config = AlphaZeroAgentConfig(
            model_class='TestNet',
            model_kwargs=kwargs,
            num_sims=50,
            c_puct=1.0,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
            illegal_action_penalty=1e9
        )

        # Roundtrip test
        config_dict = config.to_dict()
        reconstructed = AlphaZeroAgentConfig.from_dict(config_dict)
        assert reconstructed.model_kwargs == kwargs
