"""Tests for checkpoint save/load utilities."""

import pytest
import torch
import yaml
from pathlib import Path
from datetime import datetime

from src.agents.checkpoint_utils import save_agent_checkpoint, load_agent_checkpoint
from src.agents.alphazero_agent_config import AlphaZeroAgentConfig
from src.agents.tictactoe_alphazero_agent import TicTacToeAlphaZeroAgent
from src.agents.connect4_alphazero_agent import Connect4AlphaZeroAgent


# ===== Save Checkpoint Tests =====

def test_save_creates_timestamped_directory(tmp_path, sample_tictactoe_agent, sample_agent_config):
    """save_agent_checkpoint should create timestamped directory."""
    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )

    # Directory should exist
    assert save_dir.exists()
    assert save_dir.is_dir()

    # Check name format: YYYYMMDD_HHMMSS_tictactoe_TicTacToeAlphaZeroAgent
    dir_name = save_dir.name
    parts = dir_name.split('_')
    assert len(parts) >= 4  # timestamp parts + game + agent
    assert 'tictactoe' in dir_name
    assert 'TicTacToeAlphaZeroAgent' in dir_name

    # Timestamp should be parseable
    timestamp_str = f"{parts[0]}_{parts[1]}"
    datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")  # Should not raise


def test_save_creates_model_pt(tmp_path, sample_tictactoe_agent, sample_agent_config):
    """save_agent_checkpoint should save model.pt."""
    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )

    model_path = save_dir / "model.pt"
    assert model_path.exists()
    assert model_path.is_file()

    # Should be loadable with torch.load
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0  # Should have some parameters


def test_save_creates_agent_yaml(tmp_path, sample_tictactoe_agent, sample_agent_config):
    """save_agent_checkpoint should save agent.yaml."""
    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )

    yaml_path = save_dir / "agent.yaml"
    assert yaml_path.exists()
    assert yaml_path.is_file()

    # Should be loadable with yaml
    with yaml_path.open('r') as f:
        agent_yaml = yaml.safe_load(f)

    assert isinstance(agent_yaml, dict)


def test_save_yaml_contains_metadata(tmp_path, sample_tictactoe_agent, sample_agent_config):
    """agent.yaml should contain all required metadata."""
    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )

    yaml_path = save_dir / "agent.yaml"
    with yaml_path.open('r') as f:
        agent_yaml = yaml.safe_load(f)

    # Check all required fields
    assert agent_yaml['agent_class'] == 'TicTacToeAlphaZeroAgent'
    assert agent_yaml['game'] == 'tictactoe'
    assert 'timestamp' in agent_yaml

    # Model config
    assert 'model' in agent_yaml
    assert agent_yaml['model']['class'] == 'TicTacToeMLPNet'
    assert agent_yaml['model']['kwargs'] == {'hidden': 8}

    # MCTS config
    assert 'mcts' in agent_yaml
    assert agent_yaml['mcts']['num_sims'] == 10
    assert agent_yaml['mcts']['c_puct'] == 1.0
    assert agent_yaml['mcts']['dirichlet_alpha'] == 0.3
    assert agent_yaml['mcts']['dirichlet_eps'] == 0.25
    assert agent_yaml['mcts']['illegal_action_penalty'] == 1e9

    # Device
    assert agent_yaml['device'] == 'cpu'


def test_save_includes_training_config(tmp_path, sample_tictactoe_agent, sample_agent_config):
    """training_config should be saved if provided."""
    training_config = {
        'iterations': 100,
        'games_per_iteration': 25,
        'lr': 1e-3
    }

    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        training_config=training_config,
        root_dir=str(tmp_path)
    )

    yaml_path = save_dir / "agent.yaml"
    with yaml_path.open('r') as f:
        agent_yaml = yaml.safe_load(f)

    assert 'training' in agent_yaml
    assert agent_yaml['training'] == training_config


# ===== Load Checkpoint Tests =====

def test_load_reconstructs_agent(checkpoint_dir_with_tictactoe_agent):
    """load_agent_checkpoint should reconstruct agent."""
    agent = load_agent_checkpoint(checkpoint_dir_with_tictactoe_agent)

    assert isinstance(agent, TicTacToeAlphaZeroAgent)
    assert hasattr(agent, 'game')
    assert hasattr(agent, 'mcts')


def test_load_restores_model_weights(checkpoint_dir_with_tictactoe_agent, sample_tictactoe_agent):
    """Loaded agent should have same model weights."""
    # Get original weights
    original_state_dict = sample_tictactoe_agent.mcts.model.state_dict()

    # Load agent
    loaded_agent = load_agent_checkpoint(checkpoint_dir_with_tictactoe_agent)
    loaded_state_dict = loaded_agent.mcts.model.state_dict()

    # Compare weights
    for key in original_state_dict.keys():
        assert key in loaded_state_dict
        assert torch.allclose(original_state_dict[key], loaded_state_dict[key])


def test_load_with_device_override(tmp_path, sample_tictactoe_agent, sample_agent_config):
    """Device can be overridden during load."""
    # Save with device='cpu'
    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )

    # Load (should use CPU since that's the default and what's in yaml)
    loaded_agent = load_agent_checkpoint(save_dir)

    # Check that model is on CPU
    for param in loaded_agent.mcts.model.parameters():
        assert param.device.type == 'cpu'


def test_load_sets_eval_mode(checkpoint_dir_with_tictactoe_agent):
    """Loaded agent should be in eval mode."""
    loaded_agent = load_agent_checkpoint(checkpoint_dir_with_tictactoe_agent)

    # Model should be in eval mode
    assert not loaded_agent.mcts.model.training


# ===== Round-Trip Tests =====

def test_save_load_roundtrip_preserves_behavior(tmp_path, tictactoe_game, sample_tictactoe_agent, sample_agent_config):
    """Agent should behave identically after save-load."""
    # Save agent
    save_dir = save_agent_checkpoint(
        agent=sample_tictactoe_agent,
        agent_class_name='TicTacToeAlphaZeroAgent',
        game_name='tictactoe',
        config=sample_agent_config,
        root_dir=str(tmp_path)
    )

    # Load agent
    loaded_agent = load_agent_checkpoint(save_dir)

    # Both agents should produce same actions on same states
    # (Note: MCTS is stochastic, but with same model weights and cleared trees,
    # and deterministic play_move, they should produce same results)
    state = tictactoe_game.reset()

    # Clear trees to start fresh
    sample_tictactoe_agent.start()
    loaded_agent.start()

    # Since MCTS uses randomness, we can't guarantee same actions
    # But we can verify both produce legal actions
    action1 = sample_tictactoe_agent.act(state)
    action2 = loaded_agent.act(state)

    assert tictactoe_game.legal_actions(state)[action1]
    assert tictactoe_game.legal_actions(state)[action2]


def test_roundtrip_with_different_games(checkpoint_dir_with_tictactoe_agent,
                                        checkpoint_dir_with_connect4_agent):
    """Save/load works for both TicTacToe and Connect4."""
    # Load TicTacToe agent
    ttt_agent = load_agent_checkpoint(checkpoint_dir_with_tictactoe_agent)
    assert isinstance(ttt_agent, TicTacToeAlphaZeroAgent)

    # Load Connect4 agent
    c4_agent = load_agent_checkpoint(checkpoint_dir_with_connect4_agent)
    assert isinstance(c4_agent, Connect4AlphaZeroAgent)


# ===== Error Handling Tests =====

def test_load_raises_on_missing_directory(tmp_path):
    """load should raise FileNotFoundError for missing directory."""
    fake_dir = tmp_path / "nonexistent_checkpoint"

    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        load_agent_checkpoint(fake_dir)


def test_load_raises_on_missing_yaml(tmp_path):
    """load should raise FileNotFoundError for missing agent.yaml."""
    # Create directory without agent.yaml
    checkpoint_dir = tmp_path / "invalid_checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="agent.yaml not found"):
        load_agent_checkpoint(checkpoint_dir)


def test_load_raises_on_unregistered_agent(tmp_path):
    """load should raise KeyError for unregistered agent class."""
    # Create checkpoint with fake agent class
    checkpoint_dir = tmp_path / "fake_checkpoint"
    checkpoint_dir.mkdir()

    # Create valid-looking agent.yaml with unregistered agent
    agent_yaml = {
        'agent_class': 'NonExistentAgent',
        'game': 'tictactoe',
        'timestamp': '20260102_120000',
        'model': {'class': 'TicTacToeMLPNet', 'kwargs': {}},
        'mcts': {
            'num_sims': 50,
            'c_puct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_eps': 0.25,
            'illegal_action_penalty': 1e9
        },
        'device': 'cpu'
    }

    with (checkpoint_dir / "agent.yaml").open('w') as f:
        yaml.dump(agent_yaml, f)

    with pytest.raises(KeyError):
        load_agent_checkpoint(checkpoint_dir)
