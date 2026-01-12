"""
Tests for scripts/simulate_match.py

Tests the simulate_match.py logic for handling different agent types.
Uses mocks to avoid depending on actual agent implementations.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml


# ===== Test Fixtures =====

@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create a mock checkpoint directory with agent.yaml only (no model.pt)"""
    checkpoint_dir = tmp_path / "test_checkpoint"
    checkpoint_dir.mkdir()

    agent_yaml = {
        'agent_class': 'VanillaMCTSAgent',
        'game': 'tictactoe',
        'mcts': {
            'num_sims': 100,
            'c_exploration': 1.414
        }
    }

    with open(checkpoint_dir / "agent.yaml", 'w') as f:
        yaml.dump(agent_yaml, f)

    return checkpoint_dir


# ===== Agent Type Validation Tests =====

class TestValidateAgentType:
    """Tests for validate_agent_type function"""

    def test_validate_agent_type_accepts_random(self):
        """random should be a valid agent type"""
        from scripts.simulate_match import validate_agent_type
        # Should not raise
        validate_agent_type('random', 1)

    def test_validate_agent_type_accepts_vanilla_mcts(self):
        """vanilla_mcts should be a valid agent type"""
        from scripts.simulate_match import validate_agent_type
        # Should not raise
        validate_agent_type('vanilla_mcts', 1)

    def test_validate_agent_type_rejects_unknown(self):
        """Unknown agent types should raise ValueError"""
        from scripts.simulate_match import validate_agent_type
        with pytest.raises(ValueError, match="Invalid agent"):
            validate_agent_type('unknown_type', 1)


# ===== Checkpoint Validation Tests =====

class TestValidateCheckpointExists:
    """Tests for validate_checkpoint_exists function"""

    def test_vanilla_mcts_checkpoint_no_model_required(self, mock_checkpoint_dir):
        """vanilla_mcts checkpoint should NOT require model.pt"""
        from scripts.simulate_match import validate_checkpoint_exists

        # Checkpoint has agent.yaml but NO model.pt
        assert not (mock_checkpoint_dir / "model.pt").exists()

        # Should NOT raise for vanilla_mcts
        validate_checkpoint_exists(mock_checkpoint_dir, 1, 'vanilla_mcts')

    def test_missing_agent_yaml_raises(self, tmp_path):
        """Checkpoint missing agent.yaml should raise for any type"""
        from scripts.simulate_match import validate_checkpoint_exists

        empty_dir = tmp_path / "empty_checkpoint"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="agent.yaml"):
            validate_checkpoint_exists(empty_dir, 1, 'vanilla_mcts')


# ===== Create Agent Tests =====

class TestCreateAgent:
    """Tests for create_agent function"""

    @patch('scripts.simulate_match.load_agent_checkpoint')
    def test_create_agent_vanilla_mcts_calls_load_checkpoint(self, mock_load, tmp_path):
        """vanilla_mcts should use load_agent_checkpoint"""
        from scripts.simulate_match import create_agent

        mock_agent = Mock()
        mock_load.return_value = mock_agent

        checkpoint_dir = tmp_path / "checkpoint"
        game = Mock()

        result = create_agent('vanilla_mcts', checkpoint_dir, game)

        mock_load.assert_called_once_with(checkpoint_dir)
        assert result == mock_agent

    def test_create_agent_random_creates_random_agent(self):
        """random should create RandomAgent"""
        from scripts.simulate_match import create_agent
        from src.agents import RandomAgent

        game = Mock()
        game.action_size = 9

        result = create_agent('random', None, game)

        assert isinstance(result, RandomAgent)

    def test_create_agent_unknown_raises(self):
        """Unknown agent type should raise KeyError from registry"""
        from scripts.simulate_match import create_agent

        with pytest.raises(KeyError, match="No metadata registered for 'unknown'"):
            create_agent('unknown', None, Mock())
