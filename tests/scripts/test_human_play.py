"""
Tests for scripts/human_play.py

Tests that human_play.py uses AlgorithmRegistry instead of hardcoded agent types.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from src.algorithms.registry import AlgorithmRegistry


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


# ===== Registry-Driven Validation Tests =====

class TestHumanPlayUsesRegistries:
    """Tests that human_play uses registry-driven validation."""

    def test_accepts_all_registered_algorithms(self):
        """validate_agent_type should accept all registered algorithms."""
        from scripts.human_play import validate_agent_type

        for algo in AlgorithmRegistry.get_all_algorithms():
            # Should not raise
            validate_agent_type(algo)

    def test_rejects_unknown_algorithm(self):
        """Unknown types should raise with list of valid types."""
        from scripts.human_play import validate_agent_type

        with pytest.raises(ValueError) as exc:
            validate_agent_type('nonexistent')

        # Error should list valid algorithms
        error_msg = str(exc.value)
        for algo in AlgorithmRegistry.get_all_algorithms():
            assert algo in error_msg


# ===== Checkpoint Validation Tests =====

class TestCheckpointValidation:
    """Tests that checkpoint validation uses AlgorithmMetadata."""

    def test_vanilla_mcts_no_model_required(self, mock_checkpoint_dir):
        """vanilla_mcts checkpoint should NOT require model.pt (uses metadata)."""
        from scripts.human_play import validate_checkpoint_exists

        # Checkpoint has agent.yaml but NO model.pt
        assert not (mock_checkpoint_dir / "model.pt").exists()

        # Should NOT raise for vanilla_mcts
        validate_checkpoint_exists(mock_checkpoint_dir, 'vanilla_mcts')

    def test_missing_agent_yaml_raises(self, tmp_path):
        """Checkpoint missing agent.yaml should raise."""
        from scripts.human_play import validate_checkpoint_exists

        empty_dir = tmp_path / "empty_checkpoint"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="agent.yaml"):
            validate_checkpoint_exists(empty_dir, 'vanilla_mcts')


# ===== Create Agent Tests =====

class TestCreateAgent:
    """Tests for create_agent function."""

    @patch('scripts.human_play.load_agent_checkpoint')
    def test_checkpoint_agents_use_load_checkpoint(self, mock_load, tmp_path):
        """Agents with requires_checkpoint=True use load_agent_checkpoint."""
        from scripts.human_play import create_agent, PlayConfig

        mock_agent = Mock()
        mock_load.return_value = mock_agent

        checkpoint_dir = tmp_path / "checkpoint"
        game = Mock()

        config = PlayConfig(
            game_name='tictactoe',
            agent_type='vanilla_mcts',
            checkpoint_dir=checkpoint_dir,
        )

        result = create_agent(config, game)

        mock_load.assert_called_once_with(checkpoint_dir)
        assert result == mock_agent

    def test_non_checkpoint_agents_create_random(self):
        """Agents with requires_checkpoint=False create RandomAgent."""
        from scripts.human_play import create_agent, PlayConfig
        from src.agents import RandomAgent

        game = Mock()
        game.action_size = 9

        config = PlayConfig(
            game_name='tictactoe',
            agent_type='random',
            checkpoint_dir=None,
        )

        result = create_agent(config, game)

        assert isinstance(result, RandomAgent)
