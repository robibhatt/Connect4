"""
Tests for scripts/train.py

Verifies:
- load_config returns correct algo_name for any algorithm
- Model creation is conditional (only when config has model_class)
- Correct trainer factory is used based on algo_name
- Agent config factory works via registry
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

# Import will be tested after implementation changes
# For now we test the expected behavior


# ===== Test Fixtures =====

@pytest.fixture
def alphazero_config_yaml():
    """Sample AlphaZero YAML config"""
    return {
        'game': 'tictactoe',
        'algorithm': {
            'name': 'alphazero',
            'model': {
                'class': 'TicTacToeMLPNet',
                'hidden': 64
            },
            'iterations': 10,
            'num_sims': 50,
            'device': 'cpu'
        }
    }


@pytest.fixture
def vanilla_mcts_config_yaml():
    """Sample Vanilla MCTS YAML config (no model section) with nested structure"""
    return {
        'game': 'tictactoe',
        'algorithm': {
            'name': 'vanilla_mcts',
            'core': {
                'num_sims': 100,
                'c_exploration': 1.414,
            },
            'trainer': {
                'num_test_games': 5,
            }
        }
    }


@pytest.fixture
def temp_config_file(tmp_path):
    """Factory to create temporary config files"""
    def _create(config_dict):
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        return config_file
    return _create


# ===== load_config Tests =====

class TestLoadConfig:
    """Tests for load_config function"""

    def test_load_config_alphazero_returns_algo_name(self, alphazero_config_yaml, temp_config_file):
        """load_config should return algo_name='alphazero' for alphazero config"""
        from scripts.train import load_config

        config_file = temp_config_file(alphazero_config_yaml)
        game_name, algo_name, config, full_config = load_config(config_file)

        assert algo_name == 'alphazero'
        assert game_name == 'tictactoe'

    def test_load_config_vanilla_mcts_returns_algo_name(self, vanilla_mcts_config_yaml, temp_config_file):
        """load_config should return algo_name='vanilla_mcts' for vanilla_mcts config"""
        from scripts.train import load_config

        config_file = temp_config_file(vanilla_mcts_config_yaml)
        game_name, algo_name, config, full_config = load_config(config_file)

        assert algo_name == 'vanilla_mcts'
        assert game_name == 'tictactoe'

    def test_load_config_alphazero_returns_correct_config_type(self, alphazero_config_yaml, temp_config_file):
        """load_config should return AlphaZeroConfig for alphazero"""
        from scripts.train import load_config
        from src.algorithms.alphazero.config import AlphaZeroConfig

        config_file = temp_config_file(alphazero_config_yaml)
        game_name, algo_name, config, full_config = load_config(config_file)

        assert isinstance(config, AlphaZeroConfig)
        assert config.model_class == 'TicTacToeMLPNet'

    def test_load_config_vanilla_mcts_returns_correct_config_type(self, vanilla_mcts_config_yaml, temp_config_file):
        """load_config should return VanillaMCTSConfig for vanilla_mcts"""
        from scripts.train import load_config
        from src.algorithms.vanilla_mcts.config import VanillaMCTSConfig

        config_file = temp_config_file(vanilla_mcts_config_yaml)
        game_name, algo_name, config, full_config = load_config(config_file)

        assert isinstance(config, VanillaMCTSConfig)
        assert config.core.num_sims == 100

    def test_load_config_vanilla_mcts_no_model_required(self, vanilla_mcts_config_yaml, temp_config_file):
        """load_config should NOT require model section for vanilla_mcts"""
        from scripts.train import load_config

        # Ensure no model section
        assert 'model' not in vanilla_mcts_config_yaml['algorithm']

        config_file = temp_config_file(vanilla_mcts_config_yaml)
        # Should NOT raise an error
        game_name, algo_name, config, full_config = load_config(config_file)

        assert algo_name == 'vanilla_mcts'
        # Config should not have model_class attribute (or it should be None/empty)
        assert not hasattr(config, 'model_class') or not config.model_class


# ===== Model Creation Tests =====

class TestModelCreation:
    """Tests for conditional model creation in main()"""

    def test_config_with_model_class_creates_model(self, alphazero_config_yaml, temp_config_file):
        """Config with model_class should trigger model creation"""
        from scripts.train import load_config

        config_file = temp_config_file(alphazero_config_yaml)
        game_name, algo_name, config, full_config = load_config(config_file)

        # AlphaZero config should have model_class
        assert hasattr(config, 'model_class')
        assert config.model_class == 'TicTacToeMLPNet'

    def test_config_without_model_class_no_model(self, vanilla_mcts_config_yaml, temp_config_file):
        """Config without model_class should not require model creation"""
        from scripts.train import load_config

        config_file = temp_config_file(vanilla_mcts_config_yaml)
        game_name, algo_name, config, full_config = load_config(config_file)

        # VanillaMCTS config should not have model_class
        has_model = hasattr(config, 'model_class') and config.model_class
        assert not has_model


# ===== Trainer Factory Tests =====

class TestTrainerFactory:
    """Tests for trainer factory selection"""

    def test_trainer_factory_uses_algo_name_alphazero(self):
        """Trainer factory should be selected based on algo_name, not hardcoded"""
        from src.algorithms.registry import AlgorithmRegistry

        factory = AlgorithmRegistry.get_trainer_factory('alphazero')
        assert callable(factory)

    def test_trainer_factory_uses_algo_name_vanilla_mcts(self):
        """Trainer factory should be available for vanilla_mcts"""
        from src.algorithms.registry import AlgorithmRegistry

        factory = AlgorithmRegistry.get_trainer_factory('vanilla_mcts')
        assert callable(factory)


# ===== Agent Config Factory Tests =====

class TestAgentConfigFactory:
    """Tests for agent config factory via registry"""

    def test_get_agent_config_factory_alphazero(self):
        """Registry should provide agent config factory for alphazero"""
        from src.algorithms.registry import AlgorithmRegistry
        from src.algorithms.alphazero.config import AlphaZeroConfig
        from src.algorithms.alphazero import AlphaZeroAgentConfig

        factory = AlgorithmRegistry.get_agent_config_factory('alphazero')
        assert callable(factory)

        # Test that factory produces correct type
        config = AlphaZeroConfig(
            model_class='TicTacToeMLPNet',
            model_kwargs={'hidden': 64},
            num_sims=50
        )
        agent_config = factory(config)
        assert isinstance(agent_config, AlphaZeroAgentConfig)

    def test_get_agent_config_factory_vanilla_mcts(self):
        """Registry should provide agent config factory for vanilla_mcts"""
        from src.algorithms.registry import AlgorithmRegistry
        from src.algorithms.vanilla_mcts.config import VanillaMCTSConfig
        from src.algorithms.vanilla_mcts.mcts import MCTSConfig
        from src.algorithms.vanilla_mcts import VanillaMCTSAgentConfig

        factory = AlgorithmRegistry.get_agent_config_factory('vanilla_mcts')
        assert callable(factory)

        # Test that factory produces correct type with composed config
        config = VanillaMCTSConfig(core=MCTSConfig(num_sims=100))
        agent_config = factory(config)
        assert isinstance(agent_config, VanillaMCTSAgentConfig)

    def test_get_agent_config_factory_unknown_raises(self):
        """Registry should raise KeyError for unknown algorithm"""
        from src.algorithms.registry import AlgorithmRegistry

        with pytest.raises(KeyError):
            AlgorithmRegistry.get_agent_config_factory('nonexistent_algo')
