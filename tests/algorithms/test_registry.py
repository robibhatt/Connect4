"""
Tests for AlgorithmRegistry.

Verifies:
- Registration and retrieval of algorithms
- Config class access
- Trainer factory access
- Lazy loading
- Error handling for missing algorithms
"""

import pytest
from unittest.mock import Mock

from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.vanilla_mcts.config import VanillaMCTSConfig


# ===== Registration Tests =====

def test_register_algorithm():
    """Test basic algorithm registration"""
    # Create mock config class and factory
    MockConfig = Mock()
    mock_factory = Mock()

    # Register the algorithm
    AlgorithmRegistry.register('test_algo', MockConfig, mock_factory)

    # Should be able to retrieve it
    retrieved_config = AlgorithmRegistry.get_config_class('test_algo')
    retrieved_factory = AlgorithmRegistry.get_trainer_factory('test_algo')

    assert retrieved_config == MockConfig
    assert retrieved_factory == mock_factory


def test_register_duplicate_algorithm_same_class():
    """Test idempotent registration (same class twice should succeed)"""
    MockConfig = Mock()
    mock_factory = Mock()

    # Register twice with same classes
    AlgorithmRegistry.register('test_algo2', MockConfig, mock_factory)
    AlgorithmRegistry.register('test_algo2', MockConfig, mock_factory)

    # Should succeed silently (idempotent)
    retrieved_config = AlgorithmRegistry.get_config_class('test_algo2')
    assert retrieved_config == MockConfig


def test_register_duplicate_algorithm_different_class():
    """Test conflicting duplicate registration should raise ValueError"""
    MockConfig1 = Mock()
    MockConfig2 = Mock()
    mock_factory = Mock()

    # Register first time
    AlgorithmRegistry.register('test_algo3', MockConfig1, mock_factory)

    # Register again with different config class should raise
    with pytest.raises(ValueError, match="already registered"):
        AlgorithmRegistry.register('test_algo3', MockConfig2, mock_factory)


# ===== Retrieval Tests =====

def test_get_config_class():
    """Test retrieving config class by algorithm name"""
    # VanillaMCTS should be auto-registered
    config_class = AlgorithmRegistry.get_config_class('vanilla_mcts')
    assert config_class == VanillaMCTSConfig


def test_get_trainer_factory():
    """Test retrieving trainer factory by algorithm name"""
    # VanillaMCTS should be auto-registered
    factory = AlgorithmRegistry.get_trainer_factory('vanilla_mcts')
    assert callable(factory)


def test_list_algorithms():
    """Test listing all registered algorithms"""
    algorithms = AlgorithmRegistry.list_algorithms()
    assert 'vanilla_mcts' in algorithms
    assert isinstance(algorithms, list)


# ===== Lazy Loading Tests =====

def test_lazy_load_vanilla_mcts():
    """Test that vanilla_mcts is automatically registered on first access"""
    # This test verifies lazy loading works
    # Even if not explicitly registered, accessing should trigger registration
    config_class = AlgorithmRegistry.get_config_class('vanilla_mcts')
    assert config_class is not None


def test_unknown_algorithm_raises_keyerror():
    """Test that unknown algorithm raises helpful KeyError"""
    with pytest.raises(KeyError) as exc_info:
        AlgorithmRegistry.get_config_class('nonexistent_algo')

    # Should include available algorithms in error message
    assert 'nonexistent_algo' in str(exc_info.value)
    assert 'Available algorithms' in str(exc_info.value)


# ===== Integration Tests =====

def test_vanilla_mcts_auto_registered():
    """Test that vanilla_mcts is automatically registered on import"""
    # After importing registry, 'vanilla_mcts' should be in list
    algorithms = AlgorithmRegistry.list_algorithms()
    assert 'vanilla_mcts' in algorithms
