"""Test AlgorithmRegistry metadata and auto-discovery."""
import pytest
from src.algorithms.registry import AlgorithmRegistry, AlgorithmMetadata


class TestAlgorithmMetadata:
    """Tests for AlgorithmMetadata dataclass and registry methods."""

    def test_metadata_dataclass_exists(self):
        """AlgorithmMetadata should be a dataclass with required fields."""
        metadata = AlgorithmMetadata()
        assert hasattr(metadata, 'requires_model')
        assert hasattr(metadata, 'requires_checkpoint')
        assert hasattr(metadata, 'checkpoint_files')

    def test_alphazero_metadata(self):
        """AlphaZero should have metadata indicating it needs model.pt."""
        metadata = AlgorithmRegistry.get_metadata('alphazero')
        assert metadata.requires_model is True
        assert metadata.requires_checkpoint is True
        assert 'model.pt' in metadata.checkpoint_files
        assert 'agent.yaml' in metadata.checkpoint_files

    def test_vanilla_mcts_metadata(self):
        """VanillaMCTS should have metadata indicating no model needed."""
        metadata = AlgorithmRegistry.get_metadata('vanilla_mcts')
        assert metadata.requires_model is False
        assert metadata.requires_checkpoint is True
        assert 'agent.yaml' in metadata.checkpoint_files

    def test_random_metadata(self):
        """Random should have metadata indicating no checkpoint needed."""
        metadata = AlgorithmRegistry.get_metadata('random')
        assert metadata.requires_model is False
        assert metadata.requires_checkpoint is False
        assert metadata.checkpoint_files == ()

    def test_get_all_algorithms(self):
        """get_all_algorithms should return all registered algorithms."""
        algorithms = AlgorithmRegistry.get_all_algorithms()
        assert 'alphazero' in algorithms
        assert 'vanilla_mcts' in algorithms
        assert 'random' in algorithms


class TestAutoDiscovery:
    """Tests for algorithm auto-discovery from directory structure."""

    def test_algorithms_discovered_from_directories(self):
        """Algorithms should be discovered by scanning src/algorithms/*/."""
        # Clear registry and re-discover
        AlgorithmRegistry._registry.clear()
        AlgorithmRegistry._metadata.clear()
        AlgorithmRegistry._discover_algorithms()

        assert 'alphazero' in AlgorithmRegistry._registry
        assert 'vanilla_mcts' in AlgorithmRegistry._registry
        assert 'random' in AlgorithmRegistry._registry
