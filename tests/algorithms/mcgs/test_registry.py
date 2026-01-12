"""Tests for MCGS algorithm registration."""

import pytest

from src.algorithms.registry import AlgorithmRegistry


class TestMCGSRegistration:
    """Tests for MCGS registration in AlgorithmRegistry."""

    def test_mcgs_is_registered(self):
        """MCGS should be discoverable in AlgorithmRegistry."""
        assert 'mcgs' in AlgorithmRegistry.list_algorithms()

    def test_mcgs_metadata(self):
        """MCGS metadata should match vanilla MCTS pattern (no model required)."""
        metadata = AlgorithmRegistry.get_metadata('mcgs')
        assert metadata.requires_model is False
        assert metadata.requires_checkpoint is True
        assert metadata.checkpoint_files == ('agent.yaml',)

    def test_mcgs_config_class(self):
        """Should return MCGSConfig class."""
        config_cls = AlgorithmRegistry.get_config_class('mcgs')
        assert config_cls.__name__ == 'MCGSConfig'

    def test_mcgs_trainer_factory(self):
        """Should return create_mcgs_trainer function."""
        factory = AlgorithmRegistry.get_trainer_factory('mcgs')
        assert callable(factory)
        assert factory.__name__ == 'create_mcgs_trainer'

    def test_mcgs_agent_config_factory(self):
        """Should return create_mcgs_agent_config function."""
        factory = AlgorithmRegistry.get_agent_config_factory('mcgs')
        assert callable(factory)
        assert factory.__name__ == 'create_mcgs_agent_config'
