"""Tests for MCGSConfig composition structure."""

import pytest

from src.algorithms.mcgs.config import MCGSConfig
from src.algorithms.mcgs.mcgs import MCGSCoreConfig
from src.algorithms.shared.trainer_args import TrainerArgs


class TestMCGSConfigComposition:
    """Test that MCGSConfig uses composition with embedded sub-configs."""

    def test_mcgs_config_has_core_subconfig(self):
        """MCGSConfig should have an embedded MCGSCoreConfig."""
        config = MCGSConfig()
        assert hasattr(config, "core")
        assert isinstance(config.core, MCGSCoreConfig)

    def test_mcgs_config_has_trainer_subconfig(self):
        """MCGSConfig should have an embedded TrainerArgs."""
        config = MCGSConfig()
        assert hasattr(config, "trainer")
        assert isinstance(config.trainer, TrainerArgs)

    def test_nested_field_access_core(self):
        """Can access core config fields via nested path."""
        config = MCGSConfig()
        assert config.core.num_sims == 1000
        assert config.core.c_exploration == 1.414
        assert config.core.max_rollout_depth is None
        assert config.core.rollout_seed is None
        assert config.core.illegal_action_penalty == 1e9
        assert config.core.batch_size == 1

    def test_nested_field_access_trainer(self):
        """Can access trainer args fields via nested path."""
        config = MCGSConfig()
        assert config.trainer.num_test_games == 10
        assert config.trainer.device == "cpu"
        assert config.trainer.random_seed is None
        assert config.trainer.verbose is True

    def test_can_construct_with_custom_core(self):
        """Can construct MCGSConfig with custom core config."""
        custom_core = MCGSCoreConfig(num_sims=500, c_exploration=2.0)
        config = MCGSConfig(core=custom_core)
        assert config.core.num_sims == 500
        assert config.core.c_exploration == 2.0

    def test_can_construct_with_custom_trainer(self):
        """Can construct MCGSConfig with custom trainer args."""
        custom_trainer = TrainerArgs(num_test_games=20, verbose=False)
        config = MCGSConfig(trainer=custom_trainer)
        assert config.trainer.num_test_games == 20
        assert config.trainer.verbose is False
