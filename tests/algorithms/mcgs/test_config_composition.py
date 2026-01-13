"""Tests for MCGSConfig composition structure."""

import pytest
from dataclasses import asdict

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


class TestMCGSConfigSerialization:
    """Test MCGSConfig serialization to/from dict."""

    def test_to_dict_produces_nested_structure(self):
        """to_dict should produce a nested dict structure."""
        config = MCGSConfig(
            core=MCGSCoreConfig(num_sims=500),
            trainer=TrainerArgs(verbose=False),
        )
        d = config.to_dict()

        assert "core" in d
        assert "trainer" in d
        assert d["core"]["num_sims"] == 500
        assert d["trainer"]["verbose"] is False

    def test_from_dict_nested_format(self):
        """from_dict should load nested dict structure."""
        d = {
            "core": {
                "num_sims": 250,
                "c_exploration": 1.5,
                "max_rollout_depth": 50,
                "rollout_seed": 42,
                "illegal_action_penalty": 1e6,
                "batch_size": 64,
            },
            "trainer": {
                "num_test_games": 5,
                "device": "cuda",
                "random_seed": 123,
                "verbose": False,
            },
        }
        config = MCGSConfig.from_dict(d)

        assert config.core.num_sims == 250
        assert config.core.c_exploration == 1.5
        assert config.core.max_rollout_depth == 50
        assert config.core.rollout_seed == 42
        assert config.core.illegal_action_penalty == 1e6
        assert config.core.batch_size == 64
        assert config.trainer.num_test_games == 5
        assert config.trainer.device == "cuda"
        assert config.trainer.random_seed == 123
        assert config.trainer.verbose is False

    def test_roundtrip(self):
        """Config should survive serialization roundtrip."""
        original = MCGSConfig(
            core=MCGSCoreConfig(num_sims=777, c_exploration=1.8),
            trainer=TrainerArgs(num_test_games=15, random_seed=99),
        )
        d = original.to_dict()
        restored = MCGSConfig.from_dict(d)

        assert restored.core.num_sims == original.core.num_sims
        assert restored.core.c_exploration == original.core.c_exploration
        assert restored.trainer.num_test_games == original.trainer.num_test_games
        assert restored.trainer.random_seed == original.trainer.random_seed

    def test_from_dict_with_partial_nested(self):
        """from_dict should handle partial nested dicts with defaults."""
        d = {
            "core": {"num_sims": 100},
            "trainer": {"verbose": False},
        }
        config = MCGSConfig.from_dict(d)

        # Specified values
        assert config.core.num_sims == 100
        assert config.trainer.verbose is False

        # Defaults for unspecified values
        assert config.core.c_exploration == 1.414
        assert config.core.batch_size == 1
        assert config.trainer.num_test_games == 10
