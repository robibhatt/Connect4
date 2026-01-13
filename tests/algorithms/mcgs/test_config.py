"""Tests for MCGS configuration classes."""

import pytest

from src.algorithms.mcgs.config import MCGSConfig
from src.algorithms.mcgs.mcgs import MCGSCoreConfig
from src.algorithms.shared.trainer_args import TrainerArgs


class TestMCGSConfig:
    """Tests for MCGSConfig unified configuration with composition."""

    def test_mcgs_config_has_trainer_via_embedded_config(self):
        """MCGSConfig should have embedded trainer with all fields."""
        config = MCGSConfig()
        assert hasattr(config, 'trainer')
        assert hasattr(config.trainer, 'num_test_games')
        assert hasattr(config.trainer, 'device')
        assert hasattr(config.trainer, 'random_seed')
        assert hasattr(config.trainer, 'verbose')

    def test_mcgs_config_has_core_via_embedded_config(self):
        """MCGSConfig should have embedded core with all fields."""
        config = MCGSConfig()
        assert hasattr(config, 'core')
        assert hasattr(config.core, 'num_sims')
        assert hasattr(config.core, 'c_exploration')
        assert hasattr(config.core, 'max_rollout_depth')
        assert hasattr(config.core, 'rollout_seed')
        assert hasattr(config.core, 'illegal_action_penalty')
        assert hasattr(config.core, 'batch_size')

    def test_mcgs_config_access_trainer_directly(self):
        """Should access TrainerArgs directly via .trainer."""
        config = MCGSConfig(
            trainer=TrainerArgs(
                num_test_games=5,
                device='cpu',
                random_seed=42,
                verbose=False
            )
        )
        assert isinstance(config.trainer, TrainerArgs)
        assert config.trainer.num_test_games == 5
        assert config.trainer.device == 'cpu'
        assert config.trainer.random_seed == 42
        assert config.trainer.verbose is False

    def test_mcgs_config_access_core_directly(self):
        """Should access MCGSCoreConfig directly via .core."""
        config = MCGSConfig(
            core=MCGSCoreConfig(
                num_sims=500,
                c_exploration=2.0,
                max_rollout_depth=50,
                batch_size=16
            )
        )
        assert isinstance(config.core, MCGSCoreConfig)
        assert config.core.num_sims == 500
        assert config.core.c_exploration == 2.0
        assert config.core.max_rollout_depth == 50
        assert config.core.batch_size == 16


class TestTrainerArgs:
    """Tests for TrainerArgs dataclass."""

    def test_trainer_args_defaults(self):
        """TrainerArgs should have sensible defaults."""
        args = TrainerArgs()
        assert args.num_test_games == 10
        assert args.device == 'cpu'
        assert args.random_seed is None
        assert args.verbose is True


class TestMCGSCoreConfig:
    """Tests for MCGSCoreConfig dataclass."""

    def test_mcgs_core_config_defaults(self):
        """MCGSCoreConfig should have sensible defaults."""
        cfg = MCGSCoreConfig()
        assert cfg.num_sims == 1000
        assert cfg.c_exploration == pytest.approx(1.414, abs=0.001)
        assert cfg.max_rollout_depth is None
        assert cfg.rollout_seed is None
        assert cfg.illegal_action_penalty == 1e9
        assert cfg.batch_size == 1
