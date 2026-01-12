"""Tests for MCGS factory functions."""

import pytest

from src.algorithms.mcgs.factories import (
    create_mcgs_trainer,
    create_mcgs_agent_config
)
from src.algorithms.mcgs.config import MCGSConfig
from src.algorithms.mcgs.mcgs import MCGSCoreConfig
from src.algorithms.mcgs.trainer import Trainer
from src.algorithms.mcgs.agent_config import MCGSAgentConfig
from src.algorithms.shared.trainer_args import TrainerArgs


class TestCreateMCGSTrainer:
    """Tests for create_mcgs_trainer factory function."""

    def test_create_mcgs_trainer_returns_trainer(self, tictactoe_game):
        """Factory should return a Trainer instance."""
        config = MCGSConfig(
            core=MCGSCoreConfig(num_sims=10),
            trainer=TrainerArgs(num_test_games=1),
        )

        trainer = create_mcgs_trainer(
            game=tictactoe_game,
            model=None,  # No model needed
            config=config
        )

        assert isinstance(trainer, Trainer)

    def test_create_mcgs_trainer_ignores_model(self, tictactoe_game):
        """Factory should ignore model parameter."""
        config = MCGSConfig(
            core=MCGSCoreConfig(num_sims=10),
            trainer=TrainerArgs(num_test_games=1),
        )

        # Should work with None
        trainer1 = create_mcgs_trainer(tictactoe_game, None, config)
        assert trainer1 is not None

        # Should also work with some dummy value
        trainer2 = create_mcgs_trainer(tictactoe_game, "ignored", config)
        assert trainer2 is not None

    def test_create_mcgs_trainer_uses_config_values(self, tictactoe_game):
        """Trainer should use config values."""
        config = MCGSConfig(
            core=MCGSCoreConfig(num_sims=50, c_exploration=2.0),
            trainer=TrainerArgs(num_test_games=5),
        )

        trainer = create_mcgs_trainer(tictactoe_game, None, config)

        assert trainer.args.num_test_games == 5
        assert trainer.mcgs.cfg.num_sims == 50
        assert trainer.mcgs.cfg.c_exploration == 2.0


class TestCreateMCGSAgentConfig:
    """Tests for create_mcgs_agent_config factory function."""

    def test_create_mcgs_agent_config_returns_agent_config(self):
        """Factory should return MCGSAgentConfig instance."""
        config = MCGSConfig(
            core=MCGSCoreConfig(num_sims=100),
        )

        agent_config = create_mcgs_agent_config(config)

        assert isinstance(agent_config, MCGSAgentConfig)

    def test_create_mcgs_agent_config_embeds_core(self):
        """Agent config should embed core config from training config."""
        config = MCGSConfig(
            core=MCGSCoreConfig(
                num_sims=200,
                c_exploration=1.5,
                max_rollout_depth=75,
                rollout_seed=42,
                illegal_action_penalty=1e8,
            ),
            trainer=TrainerArgs(device='mps'),
        )

        agent_config = create_mcgs_agent_config(config)

        assert agent_config.mcgs.num_sims == 200
        assert agent_config.mcgs.c_exploration == 1.5
        assert agent_config.mcgs.max_rollout_depth == 75
        assert agent_config.mcgs.rollout_seed == 42
        assert agent_config.mcgs.illegal_action_penalty == 1e8
        assert agent_config.device == 'mps'
