"""Tests for vanilla_mcts factory functions with composed configs."""

import pytest

from src.games.tictactoe import TicTacToe
from src.algorithms.vanilla_mcts.config import VanillaMCTSConfig
from src.algorithms.vanilla_mcts.mcts import MCTSConfig
from src.algorithms.vanilla_mcts.trainer import Trainer
from src.algorithms.vanilla_mcts.factories import (
    create_vanilla_mcts_trainer,
    create_vanilla_mcts_agent_config,
)
from src.algorithms.vanilla_mcts.agent_config import VanillaMCTSAgentConfig
from src.algorithms.shared.trainer_args import TrainerArgs


class TestCreateVanillaMCTSTrainer:
    """Test create_vanilla_mcts_trainer factory with composed config."""

    def test_returns_trainer(self):
        """Factory should return a Trainer instance."""
        game = TicTacToe()
        config = VanillaMCTSConfig(
            core=MCTSConfig(num_sims=10),
            trainer=TrainerArgs(num_test_games=1),
        )
        trainer = create_vanilla_mcts_trainer(game, None, config)
        assert isinstance(trainer, Trainer)

    def test_uses_embedded_core_config(self):
        """Trainer should use core config from composed config."""
        game = TicTacToe()
        config = VanillaMCTSConfig(
            core=MCTSConfig(num_sims=123, c_exploration=2.5),
            trainer=TrainerArgs(num_test_games=1),
        )
        trainer = create_vanilla_mcts_trainer(game, None, config)

        assert trainer.mcts.cfg.num_sims == 123
        assert trainer.mcts.cfg.c_exploration == 2.5

    def test_uses_embedded_trainer_args(self):
        """Trainer should use trainer args from composed config."""
        game = TicTacToe()
        config = VanillaMCTSConfig(
            core=MCTSConfig(num_sims=10),
            trainer=TrainerArgs(num_test_games=5, verbose=False),
        )
        trainer = create_vanilla_mcts_trainer(game, None, config)

        assert trainer.args.num_test_games == 5
        assert trainer.args.verbose is False


class TestCreateVanillaMCTSAgentConfig:
    """Test create_vanilla_mcts_agent_config factory with composed config."""

    def test_returns_agent_config(self):
        """Factory should return VanillaMCTSAgentConfig."""
        config = VanillaMCTSConfig(
            core=MCTSConfig(num_sims=100),
        )
        agent_config = create_vanilla_mcts_agent_config(config)
        assert isinstance(agent_config, VanillaMCTSAgentConfig)

    def test_embeds_core_config(self):
        """Agent config should embed the core config from training config."""
        config = VanillaMCTSConfig(
            core=MCTSConfig(
                num_sims=456,
                c_exploration=1.7,
                max_rollout_depth=30,
                rollout_seed=77,
                illegal_action_penalty=1e5,
            ),
            trainer=TrainerArgs(device="cuda"),
        )
        agent_config = create_vanilla_mcts_agent_config(config)

        assert agent_config.mcts.num_sims == 456
        assert agent_config.mcts.c_exploration == 1.7
        assert agent_config.mcts.max_rollout_depth == 30
        assert agent_config.mcts.rollout_seed == 77
        assert agent_config.mcts.illegal_action_penalty == 1e5
        assert agent_config.device == "cuda"
