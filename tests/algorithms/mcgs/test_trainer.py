"""Tests for MCGS Trainer class."""

import pytest

from src.algorithms.mcgs.trainer import Trainer, TrainerArgs
from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig
from src.algorithms.mcgs.agent import MCGSAgent


class TestTrainerArgs:
    """Tests for TrainerArgs dataclass."""

    def test_trainer_args_defaults(self):
        """TrainerArgs should have sensible defaults."""
        args = TrainerArgs()
        assert args.num_test_games == 10
        assert args.device == 'cpu'
        assert args.random_seed is None
        assert args.verbose is True

    def test_trainer_args_custom_values(self):
        """TrainerArgs should accept custom values."""
        args = TrainerArgs(
            num_test_games=5,
            device='cuda',
            random_seed=42,
            verbose=False
        )
        assert args.num_test_games == 5
        assert args.random_seed == 42
        assert args.verbose is False


class TestTrainer:
    """Tests for MCGS Trainer class."""

    def test_trainer_initialization(self, tictactoe_game):
        """Trainer should initialize with game, mcgs, and args."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        args = TrainerArgs(num_test_games=1)

        trainer = Trainer(game=tictactoe_game, mcgs=mcgs, args=args)

        assert trainer.game is tictactoe_game
        assert trainer.mcgs is mcgs
        assert trainer.args is args

    def test_trainer_run_completes(self, tictactoe_game):
        """Trainer.run() should complete without errors."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        args = TrainerArgs(num_test_games=1, verbose=False)
        trainer = Trainer(game=tictactoe_game, mcgs=mcgs, args=args)

        # Should not raise
        trainer.run()

    def test_trainer_run_with_zero_test_games(self, tictactoe_game):
        """Trainer should handle zero test games."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        args = TrainerArgs(num_test_games=0, verbose=False)
        trainer = Trainer(game=tictactoe_game, mcgs=mcgs, args=args)

        # Should not raise
        trainer.run()

    def test_trainer_create_agent(self, tictactoe_game):
        """Trainer.create_agent() should return MCGSAgent."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        args = TrainerArgs(num_test_games=1)
        trainer = Trainer(game=tictactoe_game, mcgs=mcgs, args=args)

        agent = trainer.create_agent()

        assert isinstance(agent, MCGSAgent)
        assert agent.game is tictactoe_game

    def test_trainer_with_connect4(self, connect4_game):
        """Trainer should work with Connect4."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=connect4_game, cfg=cfg)
        args = TrainerArgs(num_test_games=1, verbose=False)
        trainer = Trainer(game=connect4_game, mcgs=mcgs, args=args)

        trainer.run()  # Should not raise

    def test_trainer_random_seed(self, tictactoe_game):
        """Trainer should set random seed when provided."""
        import numpy as np

        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        args = TrainerArgs(num_test_games=0, random_seed=42, verbose=False)

        # This should set np.random.seed(42)
        trainer = Trainer(game=tictactoe_game, mcgs=mcgs, args=args)

        # Verify by checking random state is deterministic
        val1 = np.random.rand()

        # Re-create with same seed
        args2 = TrainerArgs(num_test_games=0, random_seed=42, verbose=False)
        trainer2 = Trainer(game=tictactoe_game, mcgs=mcgs, args=args2)
        val2 = np.random.rand()

        assert val1 == val2
