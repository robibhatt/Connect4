"""Tests for MCGS core algorithm."""

import pytest
import numpy as np

from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig, Node


class TestNode:
    """Tests for MCGS Node class."""

    def test_node_creation(self, tictactoe_game):
        """Node should initialize with state and action count."""
        state = tictactoe_game.reset()
        node = Node(state=state, num_actions=9)
        assert node.state is state
        assert len(node.children) == 9
        assert node.is_expanded is False
        assert node.N == 0
        assert node.W == 0
        assert node.Q == 0


class TestMCGSCoreConfig:
    """Tests for MCGSCoreConfig dataclass."""

    def test_config_defaults(self):
        """MCGSCoreConfig should have default values."""
        cfg = MCGSCoreConfig()
        assert cfg.num_sims == 1000
        assert cfg.c_exploration == pytest.approx(1.414, abs=0.001)

    def test_config_custom_values(self):
        """MCGSCoreConfig should accept custom values."""
        cfg = MCGSCoreConfig(num_sims=50, c_exploration=2.0)
        assert cfg.num_sims == 50
        assert cfg.c_exploration == 2.0


class TestMCGS:
    """Tests for MCGS algorithm class."""

    def test_mcgs_instantiation(self, tictactoe_game):
        """MCGS should instantiate with game and config."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        assert mcgs.game is tictactoe_game
        assert mcgs.cfg is cfg

    def test_mcgs_instantiation_default_config(self, tictactoe_game):
        """MCGS should use default config when none provided."""
        mcgs = MCGS(game=tictactoe_game)
        assert mcgs.cfg.num_sims == 1000

    def test_mcgs_run_returns_counts(self, tictactoe_game):
        """MCGS.run() should return visit counts array."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        state = tictactoe_game.reset()

        counts = mcgs.run(root=state)

        assert isinstance(counts, np.ndarray)
        assert len(counts) == tictactoe_game.action_size
        assert counts.sum() > 0  # Some simulations ran

    def test_mcgs_play_move_returns_valid_action(self, tictactoe_game):
        """MCGS.play_move() should return a valid action index."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        state = tictactoe_game.reset()

        action = mcgs.play_move(s=state)

        assert isinstance(action, (int, np.integer))
        assert 0 <= action < tictactoe_game.action_size
        # Verify action is legal
        legal = tictactoe_game.legal_actions(state)
        assert legal[action]

    def test_mcgs_play_move_deterministic(self, tictactoe_game):
        """Deterministic mode should always return same action for same state."""
        cfg = MCGSCoreConfig(num_sims=50, rollout_seed=42)
        rng = np.random.default_rng(42)

        mcgs1 = MCGS(game=tictactoe_game, cfg=cfg, rng=rng)
        state = tictactoe_game.reset()
        action1 = mcgs1.play_move(s=state, deterministic=True)

        rng2 = np.random.default_rng(42)
        mcgs2 = MCGS(game=tictactoe_game, cfg=cfg, rng=rng2)
        action2 = mcgs2.play_move(s=state, deterministic=True)

        assert action1 == action2

    def test_mcgs_clear(self, tictactoe_game):
        """MCGS.clear() should reset the node cache."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        state = tictactoe_game.reset()

        # Run to populate cache
        mcgs.run(root=state)

        # Clear
        mcgs.clear()

        assert len(mcgs.nodes) == 0

    def test_mcgs_select_action_deterministic(self, tictactoe_game):
        """select_action with deterministic=True should return argmax."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)

        pi = np.array([1, 5, 3, 0, 2, 0, 0, 0, 0])
        action = mcgs.select_action(pi, deterministic=True)

        assert action == 1  # Index of max value

    def test_mcgs_with_connect4(self, connect4_game):
        """MCGS should work with Connect4 game too."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=connect4_game, cfg=cfg)
        state = connect4_game.reset()

        action = mcgs.play_move(s=state)

        assert 0 <= action < connect4_game.action_size
