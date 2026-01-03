"""Tests for Agent ABC and RandomAgent."""

import pytest
import numpy as np
from unittest.mock import Mock

from src.agents.agent import Agent, RandomAgent
from src.games.core.game import Game


# ===== Agent ABC Tests =====

def test_cannot_instantiate_agent_directly(tictactoe_game):
    """Agent is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        Agent(tictactoe_game)


def test_agent_stores_game_reference(tictactoe_game):
    """Agent subclass should store game reference."""
    class TestAgent(Agent):
        def act(self, s):
            return 0

    agent = TestAgent(tictactoe_game)
    assert agent.game is tictactoe_game


def test_incomplete_subclass_raises():
    """Subclass without implementing act() should raise TypeError."""
    class IncompleteAgent(Agent):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteAgent(Mock(spec=Game))


# ===== RandomAgent Basic Behavior =====

def test_random_agent_returns_legal_action(random_agent_tictactoe, tictactoe_game):
    """RandomAgent should only return legal actions."""
    state = tictactoe_game.reset()
    action = random_agent_tictactoe.act(state)
    legal_actions = tictactoe_game.legal_actions(state)
    assert legal_actions[action], f"Action {action} should be legal"
    assert 0 <= action < len(legal_actions), f"Action {action} out of bounds"


def test_random_agent_deterministic_with_seed(tictactoe_game):
    """Two RandomAgents with same seed should produce same sequence."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    agent1 = RandomAgent(tictactoe_game, rng1)
    agent2 = RandomAgent(tictactoe_game, rng2)

    state = tictactoe_game.reset()

    # Both agents should return same action sequence
    for _ in range(5):
        action1 = agent1.act(state)
        action2 = agent2.act(state)
        assert action1 == action2, "Agents with same seed should produce same actions"
        # Advance state
        state = tictactoe_game.next_state(state, action1)
        if tictactoe_game.terminal_value(state)[0]:
            break


def test_random_agent_uniform_distribution(random_agent_tictactoe, tictactoe_game):
    """RandomAgent should sample uniformly over many trials."""
    state = tictactoe_game.reset()
    num_trials = 1000

    # Collect actions
    actions = [random_agent_tictactoe.act(state) for _ in range(num_trials)]

    # Count frequency of each legal action
    legal_actions = np.flatnonzero(tictactoe_game.legal_actions(state))
    counts = {a: actions.count(a) for a in legal_actions}

    # Each legal action should be chosen roughly equally (within 20% tolerance)
    expected_count = num_trials / len(legal_actions)
    for action, count in counts.items():
        ratio = count / expected_count
        assert 0.8 < ratio < 1.2, \
            f"Action {action} chosen {count} times (expected ~{expected_count:.0f}), ratio {ratio:.2f}"


# ===== RandomAgent Edge Cases =====

def test_random_agent_single_legal_action(tictactoe_game):
    """With one legal action, RandomAgent should be deterministic."""
    # Create a state with only one legal move
    # TicTacToe initial state has 9 legal moves, so we need to play some moves
    # Let's manually create a nearly full board
    state = tictactoe_game.reset()

    # Play moves to create a board with only one legal action
    # X X _
    # O O _
    # _ _ _
    moves = [0, 3, 1, 4]  # This creates a board with 5 empty spaces
    for move in moves:
        state = tictactoe_game.next_state(state, move)

    # Now find a state with just one legal action by playing more
    # Continue until we get close to a full board
    rng = np.random.default_rng(123)
    agent = RandomAgent(tictactoe_game, rng)

    # Play until we have only one legal action
    while True:
        legal = tictactoe_game.legal_actions(state)
        legal_count = np.sum(legal)

        if legal_count == 1:
            # Found a state with one legal action
            legal_action = int(np.flatnonzero(legal)[0])

            # Agent should always return this action
            for _ in range(10):
                action = agent.act(state)
                assert action == legal_action, "With one legal action, should always return it"
            break

        if legal_count == 0 or tictactoe_game.terminal_value(state)[0]:
            # Game ended, try a different sequence
            pytest.skip("Could not create state with exactly one legal action")
            break

        # Continue playing
        action = agent.act(state)
        state = tictactoe_game.next_state(state, action)


def test_random_agent_raises_on_no_legal_actions(tictactoe_game):
    """RandomAgent should raise RuntimeError when no legal actions."""
    rng = np.random.default_rng(42)
    agent = RandomAgent(tictactoe_game, rng)

    # Mock the game to return no legal actions
    mock_game = Mock(spec=Game)
    mock_game.legal_actions = Mock(return_value=np.zeros(9, dtype=bool))

    # Temporarily replace agent's game
    agent.game = mock_game

    with pytest.raises(RuntimeError, match="No legal actions"):
        agent.act(Mock())  # State doesn't matter with mocked game


def test_random_agent_works_with_both_games(random_agent_tictactoe, random_agent_connect4,
                                            tictactoe_game, connect4_game):
    """RandomAgent should work with both TicTacToe and Connect4."""
    # Test TicTacToe
    state_ttt = tictactoe_game.reset()
    action_ttt = random_agent_tictactoe.act(state_ttt)
    assert tictactoe_game.legal_actions(state_ttt)[action_ttt]

    # Test Connect4
    state_c4 = connect4_game.reset()
    action_c4 = random_agent_connect4.act(state_c4)
    assert connect4_game.legal_actions(state_c4)[action_c4]


# ===== RandomAgent RNG Control =====

def test_random_agent_uses_provided_rng(tictactoe_game):
    """RandomAgent should use provided RNG for reproducibility."""
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)

    agent1 = RandomAgent(tictactoe_game, rng1)
    agent2 = RandomAgent(tictactoe_game, rng2)

    state = tictactoe_game.reset()

    # Should produce identical sequences
    for _ in range(3):
        assert agent1.act(state) == agent2.act(state)


def test_random_agent_default_rng_creates_new(tictactoe_game):
    """Default RNG should not share state between agents."""
    agent1 = RandomAgent(tictactoe_game)
    agent2 = RandomAgent(tictactoe_game)

    # These agents have different RNG instances, so they likely produce different sequences
    # We can't guarantee they're different on the first call, but over many calls they should diverge
    state = tictactoe_game.reset()

    actions1 = [agent1.act(state) for _ in range(10)]
    actions2 = [agent2.act(state) for _ in range(10)]

    # Very unlikely that two unseeded RNGs produce the exact same 10-action sequence
    # (probability is roughly (1/9)^10 for TicTacToe initial state)
    assert actions1 != actions2, "Two agents with default RNG should have independent state"
