"""Tests for GameRegistry."""
import pytest

from src.games.core.registry import GameRegistry
from src.games.core.game import Game
from src.games.tictactoe import TicTacToe
from src.games.connect4 import Connect4


# ===== Test Fixtures =====

class TestGame(Game):
    """Mock game for testing."""
    action_size = 4

    def reset(self):
        pass

    def to_play(self, s):
        return +1

    def legal_actions(self, s):
        pass

    def next_state(self, s, a):
        pass

    def terminal_value(self, s):
        return False, 0.0

    def encode(self, s):
        pass

    def key(self, s):
        return b''

    def symmetries(self, x, pi):
        return [(x, pi)]


@pytest.fixture
def clean_test_game_from_registry():
    """Remove test game from registry after test."""
    yield
    # Cleanup after test
    if 'test_game' in GameRegistry._registry:
        del GameRegistry._registry['test_game']
    if 'another_game' in GameRegistry._registry:
        del GameRegistry._registry['another_game']


# ===== Auto-Registration Tests =====

def test_tictactoe_auto_registered():
    """TicTacToe should be auto-registered on module load."""
    game_cls = GameRegistry.get_game('tictactoe')
    assert game_cls is TicTacToe


def test_connect4_auto_registered():
    """Connect4 should be auto-registered on module load."""
    game_cls = GameRegistry.get_game('connect4')
    assert game_cls is Connect4


# ===== Registration Tests =====

def test_register_new_game(clean_test_game_from_registry):
    """Should be able to register a new game."""
    GameRegistry.register('test_game', TestGame)

    game_cls = GameRegistry.get_game('test_game')
    assert game_cls is TestGame


def test_register_idempotent(clean_test_game_from_registry):
    """Registering same game with same class twice should succeed."""
    GameRegistry.register('test_game', TestGame)
    GameRegistry.register('test_game', TestGame)  # Should not raise

    game_cls = GameRegistry.get_game('test_game')
    assert game_cls is TestGame


def test_register_duplicate_different_class_raises(clean_test_game_from_registry):
    """Registering same game name with different class should raise ValueError."""
    class AnotherTestGame(Game):
        action_size = 5

        def reset(self):
            pass

        def to_play(self, s):
            return +1

        def legal_actions(self, s):
            pass

        def next_state(self, s, a):
            pass

        def terminal_value(self, s):
            return False, 0.0

        def encode(self, s):
            pass

        def key(self, s):
            return b''

        def symmetries(self, x, pi):
            return [(x, pi)]

    GameRegistry.register('test_game', TestGame)

    with pytest.raises(ValueError, match="already registered"):
        GameRegistry.register('test_game', AnotherTestGame)


# ===== Retrieval Tests =====

def test_get_game_returns_correct_class(clean_test_game_from_registry):
    """get_game should return the correct game class."""
    GameRegistry.register('test_game', TestGame)

    game_cls = GameRegistry.get_game('test_game')
    assert game_cls is TestGame

    # Should be able to instantiate it
    game = game_cls()
    assert isinstance(game, TestGame)


def test_get_game_unknown_raises_keyerror():
    """get_game should raise KeyError for unknown game."""
    with pytest.raises(KeyError, match="No game registered"):
        GameRegistry.get_game('nonexistent_game')


def test_get_game_error_message_shows_available_games():
    """KeyError message should list available games."""
    try:
        GameRegistry.get_game('nonexistent_game')
    except KeyError as e:
        error_msg = str(e)
        assert 'Available games:' in error_msg
        # Should list at least tictactoe and connect4
        assert 'tictactoe' in error_msg or 'connect4' in error_msg


# ===== Lazy Loading Tests =====

def test_lazy_load_tictactoe():
    """TicTacToe should be lazily loaded on first access."""
    # TicTacToe is already auto-registered, but this tests the lazy load path
    game_cls = GameRegistry.get_game('tictactoe')
    assert game_cls is TicTacToe


def test_lazy_load_connect4():
    """Connect4 should be lazily loaded on first access."""
    # Connect4 is already auto-registered, but this tests the lazy load path
    game_cls = GameRegistry.get_game('connect4')
    assert game_cls is Connect4


# ===== Listing Tests =====

def test_list_games_returns_all_registered():
    """list_games should return all registered game names."""
    games = GameRegistry.list_games()

    # At minimum, should have tictactoe and connect4
    assert 'tictactoe' in games
    assert 'connect4' in games


def test_list_games_is_sorted():
    """list_games should return sorted list."""
    games = GameRegistry.list_games()

    # Check if sorted
    assert games == sorted(games)


def test_list_games_includes_manually_registered(clean_test_game_from_registry):
    """list_games should include manually registered games."""
    GameRegistry.register('test_game', TestGame)

    games = GameRegistry.list_games()
    assert 'test_game' in games


# ===== Integration Tests =====

def test_can_instantiate_registered_games():
    """Should be able to instantiate games from registry."""
    # TicTacToe
    tictactoe_cls = GameRegistry.get_game('tictactoe')
    tictactoe = tictactoe_cls()
    assert isinstance(tictactoe, TicTacToe)
    assert tictactoe.action_size == 9

    # Connect4
    connect4_cls = GameRegistry.get_game('connect4')
    connect4 = connect4_cls()
    assert isinstance(connect4, Connect4)
    assert connect4.action_size == 7


def test_registry_works_with_custom_game(clean_test_game_from_registry):
    """Should work with custom game classes."""
    GameRegistry.register('test_game', TestGame)

    game_cls = GameRegistry.get_game('test_game')
    game = game_cls()

    assert isinstance(game, TestGame)
    assert game.action_size == 4
