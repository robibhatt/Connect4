"""Tests for ModelRegistry."""
import pytest
import torch.nn as nn

from src.models.registry import ModelRegistry
from src.games.tictactoe import TicTacToe
from src.games.connect4 import Connect4


# ===== Test Fixtures =====

class TestGameMLPNet(nn.Module):
    """Mock model for testing - follows naming convention."""
    action_size = 4

    def forward(self, x):
        return x, x


class AnotherTestGameMLPNet(nn.Module):
    """Different model with same name for duplicate testing."""
    action_size = 5

    def forward(self, x):
        return x, x


class InvalidModelName(nn.Module):
    """Model with invalid name (doesn't follow pattern)."""

    def forward(self, x):
        return x, x


@pytest.fixture
def clean_test_models_from_registry():
    """Remove test models from registry after test."""
    yield
    # Cleanup after test
    if 'TestGameMLPNet' in ModelRegistry._registry:
        del ModelRegistry._registry['TestGameMLPNet']
    if 'testgame' in ModelRegistry._game_to_models:
        del ModelRegistry._game_to_models['testgame']
    if 'AnotherTestGameMLPNet' in ModelRegistry._registry:
        del ModelRegistry._registry['AnotherTestGameMLPNet']
    if 'anothertestgame' in ModelRegistry._game_to_models:
        del ModelRegistry._game_to_models['anothertestgame']


# ===== Auto-Registration Tests =====

def test_tictactoe_mlp_auto_registered():
    """TicTacToeMLPNet should be auto-registered on module load."""
    model_cls = ModelRegistry.get_model('TicTacToeMLPNet')
    assert model_cls.__name__ == 'TicTacToeMLPNet'


def test_connect4_mlp_auto_registered():
    """Connect4MLPNet should be auto-registered on module load."""
    model_cls = ModelRegistry.get_model('Connect4MLPNet')
    assert model_cls.__name__ == 'Connect4MLPNet'


# ===== Registration Tests =====

def test_register_new_model(clean_test_models_from_registry):
    """Should be able to register a new model."""
    ModelRegistry.register(TestGameMLPNet)

    model_cls = ModelRegistry.get_model('TestGameMLPNet')
    assert model_cls is TestGameMLPNet


def test_register_extracts_game_name(clean_test_models_from_registry):
    """Registration should extract game name from class name."""
    ModelRegistry.register(TestGameMLPNet)

    # Should be tracked under 'testgame' (lowercase)
    models = ModelRegistry.get_models_for_game('testgame')
    assert 'TestGameMLPNet' in models


def test_register_idempotent(clean_test_models_from_registry):
    """Registering same model twice should succeed."""
    ModelRegistry.register(TestGameMLPNet)
    ModelRegistry.register(TestGameMLPNet)  # Should not raise

    model_cls = ModelRegistry.get_model('TestGameMLPNet')
    assert model_cls is TestGameMLPNet


def test_register_duplicate_different_class_raises(clean_test_models_from_registry):
    """Registering same name with different class should raise ValueError."""
    # Register first model
    ModelRegistry.register(TestGameMLPNet)

    # Try to register different class with same name - need to bypass Python's
    # namespace, so we'll modify the class name
    AnotherTestGameMLPNet.__name__ = 'TestGameMLPNet'

    with pytest.raises(ValueError, match="already registered"):
        ModelRegistry.register(AnotherTestGameMLPNet)

    # Restore original name
    AnotherTestGameMLPNet.__name__ = 'AnotherTestGameMLPNet'


def test_register_invalid_name_raises(clean_test_models_from_registry):
    """Registration with invalid model name should raise ValueError."""
    with pytest.raises(ValueError, match="does not follow pattern"):
        ModelRegistry.register(InvalidModelName)


# ===== Game Name Extraction Tests =====

def test_game_name_extraction_connect4(clean_test_models_from_registry):
    """Should correctly extract 'connect4' from Connect4MLPNet."""
    # Connect4MLPNet is already registered, check the mapping
    models = ModelRegistry.get_models_for_game('connect4')
    assert 'Connect4MLPNet' in models


def test_game_name_extraction_tictactoe(clean_test_models_from_registry):
    """Should correctly extract 'tictactoe' from TicTacToeMLPNet."""
    # TicTacToeMLPNet is already registered
    models = ModelRegistry.get_models_for_game('tictactoe')
    assert 'TicTacToeMLPNet' in models


# ===== Retrieval by Class Name Tests =====

def test_get_model_returns_correct_class(clean_test_models_from_registry):
    """get_model should return the correct model class."""
    ModelRegistry.register(TestGameMLPNet)

    model_cls = ModelRegistry.get_model('TestGameMLPNet')
    assert model_cls is TestGameMLPNet

    # Should be able to instantiate it
    model = model_cls()
    assert isinstance(model, TestGameMLPNet)


def test_get_model_unknown_raises_keyerror(clean_test_models_from_registry):
    """get_model should raise KeyError for unknown model."""
    with pytest.raises(KeyError, match="No model registered"):
        ModelRegistry.get_model('NonexistentModel')


def test_get_model_error_message_shows_available_models():
    """KeyError message should list available models."""
    try:
        ModelRegistry.get_model('NonexistentModel')
    except KeyError as e:
        error_msg = str(e)
        assert 'Available models:' in error_msg


# ===== Retrieval by Game Tests =====

def test_get_models_for_game_returns_all_models():
    """get_models_for_game should return all models for a game."""
    # TicTacToe should have at least TicTacToeMLPNet
    models = ModelRegistry.get_models_for_game('tictactoe')
    assert 'TicTacToeMLPNet' in models

    # Connect4 should have at least Connect4MLPNet
    models = ModelRegistry.get_models_for_game('connect4')
    assert 'Connect4MLPNet' in models


def test_get_models_for_game_returns_sorted():
    """get_models_for_game should return sorted list."""
    models = ModelRegistry.get_models_for_game('connect4')
    assert models == sorted(models)


def test_get_models_for_game_empty_for_unknown_game():
    """get_models_for_game should return empty list for unknown game."""
    models = ModelRegistry.get_models_for_game('nonexistent_game')
    assert models == []


# ===== Reverse Lookup Tests =====

def test_get_game_for_model_returns_correct_game():
    """get_game_for_model should return the correct game name."""
    from src.games.tictactoe.models.mlp import TicTacToeMLPNet
    model = TicTacToeMLPNet()

    game_name = ModelRegistry.get_game_for_model(model)
    assert game_name == 'tictactoe'


def test_get_game_for_model_none_for_unregistered():
    """get_game_for_model should return None for unregistered model."""
    model = InvalidModelName()
    game_name = ModelRegistry.get_game_for_model(model)
    assert game_name is None


# ===== Validation Tests =====

def test_validate_compatibility_action_size_match():
    """validate_compatibility should pass for matching action sizes."""
    from src.games.tictactoe.models.mlp import TicTacToeMLPNet
    game = TicTacToe()
    model = TicTacToeMLPNet()

    # Should not raise
    ModelRegistry.validate_compatibility(game, model)


def test_validate_compatibility_action_size_mismatch_raises():
    """validate_compatibility should raise for mismatched action sizes."""
    from src.games.connect4.models.mlp import Connect4MLPNet
    tictactoe = TicTacToe()
    connect4_model = Connect4MLPNet()

    # Connect4 has 7 actions, TicTacToe has 9
    with pytest.raises(ValueError, match="action_size"):
        ModelRegistry.validate_compatibility(tictactoe, connect4_model)


def test_validate_compatibility_game_name_match():
    """validate_compatibility should pass for matching game names."""
    from src.games.connect4.models.mlp import Connect4MLPNet
    game = Connect4()
    model = Connect4MLPNet()

    # Should not raise
    ModelRegistry.validate_compatibility(game, model)


def test_validate_compatibility_game_name_mismatch_raises():
    """validate_compatibility should raise for mismatched game names."""
    from src.games.tictactoe.models.mlp import TicTacToeMLPNet
    connect4 = Connect4()
    tictactoe_model = TicTacToeMLPNet()

    # Model name doesn't start with game name (also action size mismatch)
    # The action_size check happens first, so we'll match that error
    with pytest.raises(ValueError, match="action_size"):
        ModelRegistry.validate_compatibility(connect4, tictactoe_model)


# ===== Lazy Loading Tests =====

def test_lazy_load_by_class_name():
    """Should lazily load models by class name."""
    # Access TicTacToeMLPNet, should trigger lazy load if not already loaded
    model_cls = ModelRegistry.get_model('TicTacToeMLPNet')
    assert model_cls.__name__ == 'TicTacToeMLPNet'


def test_lazy_load_by_game_name():
    """Should lazily load models by game name."""
    # Access models for 'connect4', should trigger lazy load if not already loaded
    models = ModelRegistry.get_models_for_game('connect4')
    assert 'Connect4MLPNet' in models


# ===== Listing Tests =====

def test_list_models_returns_all_registered():
    """list_models should return all registered model class names."""
    models = ModelRegistry.list_models()

    # Should include at least the auto-registered models
    assert 'TicTacToeMLPNet' in models
    assert 'Connect4MLPNet' in models


def test_list_models_is_sorted():
    """list_models should return sorted list."""
    models = ModelRegistry.list_models()
    assert models == sorted(models)


def test_list_games_returns_all_games_with_models():
    """list_games should return all games that have models."""
    games = ModelRegistry.list_games()

    # Should include at least tictactoe and connect4
    assert 'tictactoe' in games
    assert 'connect4' in games


def test_list_games_is_sorted():
    """list_games should return sorted list."""
    games = ModelRegistry.list_games()
    assert games == sorted(games)


# ===== Integration Tests =====

def test_can_instantiate_registered_models():
    """Should be able to instantiate models from registry."""
    # TicTacToeMLPNet
    model_cls = ModelRegistry.get_model('TicTacToeMLPNet')
    model = model_cls()
    assert model.action_size == 9

    # Connect4MLPNet
    model_cls = ModelRegistry.get_model('Connect4MLPNet')
    model = model_cls()
    assert model.action_size == 7


def test_multiple_models_per_game(clean_test_models_from_registry):
    """Should support multiple model architectures per game."""
    # Create another model for testgame
    class TestGameCNNNet(nn.Module):
        action_size = 4

        def forward(self, x):
            return x, x

    # Register both models
    ModelRegistry.register(TestGameMLPNet)
    ModelRegistry.register(TestGameCNNNet)

    # Should have both under 'testgame'
    models = ModelRegistry.get_models_for_game('testgame')
    assert 'TestGameMLPNet' in models
    assert 'TestGameCNNNet' in models
    assert len(models) == 2

    # Cleanup
    if 'TestGameCNNNet' in ModelRegistry._registry:
        del ModelRegistry._registry['TestGameCNNNet']
    if 'testgame' in ModelRegistry._game_to_models:
        ModelRegistry._game_to_models['testgame'].discard('TestGameCNNNet')
