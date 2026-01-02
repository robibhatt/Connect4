"""
Game-Model Registry for AlphaZero training.

Provides centralized mapping between games and their neural network models,
enabling dynamic model loading and validation.
"""

from __future__ import annotations
from typing import Type, Dict, Optional
import torch.nn as nn

from src.games.core.game import Game


class ModelRegistry:
    """
    Registry mapping game names to their corresponding model classes.

    Usage:
        # Get model class for a game
        model_cls = ModelRegistry.get_model('tictactoe')
        model = model_cls()

        # Register a new game-model pair
        ModelRegistry.register('my_game', MyGameNet)

        # Validate game-model compatibility
        ModelRegistry.validate_compatibility(game, model)
    """

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, game_name: str, model_class: Type[nn.Module]) -> None:
        """
        Register a model class for a game.

        Args:
            game_name: Unique game identifier (e.g., 'tictactoe')
            model_class: Neural network class for this game

        Raises:
            ValueError: If game already registered with a different model class
        """
        if game_name in cls._registry:
            if cls._registry[game_name] != model_class:
                raise ValueError(
                    f"Game '{game_name}' already registered with "
                    f"{cls._registry[game_name].__name__}, "
                    f"cannot register {model_class.__name__}"
                )
            # Already registered with same class, silently succeed
            return
        cls._registry[game_name] = model_class

    @classmethod
    def get_model(cls, game_name: str) -> Type[nn.Module]:
        """
        Get the model class for a game.

        Args:
            game_name: Game identifier

        Returns:
            Model class

        Raises:
            KeyError: If game not registered
        """
        # Lazy registration - try to register on first access
        if game_name not in cls._registry:
            cls._ensure_registered(game_name)

        if game_name not in cls._registry:
            raise KeyError(
                f"No model registered for game '{game_name}'. "
                f"Available games: {list(cls._registry.keys())}"
            )
        return cls._registry[game_name]

    @classmethod
    def _ensure_registered(cls, game_name: str) -> None:
        """Ensure a game's model is registered (lazy registration)."""
        if game_name == 'tictactoe' and game_name not in cls._registry:
            try:
                from src.games.tictactoe.model import TicTacToeNet
                cls.register('tictactoe', TicTacToeNet)
            except (ImportError, ValueError):
                pass

        if game_name == 'connect4' and game_name not in cls._registry:
            try:
                from src.games.connect4.model import Connect4Net
                cls.register('connect4', Connect4Net)
            except (ImportError, ValueError):
                pass

    @classmethod
    def get_game_for_model(cls, model: nn.Module) -> Optional[str]:
        """
        Get the game name for a model instance.

        Args:
            model: Model instance

        Returns:
            Game name, or None if not found
        """
        model_class = type(model)
        for game_name, registered_class in cls._registry.items():
            if model_class == registered_class:
                return game_name
        return None

    @classmethod
    def validate_compatibility(cls, game: Game, model: nn.Module) -> None:
        """
        Validate that a model is compatible with a game.

        Args:
            game: Game instance
            model: Model instance

        Raises:
            ValueError: If model incompatible with game
        """
        # Check action size
        if hasattr(model, 'action_size'):
            if model.action_size != game.action_size:
                raise ValueError(
                    f"Model action_size ({model.action_size}) doesn't match "
                    f"game action_size ({game.action_size})"
                )

        # Check game name match
        model_game_name = cls.get_game_for_model(model)
        if model_game_name and hasattr(game, '__class__'):
            game_class_name = game.__class__.__name__.lower()
            if model_game_name not in game_class_name:
                raise ValueError(
                    f"Model is for '{model_game_name}' but got game '{game_class_name}'"
                )

    @classmethod
    def list_games(cls) -> list[str]:
        """List all registered games."""
        return sorted(cls._registry.keys())


# Auto-register known models
def _auto_register():
    """Automatically register all known game-model pairs."""
    try:
        from src.games.tictactoe.model import TicTacToeNet
        ModelRegistry.register('tictactoe', TicTacToeNet)
    except ImportError:
        pass

    try:
        from src.games.connect4.model import Connect4Net
        ModelRegistry.register('connect4', Connect4Net)
    except ImportError:
        pass


_auto_register()


# Convenience function for scripts
def get_model_for_game(game: Game) -> Type[nn.Module]:
    """
    Get the appropriate model class for a game instance.

    Args:
        game: Game instance

    Returns:
        Model class

    Example:
        game = TicTacToe()
        ModelCls = get_model_for_game(game)
        model = ModelCls()
    """
    game_name = game.__class__.__name__.lower()
    return ModelRegistry.get_model(game_name)
