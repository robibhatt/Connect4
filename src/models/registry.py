"""
Game-Model Registry for AlphaZero training.

Provides centralized mapping between model class names and model classes,
enabling dynamic model loading and validation. Supports multiple models per game.
"""

from __future__ import annotations
from typing import Type, Dict, Optional, Set
import re
import torch.nn as nn

from src.games.core.game import Game


class ModelRegistry:
    """
    SINGLE centralized registry for ALL models from ALL games.

    Supports multiple model architectures per game (e.g., Connect4MLPNet, Connect4CNNNet).

    Usage:
        # Register a model (auto-extracts game name from class name)
        ModelRegistry.register(Connect4MLPNet)

        # Get model class by full class name
        model_cls = ModelRegistry.get_model('Connect4MLPNet')
        model = model_cls()

        # Get all models for a game
        models = ModelRegistry.get_models_for_game('connect4')

        # Validate game-model compatibility
        ModelRegistry.validate_compatibility(game, model)
    """

    _registry: Dict[str, Type[nn.Module]] = {}  # model_class_name -> model_class
    _game_to_models: Dict[str, Set[str]] = {}   # game_name -> set of model_class_names

    @classmethod
    def register(cls, model_class: Type[nn.Module]) -> None:
        """
        Register a model class (auto-extracts game name from class name).

        Expects model class names in format: {Game}{Architecture}Net
        E.g., Connect4MLPNet -> extracts 'connect4'

        Args:
            model_class: Neural network class to register

        Raises:
            ValueError: If model already registered or game name cannot be extracted
        """
        model_class_name = model_class.__name__

        # Extract game name from class name
        # Pattern: {Game}{Architecture}Net -> extract {Game}
        # E.g., Connect4MLPNet -> Connect4, TicTacToeMLPNet -> TicTacToe
        # Game can contain digits (Connect4), Architecture is uppercase (MLP, CNN, etc.)
        match = re.match(r'^([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*)([A-Z][A-Z]+|[A-Z][a-z]+)Net$', model_class_name)
        if not match:
            raise ValueError(
                f"Model class name '{model_class_name}' does not follow pattern "
                f"{{Game}}{{Architecture}}Net (e.g., Connect4MLPNet)"
            )

        game_name_camel = match.group(1)  # E.g., "Connect4" or "TicTacToe"
        game_name = game_name_camel.lower()  # E.g., "connect4" or "tictactoe"

        # Register in main registry
        if model_class_name in cls._registry:
            if cls._registry[model_class_name] != model_class:
                raise ValueError(
                    f"Model '{model_class_name}' already registered with "
                    f"{cls._registry[model_class_name]}, "
                    f"cannot register {model_class}"
                )
            # Already registered with same class, silently succeed
            return

        cls._registry[model_class_name] = model_class

        # Track in game-to-models mapping
        if game_name not in cls._game_to_models:
            cls._game_to_models[game_name] = set()
        cls._game_to_models[game_name].add(model_class_name)

    @classmethod
    def get_model(cls, model_class_name: str) -> Type[nn.Module]:
        """
        Get the model class by full class name.

        Args:
            model_class_name: Full model class name (e.g., 'Connect4MLPNet')

        Returns:
            Model class

        Raises:
            KeyError: If model not registered
        """
        # Lazy registration - try to register on first access
        if model_class_name not in cls._registry:
            cls._ensure_registered_by_class_name(model_class_name)

        if model_class_name not in cls._registry:
            raise KeyError(
                f"No model registered with name '{model_class_name}'. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[model_class_name]

    @classmethod
    def get_models_for_game(cls, game_name: str) -> list[str]:
        """
        Get all model class names registered for a game.

        Args:
            game_name: Game identifier (e.g., 'connect4')

        Returns:
            List of model class names (e.g., ['Connect4MLPNet', 'Connect4CNNNet'])
        """
        # Lazy registration attempt
        cls._ensure_registered(game_name)

        if game_name not in cls._game_to_models:
            return []
        return sorted(cls._game_to_models[game_name])

    @classmethod
    def _ensure_registered(cls, game_name: str) -> None:
        """Ensure a game's models are registered (lazy registration by game name)."""
        if game_name == 'tictactoe' and game_name not in cls._game_to_models:
            try:
                from src.games.tictactoe.models.mlp import TicTacToeMLPNet
                cls.register(TicTacToeMLPNet)
            except (ImportError, ValueError):
                pass

        if game_name == 'connect4' and game_name not in cls._game_to_models:
            try:
                from src.games.connect4.models.mlp import Connect4MLPNet
                cls.register(Connect4MLPNet)
            except (ImportError, ValueError):
                pass

    @classmethod
    def _ensure_registered_by_class_name(cls, model_class_name: str) -> None:
        """Ensure a model is registered (lazy registration by model class name)."""
        if model_class_name == 'TicTacToeMLPNet':
            try:
                from src.games.tictactoe.models.mlp import TicTacToeMLPNet
                cls.register(TicTacToeMLPNet)
            except (ImportError, ValueError):
                pass
        elif model_class_name == 'Connect4MLPNet':
            try:
                from src.games.connect4.models.mlp import Connect4MLPNet
                cls.register(Connect4MLPNet)
            except (ImportError, ValueError):
                pass

    @classmethod
    def get_game_for_model(cls, model: nn.Module) -> Optional[str]:
        """
        Get the game name for a model instance.

        Args:
            model: Model instance

        Returns:
            Game name (lowercase), or None if not found
        """
        model_class_name = type(model).__name__

        # Search in reverse mapping
        for game_name, model_names in cls._game_to_models.items():
            if model_class_name in model_names:
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

        # Check game name embedded in model class name
        model_class_name = type(model).__name__
        game_class_name = game.__class__.__name__

        # Model should start with game name (case-insensitive)
        # E.g., Connect4MLPNet starts with Connect4
        if not model_class_name.lower().startswith(game_class_name.lower()):
            raise ValueError(
                f"Model '{model_class_name}' is not compatible with game '{game_class_name}'. "
                f"Model class name should start with game name."
            )

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model class names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_games(cls) -> list[str]:
        """List all games that have registered models."""
        return sorted(cls._game_to_models.keys())


# Auto-register known models
def _auto_register():
    """Automatically register all known models."""
    try:
        from src.games.tictactoe.models.mlp import TicTacToeMLPNet
        ModelRegistry.register(TicTacToeMLPNet)
    except ImportError:
        pass

    try:
        from src.games.connect4.models.mlp import Connect4MLPNet
        ModelRegistry.register(Connect4MLPNet)
    except ImportError:
        pass


_auto_register()
