"""
Game Registry for AlphaZero training.

Provides centralized mapping of game names to game classes,
enabling dynamic game loading and instantiation.
"""

from __future__ import annotations
from typing import Type, Dict

from src.games.core.game import Game


class GameRegistry:
    """
    Registry mapping game names to their corresponding game classes.

    Usage:
        # Get game class
        game_cls = GameRegistry.get_game('tictactoe')
        game = game_cls()

        # Register a new game
        GameRegistry.register('my_game', MyGame)

        # List all games
        games = GameRegistry.list_games()
    """

    _registry: Dict[str, Type[Game]] = {}

    @classmethod
    def register(cls, game_name: str, game_class: Type[Game]) -> None:
        """
        Register a game class.

        Args:
            game_name: Unique game identifier (e.g., 'tictactoe')
            game_class: Game class for this game

        Raises:
            ValueError: If game already registered with a different class
        """
        if game_name in cls._registry:
            if cls._registry[game_name] != game_class:
                raise ValueError(
                    f"Game '{game_name}' already registered with "
                    f"{cls._registry[game_name].__name__}, "
                    f"cannot register {game_class.__name__}"
                )
            # Already registered with same class, silently succeed
            return
        cls._registry[game_name] = game_class

    @classmethod
    def get_game(cls, game_name: str) -> Type[Game]:
        """
        Get the game class for a game name.

        Args:
            game_name: Game identifier

        Returns:
            Game class

        Raises:
            KeyError: If game not registered
        """
        # Lazy registration - try to register on first access
        if game_name not in cls._registry:
            cls._ensure_registered(game_name)

        if game_name not in cls._registry:
            raise KeyError(
                f"No game registered for '{game_name}'. "
                f"Available games: {list(cls._registry.keys())}"
            )
        return cls._registry[game_name]

    @classmethod
    def _ensure_registered(cls, game_name: str) -> None:
        """Ensure a game is registered (lazy registration)."""
        if game_name == 'tictactoe' and game_name not in cls._registry:
            try:
                from src.games.tictactoe.tictactoe import TicTacToe
                cls.register('tictactoe', TicTacToe)
            except (ImportError, ValueError):
                pass

        if game_name == 'connect4' and game_name not in cls._registry:
            try:
                from src.games.connect4.connect4 import Connect4
                cls.register('connect4', Connect4)
            except (ImportError, ValueError):
                pass

    @classmethod
    def list_games(cls) -> list[str]:
        """List all registered games."""
        return sorted(cls._registry.keys())


# Auto-register known games
def _auto_register():
    """Automatically register all known games."""
    try:
        from src.games.tictactoe.tictactoe import TicTacToe
        GameRegistry.register('tictactoe', TicTacToe)
    except ImportError:
        pass

    try:
        from src.games.connect4.connect4 import Connect4
        GameRegistry.register('connect4', Connect4)
    except ImportError:
        pass


_auto_register()
