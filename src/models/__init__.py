"""
Neural network models for AlphaZero training.

DEPRECATED: Models have moved to game-specific directories.
This module maintains backward compatibility during transition.

Recommended Import Pattern:
    # New way (preferred)
    from src.games.tictactoe import TicTacToeNet
    from src.games.connect4 import Connect4Net

    # Old way (deprecated but still works)
    from src.models import TicTacToeNet
"""

import warnings
from src.models.registry import ModelRegistry, get_model_for_game
from src.models.base import GameNet


# Backward-compatible imports (with deprecation warnings)
def __getattr__(name):
    """Lazy import with deprecation warning for old import paths."""
    if name == 'TicTacToeNet':
        warnings.warn(
            "Importing TicTacToeNet from src.models is deprecated. "
            "Use 'from src.games.tictactoe import TicTacToeNet' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from src.games.tictactoe.model import TicTacToeNet
        return TicTacToeNet

    if name == 'Connect4Net':
        warnings.warn(
            "Importing Connect4Net from src.models is deprecated. "
            "Use 'from src.games.connect4 import Connect4Net' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        from src.games.connect4.model import Connect4Net
        return Connect4Net

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'GameNet',
    'ModelRegistry',
    'get_model_for_game',
    'TicTacToeNet',  # deprecated
    'Connect4Net',    # deprecated
]
