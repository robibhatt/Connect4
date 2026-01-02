"""
Neural network models for AlphaZero training.

Models are organized in game-specific directories:
    - src/games/connect4/models/
    - src/games/tictactoe/models/

Import Pattern:
    # Import model classes
    from src.games.connect4.models import Connect4MLPNet
    from src.games.tictactoe.models import TicTacToeMLPNet

    # Use ModelRegistry for dynamic loading
    from src.models.registry import ModelRegistry
    model_cls = ModelRegistry.get_model('Connect4MLPNet')
"""

from src.models.registry import ModelRegistry
from src.models.base import GameNet

__all__ = [
    'GameNet',
    'ModelRegistry',
]
