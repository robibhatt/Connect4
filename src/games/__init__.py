"""
Game implementations and UI components.

IMPORTANT: Circular Import Constraints
========================================

This module INTENTIONALLY does not provide re-exports at the top level.

Reason:
-------
The games module has a bidirectional dependency with the agents module:
  - Games import from agents (for type hints, testing, UI)
  - Agents import from games (for Game and State base classes)

Providing re-exports here (e.g., `from src.games.core import Game`) would
create a circular import chain when agents try to import from games:

  src.games.__init__ -> src.games.core -> src.agents -> src.games.__init__

Solution:
---------
Import from specific submodules instead of the top-level games module.

Recommended Import Patterns:
----------------------------
Core abstractions:
  from src.games.core import Game, State
  from src.games.core.game_play import simulate_match
  from src.games.core.ui import GameUI, UIConfig

TicTacToe:
  from src.games.tictactoe import TicTacToe, TicTacToeState
  from src.games.tictactoe.ui import TicTacToeUI

Connect4:
  from src.games.connect4 import Connect4, Connect4State
  from src.games.connect4.ui import Connect4UI

Note: Other modules (models, mcts, training) do not have this constraint
and provide convenient re-exports via their __init__.py files.
"""
