"""
Manual test/demo for TicTacToe UI.

This is NOT an automated test - it requires human interaction.
Run this script to play TicTacToe against a RandomAgent and verify
that the UI correctly handles input, rendering, and game flow.

Usage:
    python tests/demo_tictactoe_ui.py
"""

from src.games.tictactoe import TicTacToe
from src.agents.agent import RandomAgent
from src.games.tictactoe.ui import TicTacToeUI
from src.games.core.ui import UIConfig
import numpy as np


# Configuration (modify as needed for testing)
PAUSE_SECONDS = 0.4      # Delay between agent moves
WINDOW_SIZE = 600        # UI window size
RANDOM_SEED = None       # Set to int for reproducible behavior


def print_instructions():
    """Display test instructions to user"""
    print("=" * 80)
    print("TicTacToe UI Manual Test")
    print("=" * 80)
    print("Purpose: Test human interaction with TicTacToe UI against a RandomAgent")
    print()
    print("What to test:")
    print("  - Click detection: Click cells to make moves")
    print("  - Visual rendering: X and O symbols appear correctly")
    print("  - Turn management: Human and agent alternate properly")
    print("  - Game flow: Win/loss/draw detection works")
    print("  - UI responsiveness: Controls respond smoothly")
    print()
    print("Controls:")
    print("  - Left Click: Select cell to play")
    print("  - N: Start new game")
    print("  - Q or ESC: Quit")
    print()
    print("The agent plays randomly, so games should be easy to win.")
    print("Test by playing multiple games and verifying all UI elements work.")
    print("=" * 80)
    print()


def main():
    """Run the TicTacToe UI test"""
    print_instructions()

    # Initialize game
    game = TicTacToe()

    # Initialize random agent
    rng = np.random.default_rng(RANDOM_SEED) if RANDOM_SEED else None
    agent = RandomAgent(game=game, rng=rng)

    # Configure UI
    cfg = UIConfig(window_size=WINDOW_SIZE)

    # Create and run UI
    ui = TicTacToeUI(
        game=game,
        agent=agent,
        pause_seconds=PAUSE_SECONDS,
        rng=rng,
        cfg=cfg
    )

    print("Launching UI... (close window or press Q to exit)")
    ui.run()
    print("\nUI closed. Test complete.")


if __name__ == "__main__":
    main()
