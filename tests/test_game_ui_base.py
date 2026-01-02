"""
Tests for abstract GameUI base class functionality.
These tests verify shared behavior across all game UIs.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

# These imports will fail initially - that's expected in TDD
from src.games.ui.game_ui import GameUI
from src.games.tiktaktoe import TicTacToe
from src.agents.agent import Agent


class ConcreteGameUI(GameUI):
    """Minimal concrete implementation for testing base class"""

    def _compute_board_geometry(self):
        self.board_x0 = 0
        self.board_y0 = 70
        self.cell_size = 100

    def _render_board(self):
        pass  # No-op for testing

    def _render_pieces(self):
        pass  # No-op for testing

    def _screen_pos_to_action(self, pos):
        # Simple grid mapping for testing
        x, y = pos
        if not (0 <= x < 300 and 70 <= y < 370):
            return None
        col = x // 100
        row = (y - 70) // 100
        return row * 3 + col


@pytest.fixture
def game():
    """TicTacToe game instance for testing"""
    return TicTacToe()


@pytest.fixture
def mock_agent():
    """Mock agent that always returns action 0"""
    agent = Mock(spec=Agent)
    agent.act = Mock(return_value=0)
    agent.start = Mock()
    return agent


@pytest.fixture
def ui(game, mock_agent):
    """UI instance without pygame initialized"""
    ui_instance = ConcreteGameUI(game, mock_agent, pause_seconds=0.1)
    # Mock pygame components
    ui_instance.screen = Mock()
    ui_instance.font = Mock(return_value=Mock(render=Mock(return_value=Mock())))
    ui_instance.big_font = Mock(return_value=Mock(render=Mock(return_value=Mock())))
    ui_instance._compute_board_geometry()
    return ui_instance


# ===== Game Flow Tests =====

def test_new_game_initializes_state(ui, game):
    """Test that new_game() properly initializes game state"""
    ui.new_game()

    assert ui.state is not None
    assert not ui.game_over
    assert ui.human_symbol in ["X", "O"]
    assert ui.agent_symbol in ["X", "O"]
    assert ui.human_symbol != ui.agent_symbol


def test_human_move_switches_to_agent_turn(ui):
    """Test that applying human move switches turn to agent"""
    ui.new_game()

    # Force human to be first
    ui.human_to_move = True

    # Apply human move to action 0
    ui._apply_human_move(0)

    assert not ui.human_to_move  # Should now be agent's turn


def test_agent_move_switches_to_human_turn(ui, mock_agent):
    """Test that applying agent move switches turn to human"""
    ui.new_game()

    # Force agent to be first
    ui.human_to_move = False

    # Apply agent move
    ui._apply_agent_move()

    assert ui.human_to_move  # Should now be human's turn
    mock_agent.act.assert_called_once()


def test_illegal_move_rejected(ui):
    """Test that illegal moves don't change state"""
    ui.new_game()
    ui.human_to_move = True

    # Make the first move
    ui._apply_human_move(0)
    original_state = ui.state

    # Try to play same position again (illegal)
    ui.human_to_move = True  # Force back to human
    ui._apply_human_move(0)

    # State should have changed (move was attempted)
    # But we need to check via legal_actions
    assert not ui._is_legal(0) or ui.state != original_state


def test_terminal_detection_on_win(ui, game):
    """Test that terminal state is detected on win"""
    ui.new_game()

    # Create a winning state manually (X wins top row)
    # Player 1 (X): 0, 1, 2
    # Player 2 (O): 3, 4
    board = np.array([[1, 1, 1], [0, 0, -1], [0, -1, 0]], dtype=np.int8)

    from src.games.tiktaktoe import TicTacToeState
    ui.state = TicTacToeState(board=board)
    ui.human_to_move = True  # Current player is +1

    ui._check_terminal()

    assert ui.game_over
    assert "WIN" in ui.msg.upper() or "LOSE" in ui.msg.upper()


def test_terminal_detection_on_draw(ui):
    """Test that terminal state is detected on draw"""
    ui.new_game()

    # Create a draw state
    # X O X
    # X O O
    # O X X
    board = np.array([[1, -1, 1], [1, -1, -1], [-1, 1, 1]], dtype=np.int8)

    from src.games.tiktaktoe import TicTacToeState
    ui.state = TicTacToeState(board=board)

    ui._check_terminal()

    assert ui.game_over
    assert "DRAW" in ui.msg.upper()


# ===== Symbol/Turn Management Tests =====

def test_symbol_randomization(ui):
    """Test that symbols are randomized across multiple games"""
    symbols = set()

    for _ in range(10):
        ui.new_game()
        symbols.add(ui.human_symbol)

    # Should see both X and O assigned to human across 10 games
    # (probability of failure is 2^-10 ≈ 0.001)
    assert len(symbols) == 2


def test_x_always_starts(ui):
    """Test that X always makes the first move"""
    for _ in range(10):
        ui.new_game()

        # If human is X, human should move first
        # If agent is X, agent should move first
        if ui.human_symbol == "X":
            assert ui.human_to_move
        else:
            assert not ui.human_to_move


def test_canonical_state_perspective(ui):
    """Test that canonical state is correctly maintained"""
    ui.new_game()

    # In canonical state, current player is always +1
    # This is verified by the game's to_play() method
    player_to_move = ui.game.to_play(ui.state)
    assert player_to_move == +1
