"""
Tests for TicTacToeUI implementation.
Tests game-specific behavior like input mapping and geometry.
"""

import pytest
from unittest.mock import Mock
import numpy as np
import pygame as pg

# These imports will fail initially - that's expected in TDD
from src.games.tictactoe.ui.tictactoe_ui import TicTacToeUI
from src.games.tictactoe import TicTacToe
from src.agents.agent import Agent


@pytest.fixture
def game():
    """TicTacToe game instance"""
    return TicTacToe()


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    agent = Mock(spec=Agent)
    agent.act = Mock(return_value=4)  # Always play center
    agent.start = Mock()
    return agent


@pytest.fixture
def ui(game, mock_agent):
    """TicTacToeUI instance for testing"""
    # Initialize pygame (headless, no display needed)
    pg.init()

    ui_instance = TicTacToeUI(game, mock_agent, pause_seconds=0.1)

    # Use real pygame surfaces for rendering tests
    ui_instance.screen = pg.Surface((600, 600))
    ui_instance.font = pg.font.SysFont(None, 26)
    ui_instance.big_font = pg.font.SysFont(None, 34)

    # Compute geometry (needed for input mapping tests)
    ui_instance._compute_board_geometry()

    return ui_instance


# ===== Geometry Tests =====

def test_tictactoe_board_bounds_computed(ui):
    """Test that board geometry is computed correctly"""
    assert hasattr(ui, 'board_x0')
    assert hasattr(ui, 'board_y0')
    assert hasattr(ui, 'board_x1')
    assert hasattr(ui, 'board_y1')
    assert hasattr(ui, 'cell')

    # Bounds should be sensible
    assert ui.board_x0 >= 0
    assert ui.board_y0 >= 0
    assert ui.board_x1 > ui.board_x0
    assert ui.board_y1 > ui.board_y0


def test_tictactoe_square_cells(ui):
    """Test that TicTacToe uses square cells"""
    # Width and height should be equal (square grid)
    width = ui.board_x1 - ui.board_x0
    height = ui.board_y1 - ui.board_y0

    # 3x3 grid should be square
    assert width == height


def test_board_geometry_creates_3x3_grid(ui):
    """Test that geometry creates exactly 3x3 grid"""
    # Board should accommodate 3 cells in each direction
    width = ui.board_x1 - ui.board_x0
    assert width == 3 * ui.cell


# ===== Input Mapping Tests =====

def test_tictactoe_click_top_left_corner(ui):
    """Test clicking top-left cell (action 0)"""
    # Click in top-left cell
    x = ui.board_x0 + ui.cell // 2
    y = ui.board_y0 + ui.cell // 2

    action = ui._screen_pos_to_action((x, y))

    assert action == 0


def test_tictactoe_click_center_cell(ui):
    """Test clicking center cell (action 4)"""
    # Click in center cell
    x = ui.board_x0 + ui.cell + ui.cell // 2
    y = ui.board_y0 + ui.cell + ui.cell // 2

    action = ui._screen_pos_to_action((x, y))

    assert action == 4


def test_tictactoe_click_bottom_right(ui):
    """Test clicking bottom-right cell (action 8)"""
    # Click in bottom-right cell
    x = ui.board_x0 + 2 * ui.cell + ui.cell // 2
    y = ui.board_y0 + 2 * ui.cell + ui.cell // 2

    action = ui._screen_pos_to_action((x, y))

    assert action == 8


def test_tictactoe_click_all_cells(ui):
    """Test that all 9 cells map to correct actions"""
    for row in range(3):
        for col in range(3):
            # Click center of each cell
            x = ui.board_x0 + col * ui.cell + ui.cell // 2
            y = ui.board_y0 + row * ui.cell + ui.cell // 2

            action = ui._screen_pos_to_action((x, y))
            expected_action = row * 3 + col

            assert action == expected_action, f"Row {row}, Col {col}: got {action}, expected {expected_action}"


def test_tictactoe_click_out_of_bounds_left(ui):
    """Test clicking left of board returns None"""
    x = ui.board_x0 - 10
    y = ui.board_y0 + ui.cell

    action = ui._screen_pos_to_action((x, y))

    assert action is None


def test_tictactoe_click_out_of_bounds_above(ui):
    """Test clicking above board returns None"""
    x = ui.board_x0 + ui.cell
    y = ui.board_y0 - 10

    action = ui._screen_pos_to_action((x, y))

    assert action is None


def test_tictactoe_click_out_of_bounds_right(ui):
    """Test clicking right of board returns None"""
    x = ui.board_x1 + 10
    y = ui.board_y0 + ui.cell

    action = ui._screen_pos_to_action((x, y))

    assert action is None


def test_tictactoe_click_out_of_bounds_below(ui):
    """Test clicking below board returns None"""
    x = ui.board_x0 + ui.cell
    y = ui.board_y1 + 10

    action = ui._screen_pos_to_action((x, y))

    assert action is None


# ===== Rendering Tests (Smoke Tests) =====

def test_render_board_doesnt_crash(ui):
    """Test that rendering board doesn't raise exceptions"""
    try:
        ui._render_board()
    except Exception as e:
        pytest.fail(f"_render_board() raised an exception: {e}")


def test_render_pieces_doesnt_crash(ui):
    """Test that rendering pieces doesn't raise exceptions"""
    ui.new_game()

    try:
        ui._render_pieces()
    except Exception as e:
        pytest.fail(f"_render_pieces() raised an exception: {e}")


def test_render_pieces_with_moves(ui):
    """Test rendering after some moves have been made"""
    ui.new_game()

    # Make a few moves
    if ui.human_to_move:
        ui._apply_human_move(0)

    try:
        ui._render_pieces()
    except Exception as e:
        pytest.fail(f"_render_pieces() with moves raised an exception: {e}")
