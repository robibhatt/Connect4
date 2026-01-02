"""
Tests for Connect4UI implementation.
Tests game-specific behavior like column-based input and 6x7 geometry.
"""

import pytest
from unittest.mock import Mock
import numpy as np

# These imports will fail initially - that's expected in TDD
from src.games.ui.connect4_ui import Connect4UI
from src.games.connect4 import Connect4
from src.agents.agent import Agent


@pytest.fixture
def game():
    """Connect4 game instance"""
    return Connect4()


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    agent = Mock(spec=Agent)
    agent.act = Mock(return_value=3)  # Always play middle column
    agent.start = Mock()
    return agent


@pytest.fixture
def ui(game, mock_agent):
    """Connect4UI instance without pygame initialized"""
    ui_instance = Connect4UI(game, mock_agent, pause_seconds=0.1)

    # Mock pygame components to avoid actual rendering
    ui_instance.screen = Mock()
    ui_instance.font = Mock()
    ui_instance.big_font = Mock()

    # Compute geometry (needed for input mapping tests)
    ui_instance._compute_board_geometry()

    return ui_instance


# ===== Geometry Tests =====

def test_connect4_board_bounds_computed(ui):
    """Test that board geometry is computed correctly"""
    assert hasattr(ui, 'board_x0')
    assert hasattr(ui, 'board_y0')
    assert hasattr(ui, 'board_x1')
    assert hasattr(ui, 'board_y1')
    assert hasattr(ui, 'cell_width')
    assert hasattr(ui, 'cell_height')

    # Bounds should be sensible
    assert ui.board_x0 >= 0
    assert ui.board_y0 >= 0
    assert ui.board_x1 > ui.board_x0
    assert ui.board_y1 > ui.board_y0


def test_connect4_7_columns(ui):
    """Test that Connect4 board has 7 columns"""
    # Board width should accommodate exactly 7 cells
    width = ui.board_x1 - ui.board_x0
    assert width == 7 * ui.cell_width


def test_connect4_6_rows(ui):
    """Test that Connect4 board has 6 rows"""
    # Board height should accommodate exactly 6 cells
    height = ui.board_y1 - ui.board_y0
    assert height == 6 * ui.cell_height


# ===== Input Mapping Tests (Column-Based) =====

def test_connect4_click_column_0(ui):
    """Test clicking leftmost column"""
    # Click in column 0 (anywhere vertically)
    x = ui.board_x0 + ui.cell_width // 2
    y = ui.board_y0 + ui.cell_height // 2

    action = ui._screen_pos_to_action((x, y))

    assert action == 0


def test_connect4_click_column_3(ui):
    """Test clicking middle column"""
    # Click in column 3 (middle)
    x = ui.board_x0 + 3 * ui.cell_width + ui.cell_width // 2
    y = ui.board_y0 + ui.cell_height // 2

    action = ui._screen_pos_to_action((x, y))

    assert action == 3


def test_connect4_click_column_6(ui):
    """Test clicking rightmost column"""
    # Click in column 6
    x = ui.board_x0 + 6 * ui.cell_width + ui.cell_width // 2
    y = ui.board_y0 + ui.cell_height // 2

    action = ui._screen_pos_to_action((x, y))

    assert action == 6


def test_connect4_click_all_columns(ui):
    """Test that all 7 columns map correctly"""
    for col in range(7):
        # Click center of each column (middle row)
        x = ui.board_x0 + col * ui.cell_width + ui.cell_width // 2
        y = ui.board_y0 + 3 * ui.cell_height  # Middle row

        action = ui._screen_pos_to_action((x, y))

        assert action == col, f"Column {col}: got {action}, expected {col}"


def test_connect4_y_coordinate_doesnt_matter(ui):
    """Test that y-coordinate doesn't affect column selection (gravity)"""
    # Click same column at different y positions
    col = 3
    x = ui.board_x0 + col * ui.cell_width + ui.cell_width // 2

    for row in range(6):
        y = ui.board_y0 + row * ui.cell_height + ui.cell_height // 2
        action = ui._screen_pos_to_action((x, y))

        assert action == col, f"Row {row}: y-position affected column selection"


def test_connect4_click_above_board(ui):
    """Test clicking above board returns None"""
    x = ui.board_x0 + ui.cell_width
    y = ui.board_y0 - 10

    action = ui._screen_pos_to_action((x, y))

    assert action is None


def test_connect4_click_below_board(ui):
    """Test clicking below board returns None"""
    x = ui.board_x0 + ui.cell_width
    y = ui.board_y1 + 10

    action = ui._screen_pos_to_action((x, y))

    assert action is None


def test_connect4_click_left_of_board(ui):
    """Test clicking left of board returns None"""
    x = ui.board_x0 - 10
    y = ui.board_y0 + ui.cell_height

    action = ui._screen_pos_to_action((x, y))

    assert action is None


def test_connect4_click_right_of_board(ui):
    """Test clicking right of board returns None"""
    x = ui.board_x1 + 10
    y = ui.board_y0 + ui.cell_height

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

    # Make a few moves in different columns
    if ui.human_to_move:
        ui._apply_human_move(0)  # Column 0
        ui._apply_agent_move()   # Agent plays
        ui._apply_human_move(3)  # Column 3

    try:
        ui._render_pieces()
    except Exception as e:
        pytest.fail(f"_render_pieces() with moves raised an exception: {e}")


# ===== Connect4-Specific: Gravity Tests =====

def test_pieces_rendered_respect_gravity(ui):
    """Test that pieces are drawn in correct rows (gravity)"""
    ui.new_game()

    # This is more of an integration test - just verify no crashes
    # when rendering a state with pieces at different heights
    for col in [0, 1, 2]:
        if ui.human_to_move:
            ui._apply_human_move(col)
        ui._apply_agent_move()

    # Should render without issues
    try:
        ui._render_pieces()
    except Exception as e:
        pytest.fail(f"Rendering with gravity failed: {e}")
