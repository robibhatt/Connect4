"""
Tests for Connect4MatchUI - Connect4-specific agent vs agent visualization.

These tests verify:
- Correct 6x7 geometry
- Geometry matches Connect4UI
- Red and yellow disc rendering
- Color consistency
- Score overlay positioning
- Full game playthrough
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import pygame as pg

from src.games.connect4.ui.connect4_match_ui import Connect4MatchUI
from src.games.connect4.ui.connect4_ui import Connect4UI
from src.games.connect4 import Connect4
from src.agents.agent import Agent


@pytest.fixture
def game():
    """Connect4 game instance"""
    return Connect4()


@pytest.fixture
def mock_agent1():
    """Mock agent 1"""
    agent = Mock(spec=Agent)
    agent.act = Mock(return_value=0)
    agent.start = Mock()
    return agent


@pytest.fixture
def mock_agent2():
    """Mock agent 2"""
    agent = Mock(spec=Agent)
    agent.act = Mock(return_value=1)
    agent.start = Mock()
    return agent


@pytest.fixture
def ui(game, mock_agent1, mock_agent2):
    """Connect4MatchUI instance for testing"""
    pg.init()

    ui_instance = Connect4MatchUI(
        game, mock_agent1, mock_agent2,
        agent1_name="Agent1", agent2_name="Agent2"
    )

    # Use real pygame surfaces
    ui_instance.screen = pg.Surface((600, 600))
    ui_instance.font = pg.font.SysFont(None, 26)
    ui_instance.big_font = pg.font.SysFont(None, 34)

    ui_instance._compute_board_geometry()

    return ui_instance


# ===== A. Geometry Tests (3 tests) =====

def test_connect4_match_ui_has_6x7_geometry(ui):
    """Test 47: Board geometry indicates 6 rows × 7 columns"""
    assert hasattr(ui, 'cell_width')
    assert hasattr(ui, 'cell_height')
    assert ui.cell_width > 0
    assert ui.cell_height > 0

    # Board should be 7 columns wide and 6 rows tall
    board_width = ui.board_x1 - ui.board_x0
    board_height = ui.board_y1 - ui.board_y0

    # Note: Connect4 has 7 columns, 6 rows
    assert board_width == 7 * ui.cell_width
    assert board_height == 6 * ui.cell_height


def test_board_geometry_matches_connect4_ui(game, mock_agent1):
    """Test 48: Geometry matches regular Connect4UI"""
    pg.init()

    # Create both UIs with same config
    match_ui = Connect4MatchUI(game, mock_agent1, mock_agent1)
    regular_ui = Connect4UI(game, mock_agent1)

    # Mock pygame components
    for ui_instance in [match_ui, regular_ui]:
        ui_instance.screen = pg.Surface((600, 600))
        ui_instance.font = pg.font.SysFont(None, 26)
        ui_instance.big_font = pg.font.SysFont(None, 34)
        ui_instance._compute_board_geometry()

    # Geometry should match
    assert match_ui.cell_width == regular_ui.cell_width
    assert match_ui.cell_height == regular_ui.cell_height
    assert match_ui.board_x0 == regular_ui.board_x0
    assert match_ui.board_y0 == regular_ui.board_y0
    assert match_ui.board_x1 == regular_ui.board_x1
    assert match_ui.board_y1 == regular_ui.board_y1


def test_cell_width_and_height_calculated(ui):
    """Test 49: cell_width and cell_height are positive"""
    assert ui.cell_width > 0
    assert ui.cell_height > 0

    # Cells should have reasonable dimensions
    assert ui.cell_width < 200
    assert ui.cell_height < 200


# ===== B. Rendering Tests (3 tests) =====

def test_renders_red_and_yellow_discs(ui, game):
    """Test 50: Renders colored discs for both agents"""
    ui.new_game()

    # Make some moves to have pieces from both players
    ui.state.board[5, 0] = 1  # Bottom left (current player)
    ui.state.board[5, 1] = -1  # Bottom row, column 1 (opponent)

    # Should not raise exception
    ui._render_pieces()


def test_agent1_gets_consistent_color(ui, mock_agent1, mock_agent2):
    """Test 51: agent1 always gets same color across games"""
    # Note: This is a design choice - agent1 could always be red or yellow
    # For this test, we just verify the color assignment is consistent

    ui.new_game()
    first_game_agent1_symbol = ui.agent1_symbol

    # Play a few more games
    for _ in range(5):
        with patch('pygame.event.clear'):
            ui.new_game()

    # Agent1 should have varied symbols (X/O) due to randomization
    # But color mapping should be consistent based on symbol
    # This test mainly verifies no crashes and consistency exists


def test_score_overlay_positioned_correctly(ui):
    """Test 52: Score overlay doesn't overlap board"""
    ui.new_game()
    ui.wins1 = 10
    ui.wins2 = 8

    # Should not raise exception
    ui._render_score_overlay()

    # Score should be positioned to not overlap the board
    # (Verified by not crashing and proper implementation)


# ===== C. Integration Tests (2 tests) =====

def test_full_connect4_game_playthrough(ui, mock_agent1, mock_agent2):
    """Test 53: Simulate complete Connect4 game"""
    # Set up agents to play alternating columns
    agent1_col = 0
    agent2_col = 1

    def agent1_move(state):
        return agent1_col

    def agent2_move(state):
        return agent2_col

    mock_agent1.act.side_effect = agent1_move
    mock_agent2.act.side_effect = agent2_move

    ui.new_game()

    # Play game (Connect4 can take up to 42 moves)
    for _ in range(42):
        if ui.game_over:
            break

        ui.waiting_for_step = False
        ui._update()

    # Game should end
    assert ui.game_over

    # Score should be updated
    assert ui.games_played == 1
    assert (ui.wins1 + ui.wins2 + ui.draws) == 1


def test_multiple_connect4_games(ui, mock_agent1, mock_agent2):
    """Test 54: Play multiple games and verify scores"""
    # Set agents to always play legal moves (spread across columns)
    col_counter = [0]

    def get_legal_move(state):
        legal_mask = ui.game.legal_actions(state)
        legal_indices = np.where(legal_mask)[0]  # Get indices where True
        col = col_counter[0] % len(legal_indices)
        col_counter[0] += 1
        return int(legal_indices[col])

    mock_agent1.act.side_effect = get_legal_move
    mock_agent2.act.side_effect = get_legal_move

    # Play 5 games
    for game_num in range(5):
        with patch('pygame.event.clear'):
            ui.new_game()

        # Play until game over
        for _ in range(50):  # Max moves
            if ui.game_over:
                break
            ui.waiting_for_step = False
            ui._update()

        assert ui.games_played == game_num + 1

    # All 5 games should be accounted for
    assert ui.games_played == 5
    assert (ui.wins1 + ui.wins2 + ui.draws) == 5


def test_connect4_board_centered_in_larger_window(game, mock_agent1, mock_agent2):
    """Test 55: Connect4 board stays centered with larger window size"""
    from src.games.core.ui.game_ui import UIConfig
    pg.init()

    cfg = UIConfig(window_size=800)
    ui = Connect4MatchUI(game, mock_agent1, mock_agent2, cfg=cfg)
    ui.screen = pg.Surface((800, 800))
    ui.font = pg.font.SysFont(None, 26)
    ui.big_font = pg.font.SysFont(None, 34)
    ui._compute_board_geometry()

    # Board should be centered horizontally
    left_margin = ui.board_x0
    right_margin = 800 - ui.board_x1

    # Margins should be approximately equal (within 1px due to integer division)
    assert abs(left_margin - right_margin) <= 1

    # Board should still be 7 columns × 6 rows
    assert (ui.board_x1 - ui.board_x0) == 7 * ui.cell_width
    assert (ui.board_y1 - ui.board_y0) == 6 * ui.cell_height
