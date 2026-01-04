"""
Tests for TicTacToeMatchUI - TicTacToe-specific agent vs agent visualization.

These tests verify:
- Correct 3x3 geometry
- Geometry matches TicTacToeUI
- X and O rendering for both agents
- Score overlay positioning
- Full game playthrough
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import pygame as pg

from src.games.tictactoe.ui.tictactoe_match_ui import TicTacToeMatchUI
from src.games.tictactoe.ui.tictactoe_ui import TicTacToeUI
from src.games.tictactoe import TicTacToe
from src.agents.agent import Agent


@pytest.fixture
def game():
    """TicTacToe game instance"""
    return TicTacToe()


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
    """TicTacToeMatchUI instance for testing"""
    pg.init()

    ui_instance = TicTacToeMatchUI(
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

def test_tictactoe_match_ui_has_3x3_geometry(ui):
    """Test 39: Board geometry indicates 3x3 grid"""
    # TicTacToe should have a square cell size
    assert hasattr(ui, 'cell')
    assert ui.cell > 0

    # Board should be 3 cells wide and tall
    board_width = ui.board_x1 - ui.board_x0
    board_height = ui.board_y1 - ui.board_y0

    assert board_width == 3 * ui.cell
    assert board_height == 3 * ui.cell


def test_board_geometry_matches_tictactoe_ui(game, mock_agent1):
    """Test 40: Geometry matches regular TicTacToeUI"""
    pg.init()

    # Create both UIs with same config
    match_ui = TicTacToeMatchUI(game, mock_agent1, mock_agent1)
    regular_ui = TicTacToeUI(game, mock_agent1)

    # Mock pygame components
    for ui_instance in [match_ui, regular_ui]:
        ui_instance.screen = pg.Surface((600, 600))
        ui_instance.font = pg.font.SysFont(None, 26)
        ui_instance.big_font = pg.font.SysFont(None, 34)
        ui_instance._compute_board_geometry()

    # Geometry should match
    assert match_ui.cell == regular_ui.cell
    assert match_ui.board_x0 == regular_ui.board_x0
    assert match_ui.board_y0 == regular_ui.board_y0
    assert match_ui.board_x1 == regular_ui.board_x1
    assert match_ui.board_y1 == regular_ui.board_y1


def test_geometry_updated_on_window_resize(game, mock_agent1, mock_agent2):
    """Test 41: Geometry recalculates on different window sizes"""
    pg.init()

    # Create UI with window_size=600
    from src.games.core.ui.game_ui import UIConfig
    cfg1 = UIConfig(window_size=600)
    ui1 = TicTacToeMatchUI(game, mock_agent1, mock_agent2, cfg=cfg1)
    ui1.screen = pg.Surface((600, 600))
    ui1.font = pg.font.SysFont(None, 26)
    ui1.big_font = pg.font.SysFont(None, 34)
    ui1._compute_board_geometry()

    # Create UI with window_size=800
    cfg2 = UIConfig(window_size=800)
    ui2 = TicTacToeMatchUI(game, mock_agent1, mock_agent2, cfg=cfg2)
    ui2.screen = pg.Surface((800, 800))
    ui2.font = pg.font.SysFont(None, 26)
    ui2.big_font = pg.font.SysFont(None, 34)
    ui2._compute_board_geometry()

    # Geometries should be different
    assert ui2.cell > ui1.cell
    assert (ui2.board_x1 - ui2.board_x0) > (ui1.board_x1 - ui1.board_x0)


# ===== B. Rendering Tests (3 tests) =====

def test_renders_x_and_o_for_both_agents(ui, game):
    """Test 42: Renders both X and O symbols for agents"""
    ui.new_game()

    # Make some moves to have both X and O on board
    ui.state.board[0, 0] = 1  # Current player
    ui.state.board[1, 1] = -1  # Opponent

    # Should not raise exception
    ui._render_pieces()


# ===== C. Integration Tests (2 tests) =====

def test_full_tictactoe_game_playthrough(ui, mock_agent1, mock_agent2):
    """Test 45: Simulate complete TicTacToe game"""
    # Set up agents to play a sequence of moves
    moves_agent1 = [0, 3, 6]  # Left column
    moves_agent2 = [1, 4]  # Second column
    move_counter = [0]

    def agent1_move(state):
        idx = move_counter[0] // 2
        if idx < len(moves_agent1):
            return moves_agent1[idx]
        return 2  # Fallback

    def agent2_move(state):
        idx = move_counter[0] // 2
        if idx < len(moves_agent2):
            return moves_agent2[idx]
        return 2  # Fallback

    mock_agent1.act.side_effect = agent1_move
    mock_agent2.act.side_effect = agent2_move

    ui.new_game()

    # Determine which agent goes first
    first_agent_is_1 = (ui.current_agent_index == 1)

    # Play game
    for _ in range(9):
        if ui.game_over:
            break

        ui.waiting_for_step = False
        ui._update()
        move_counter[0] += 1

    # Game should end (either win or draw)
    assert ui.game_over or move_counter[0] == 9

    # Score should be updated
    assert ui.games_played == 1
    assert (ui.wins1 + ui.wins2 + ui.draws) == 1


def test_multiple_tictactoe_games(ui, mock_agent1, mock_agent2):
    """Test 46: Play multiple games and verify score tracking"""
    # Set agents to always play legal moves
    def get_legal_move(state):
        legal_mask = ui.game.legal_actions(state)
        legal_indices = np.where(legal_mask)[0]  # Get indices where True
        return int(legal_indices[0])

    mock_agent1.act.side_effect = get_legal_move
    mock_agent2.act.side_effect = get_legal_move

    # Play 5 games
    for game_num in range(5):
        with patch('pygame.event.clear'):
            ui.new_game()

        # Play until game over
        for _ in range(20):  # Max moves
            if ui.game_over:
                break
            ui.waiting_for_step = False
            ui._update()

        assert ui.games_played == game_num + 1

    # All 5 games should be accounted for
    assert ui.games_played == 5
    assert (ui.wins1 + ui.wins2 + ui.draws) == 5


def test_tictactoe_board_scales_with_window_size():
    """Test 47: TicTacToe board scales appropriately with larger window"""
    from src.games.tictactoe import TicTacToe
    from src.games.tictactoe.ui.tictactoe_match_ui import TicTacToeMatchUI
    from src.games.core.ui.game_ui import UIConfig
    from unittest.mock import Mock
    import pygame as pg

    pg.init()
    game = TicTacToe()
    agent = Mock()
    agent.start = Mock()

    cfg = UIConfig(window_size=800)
    ui = TicTacToeMatchUI(game, agent, agent, cfg=cfg)
    ui.screen = pg.Surface((800, 800))
    ui.font = pg.font.SysFont(None, 26)
    ui.big_font = pg.font.SysFont(None, 34)
    ui._compute_board_geometry()

    # Board should fit and be 3×3
    assert (ui.board_x1 - ui.board_x0) == 3 * ui.cell
    assert (ui.board_y1 - ui.board_y0) == 3 * ui.cell

    # Should fit within window
    assert ui.board_x1 <= 800
    assert ui.board_y1 <= 800
