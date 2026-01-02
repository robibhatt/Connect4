"""Tests for TicTacToe game logic."""
import pytest
import numpy as np

from src.games.tictactoe import TicTacToe, TicTacToeState


@pytest.fixture
def game():
    """TicTacToe game instance."""
    return TicTacToe()


@pytest.fixture
def empty_state(game):
    """Empty TicTacToe state."""
    return game.reset()


# ===== State Initialization Tests =====

def test_reset_returns_empty_board(game):
    """reset() should return empty 3x3 board."""
    state = game.reset()
    assert state.board.shape == (3, 3)
    assert state.board.dtype == np.int8
    assert (state.board == 0).all()


def test_reset_returns_tictactoe_state(game):
    """reset() should return TicTacToeState instance."""
    state = game.reset()
    assert isinstance(state, TicTacToeState)


# ===== Player Perspective Tests =====

def test_to_play_returns_plus_one(game, empty_state):
    """to_play() should always return +1 (canonical perspective)."""
    assert game.to_play(empty_state) == +1


# ===== Legal Actions Tests =====

def test_legal_actions_empty_board(game, empty_state):
    """All actions should be legal on empty board."""
    legal = game.legal_actions(empty_state)
    assert legal.shape == (9,)
    assert legal.dtype == np.bool_
    assert legal.all(), "All squares should be legal on empty board"


def test_legal_actions_partial_board(game):
    """Legal actions should exclude occupied squares."""
    # Create a board with X in center (action 4)
    board = np.zeros((3, 3), dtype=np.int8)
    board[1, 1] = 1  # Center square occupied
    state = TicTacToeState(board=board)

    legal = game.legal_actions(state)
    expected = np.ones(9, dtype=np.bool_)
    expected[4] = False  # Center is not legal

    assert np.array_equal(legal, expected)


def test_legal_actions_full_board(game):
    """No actions should be legal on full board."""
    board = np.ones((3, 3), dtype=np.int8)
    state = TicTacToeState(board=board)

    legal = game.legal_actions(state)
    assert not legal.any(), "No squares should be legal on full board"


# ===== State Transition Tests =====

def test_next_state_places_piece(game, empty_state):
    """next_state() should place piece at specified location."""
    # Play action 0 (top-left corner)
    new_state = game.next_state(empty_state, 0)

    # After perspective flip, opponent's piece shows as -1
    assert new_state.board[0, 0] == -1


def test_next_state_flips_perspective(game, empty_state):
    """next_state() should flip board perspective."""
    # Place piece at center (action 4)
    new_state = game.next_state(empty_state, 4)

    # The piece we placed should appear as opponent's piece (-1) after flip
    assert new_state.board[1, 1] == -1

    # All other squares should still be 0
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[1, 1] = -1
    assert np.array_equal(new_state.board, expected)


def test_next_state_immutability(game, empty_state):
    """next_state() should not modify original state."""
    original_board = empty_state.board.copy()
    game.next_state(empty_state, 0)

    assert np.array_equal(empty_state.board, original_board)


def test_next_state_illegal_move_occupied_square(game):
    """next_state() should raise ValueError for occupied square."""
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, 0] = 1
    state = TicTacToeState(board=board)

    with pytest.raises(ValueError, match="Illegal move"):
        game.next_state(state, 0)


def test_next_state_out_of_bounds_negative(game, empty_state):
    """next_state() should raise ValueError for negative action."""
    with pytest.raises(ValueError, match="out of range"):
        game.next_state(empty_state, -1)


def test_next_state_out_of_bounds_too_large(game, empty_state):
    """next_state() should raise ValueError for action >= 9."""
    with pytest.raises(ValueError, match="out of range"):
        game.next_state(empty_state, 9)


# ===== Terminal Detection - Wins =====

def test_terminal_row_win_current_player(game):
    """Detect row win for current player (+1)."""
    # Top row: all +1
    board = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == +1.0


def test_terminal_row_win_opponent(game):
    """Detect row win for opponent (-1)."""
    # Middle row: all -1
    board = np.array([
        [0, 0, 0],
        [-1, -1, -1],
        [0, 0, 0]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == -1.0


def test_terminal_column_win_current_player(game):
    """Detect column win for current player (+1)."""
    # Left column: all +1
    board = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == +1.0


def test_terminal_column_win_opponent(game):
    """Detect column win for opponent (-1)."""
    # Right column: all -1
    board = np.array([
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == -1.0


def test_terminal_diagonal_win_top_left_to_bottom_right(game):
    """Detect diagonal win (top-left to bottom-right)."""
    # Main diagonal: all +1
    board = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == +1.0


def test_terminal_diagonal_win_top_right_to_bottom_left(game):
    """Detect diagonal win (top-right to bottom-left)."""
    # Anti-diagonal: all -1
    board = np.array([
        [0, 0, -1],
        [0, -1, 0],
        [-1, 0, 0]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == -1.0


def test_terminal_all_rows_win_detection(game):
    """Test win detection for all three rows."""
    for row in range(3):
        board = np.zeros((3, 3), dtype=np.int8)
        board[row, :] = 1
        state = TicTacToeState(board=board)

        done, value = game.terminal_value(state)
        assert done is True, f"Row {row} win not detected"
        assert value == +1.0


def test_terminal_all_columns_win_detection(game):
    """Test win detection for all three columns."""
    for col in range(3):
        board = np.zeros((3, 3), dtype=np.int8)
        board[:, col] = 1
        state = TicTacToeState(board=board)

        done, value = game.terminal_value(state)
        assert done is True, f"Column {col} win not detected"
        assert value == +1.0


# ===== Terminal Detection - Draw =====

def test_terminal_draw_full_board_no_winner(game):
    """Detect draw when board is full with no winner."""
    # Full board with no three-in-a-row
    board = np.array([
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, -1]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == 0.0


# ===== Terminal Detection - Non-Terminal =====

def test_terminal_non_terminal_empty_board(game, empty_state):
    """Empty board should not be terminal."""
    done, value = game.terminal_value(empty_state)
    assert done is False
    assert value == 0.0


def test_terminal_non_terminal_partial_game(game):
    """Partial game should not be terminal."""
    board = np.array([
        [1, -1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    done, value = game.terminal_value(state)
    assert done is False
    assert value == 0.0


# ===== Encoding Tests =====

def test_encode_returns_correct_shape(game, empty_state):
    """encode() should return float32 array of shape (3, 3)."""
    encoded = game.encode(empty_state)
    assert encoded.shape == (3, 3)
    assert encoded.dtype == np.float32


def test_encode_matches_board_values(game):
    """encode() should return board values as float32."""
    board = np.array([
        [1, -1, 0],
        [0, 1, -1],
        [1, 0, -1]
    ], dtype=np.int8)
    state = TicTacToeState(board=board)

    encoded = game.encode(state)
    expected = board.astype(np.float32)

    assert np.array_equal(encoded, expected)


# ===== State Key Tests =====

def test_key_returns_bytes(game, empty_state):
    """key() should return bytes."""
    key = game.key(empty_state)
    assert isinstance(key, bytes)


def test_key_consistent_for_same_state(game, empty_state):
    """key() should return same bytes for same state."""
    key1 = game.key(empty_state)
    key2 = game.key(empty_state)
    assert key1 == key2


def test_key_different_for_different_states(game, empty_state):
    """key() should return different bytes for different states."""
    key1 = game.key(empty_state)

    # Create different state
    new_state = game.next_state(empty_state, 0)
    key2 = game.key(new_state)

    assert key1 != key2


# ===== Symmetries Tests =====

def test_symmetries_returns_8_symmetries(game):
    """symmetries() should return 8 transformations."""
    x = np.zeros((3, 3), dtype=np.float32)
    pi = np.ones(9, dtype=np.float32)

    symmetries = game.symmetries(x, pi)
    assert len(symmetries) == 8


def test_symmetries_structure(game):
    """Each symmetry should be (board, policy) tuple."""
    x = np.zeros((3, 3), dtype=np.float32)
    pi = np.ones(9, dtype=np.float32)

    symmetries = game.symmetries(x, pi)
    for sym_x, sym_pi in symmetries:
        assert sym_x.shape == (3, 3)
        assert sym_x.dtype == np.float32
        assert sym_pi.shape == (9,)
        assert sym_pi.dtype == np.float32


def test_symmetries_preserve_semantics(game):
    """Symmetries should preserve game semantics."""
    # Create a board with top-left corner marked
    x = np.zeros((3, 3), dtype=np.float32)
    x[0, 0] = 1.0

    # Policy favoring top-left (action 0)
    pi = np.zeros(9, dtype=np.float32)
    pi[0] = 1.0

    symmetries = game.symmetries(x, pi)

    # Each symmetry should have exactly one cell == 1.0 in board
    # and exactly one position == 1.0 in policy
    for sym_x, sym_pi in symmetries:
        assert np.sum(sym_x == 1.0) == 1
        assert np.sum(sym_pi == 1.0) == 1


def test_symmetries_rotations_and_flips(game):
    """Symmetries should include rotations and flips."""
    # Distinctive pattern to check transformations
    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)
    pi = np.arange(9, dtype=np.float32)

    symmetries = game.symmetries(x, pi)

    # First symmetry should be original (rotation 0, no flip)
    sym_x_0, sym_pi_0 = symmetries[0]
    assert np.array_equal(sym_x_0, x)
    assert np.array_equal(sym_pi_0, pi)


# ===== Action Indexing Tests =====

def test_action_0_is_top_left(game, empty_state):
    """Action 0 should correspond to top-left cell (0,0)."""
    new_state = game.next_state(empty_state, 0)
    # After perspective flip, our piece becomes -1
    assert new_state.board[0, 0] == -1


def test_action_4_is_center(game, empty_state):
    """Action 4 should correspond to center cell (1,1)."""
    new_state = game.next_state(empty_state, 4)
    assert new_state.board[1, 1] == -1


def test_action_8_is_bottom_right(game, empty_state):
    """Action 8 should correspond to bottom-right cell (2,2)."""
    new_state = game.next_state(empty_state, 8)
    assert new_state.board[2, 2] == -1


def test_all_actions_map_correctly(game, empty_state):
    """Test that all 9 actions map to correct cells."""
    for action in range(9):
        row, col = divmod(action, 3)
        new_state = game.next_state(empty_state, action)

        # Check that only the intended cell is marked
        expected = np.zeros((3, 3), dtype=np.int8)
        expected[row, col] = -1
        assert np.array_equal(new_state.board, expected), \
            f"Action {action} should map to ({row},{col})"
