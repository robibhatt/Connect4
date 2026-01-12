"""Tests for Connect4 game logic."""
import pytest
import numpy as np

from src.games.connect4 import Connect4, Connect4State


@pytest.fixture
def game():
    """Connect4 game instance."""
    return Connect4()


@pytest.fixture
def empty_state(game):
    """Empty Connect4 state."""
    return game.reset()


# ===== State Initialization Tests =====

def test_reset_returns_empty_board(game):
    """reset() should return empty 6x7 board."""
    state = game.reset()
    assert state.board.shape == (6, 7)
    assert state.board.dtype == np.int8
    assert (state.board == 0).all()


def test_reset_returns_connect4_state(game):
    """reset() should return Connect4State instance."""
    state = game.reset()
    assert isinstance(state, Connect4State)


# ===== Player Perspective Tests =====

def test_to_play_returns_plus_one(game, empty_state):
    """to_play() should always return +1 (canonical perspective)."""
    assert game.to_play(empty_state) == +1


# ===== Legal Actions Tests =====

def test_legal_actions_empty_board(game, empty_state):
    """All columns should be legal on empty board."""
    legal = game.legal_actions(empty_state)
    assert legal.shape == (7,)
    assert legal.dtype == np.bool_
    assert legal.all(), "All columns should be legal on empty board"


def test_legal_actions_partial_board(game):
    """Legal actions should include columns with space."""
    # Fill column 0 completely
    board = np.zeros((6, 7), dtype=np.int8)
    board[:, 0] = 1  # Column 0 is full
    state = Connect4State(board=board)

    legal = game.legal_actions(state)
    expected = np.ones(7, dtype=np.bool_)
    expected[0] = False  # Column 0 is not legal

    assert np.array_equal(legal, expected)


def test_legal_actions_full_board(game):
    """No columns should be legal on full board."""
    board = np.ones((6, 7), dtype=np.int8)
    state = Connect4State(board=board)

    legal = game.legal_actions(state)
    assert not legal.any(), "No columns should be legal on full board"


def test_legal_actions_column_almost_full(game):
    """Column with 5 pieces should still be legal."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[1:, 3] = 1  # Fill rows 1-5 of column 3 (top row is empty)
    state = Connect4State(board=board)

    legal = game.legal_actions(state)
    assert legal[3], "Column 3 should be legal with one space remaining"


def test_legal_actions_exactly_full_column(game):
    """Column with all 6 pieces should become illegal."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[:, 2] = 1  # Fill all 6 rows of column 2
    state = Connect4State(board=board)

    legal = game.legal_actions(state)
    assert not legal[2], "Column 2 should be illegal when full"


# ===== Gravity Mechanics Tests =====

def test_next_state_piece_falls_to_bottom(game, empty_state):
    """Piece should fall to bottom row on empty board."""
    new_state = game.next_state(empty_state, 3)

    # Bottom row is row 5 (0-indexed from top)
    # After perspective flip, our piece becomes -1
    assert new_state.board[5, 3] == -1
    # All other cells should be 0
    assert np.sum(new_state.board != 0) == 1


def test_next_state_piece_stacks_on_previous(game):
    """Piece should stack on top of existing pieces."""
    # Place one piece in column 2
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 2] = -1  # Opponent's piece at bottom
    state = Connect4State(board=board)

    # Place another piece in same column
    new_state = game.next_state(state, 2)

    # New piece should be at row 4 (one above row 5)
    assert new_state.board[4, 2] == -1


def test_next_state_multiple_stacks(game):
    """Multiple pieces stack correctly in same column."""
    state = game.reset()

    # Play column 0 three times alternating players
    # Each next_state flips all pieces, so perspective changes each time
    state1 = game.next_state(state, 0)   # Places +1 at (5,0), flips → -1
    state2 = game.next_state(state1, 0)  # Places +1 at (4,0), flips → -1 at (4,0), +1 at (5,0)
    state3 = game.next_state(state2, 0)  # Places +1 at (3,0), flips → -1 at (3,0), +1 at (4,0), -1 at (5,0)

    # Should have 3 pieces stacked in column 0 with alternating perspectives
    assert state3.board[5, 0] == -1  # Bottom (flipped 3 times: +1 → -1 → +1 → -1)
    assert state3.board[4, 0] == +1  # Middle (flipped 2 times: +1 → -1 → +1)
    assert state3.board[3, 0] == -1  # Top (flipped 1 time: +1 → -1)


# ===== State Transition Tests =====

def test_next_state_flips_perspective(game, empty_state):
    """next_state() should flip board perspective."""
    new_state = game.next_state(empty_state, 0)

    # The piece we placed should appear as opponent's piece (-1) after flip
    assert new_state.board[5, 0] == -1

    # All other squares should still be 0
    expected = np.zeros((6, 7), dtype=np.int8)
    expected[5, 0] = -1
    assert np.array_equal(new_state.board, expected)


def test_next_state_immutability(game, empty_state):
    """next_state() should not modify original state."""
    original_board = empty_state.board.copy()
    game.next_state(empty_state, 0)

    assert np.array_equal(empty_state.board, original_board)


def test_next_state_full_column_raises(game):
    """next_state() should raise ValueError for full column."""
    # Fill column 0 completely
    board = np.zeros((6, 7), dtype=np.int8)
    board[:, 0] = 1
    state = Connect4State(board=board)

    with pytest.raises(ValueError, match="column 0 is full"):
        game.next_state(state, 0)


def test_next_state_out_of_bounds_negative(game, empty_state):
    """next_state() should raise ValueError for negative action."""
    with pytest.raises(ValueError, match="out of range"):
        game.next_state(empty_state, -1)


def test_next_state_out_of_bounds_too_large(game, empty_state):
    """next_state() should raise ValueError for action >= 7."""
    with pytest.raises(ValueError, match="out of range"):
        game.next_state(empty_state, 7)


# ===== Terminal Detection - Horizontal Wins =====

def test_terminal_horizontal_win_current_player(game):
    """Detect horizontal 4-in-a-row for current player (+1)."""
    # Bottom row: four +1s starting at column 1
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 1:5] = 1
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == +1.0


def test_terminal_horizontal_win_opponent(game):
    """Detect horizontal 4-in-a-row for opponent (-1)."""
    # Middle of board: four -1s
    board = np.zeros((6, 7), dtype=np.int8)
    board[3, 2:6] = -1
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == -1.0


def test_terminal_horizontal_win_all_positions(game):
    """Test horizontal win detection at various positions."""
    # Test multiple starting positions
    for row in [0, 2, 5]:
        for start_col in [0, 1, 2, 3]:
            board = np.zeros((6, 7), dtype=np.int8)
            board[row, start_col:start_col+4] = 1
            state = Connect4State(board=board)

            done, value = game.terminal_value(state)
            assert done is True, f"Horizontal win not detected at row={row}, col={start_col}"
            assert value == +1.0


# ===== Terminal Detection - Vertical Wins =====

def test_terminal_vertical_win_current_player(game):
    """Detect vertical 4-in-a-row for current player (+1)."""
    # Column 3: four +1s in bottom rows
    board = np.zeros((6, 7), dtype=np.int8)
    board[2:6, 3] = 1
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == +1.0


def test_terminal_vertical_win_opponent(game):
    """Detect vertical 4-in-a-row for opponent (-1)."""
    # Column 0: four -1s
    board = np.zeros((6, 7), dtype=np.int8)
    board[1:5, 0] = -1
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == -1.0


def test_terminal_vertical_win_all_columns(game):
    """Test vertical win detection in all columns."""
    for col in range(7):
        board = np.zeros((6, 7), dtype=np.int8)
        board[2:6, col] = 1  # Bottom 4 rows
        state = Connect4State(board=board)

        done, value = game.terminal_value(state)
        assert done is True, f"Vertical win not detected in column {col}"
        assert value == +1.0


# ===== Terminal Detection - Diagonal Wins =====

def test_terminal_diagonal_win_ascending(game):
    """Detect ascending diagonal 4-in-a-row (bottom-left to top-right)."""
    # Diagonal from (5,0) to (2,3)
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = 1
    board[4, 1] = 1
    board[3, 2] = 1
    board[2, 3] = 1
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == +1.0


def test_terminal_diagonal_win_descending(game):
    """Detect descending diagonal 4-in-a-row (top-left to bottom-right)."""
    # Diagonal from (2,0) to (5,3)
    board = np.zeros((6, 7), dtype=np.int8)
    board[2, 0] = -1
    board[3, 1] = -1
    board[4, 2] = -1
    board[5, 3] = -1
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == -1.0


def test_terminal_diagonal_win_corner(game):
    """Detect diagonal win in corner positions."""
    # Bottom-right corner ascending diagonal
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 3] = 1
    board[4, 4] = 1
    board[3, 5] = 1
    board[2, 6] = 1
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is True
    assert value == +1.0


# ===== Terminal Detection - Draw =====

def test_terminal_draw_full_board_no_winner(game):
    """Detect draw when board is full with no 4-in-a-row."""
    # Create pattern with no 4-in-a-row (careful to avoid diagonals too)
    # Pattern: groups of 3 alternating
    board = np.array([
        [1, 1, 1, -1, -1, -1, 1],
        [-1, -1, -1, 1, 1, 1, -1],
        [1, 1, 1, -1, -1, -1, 1],
        [-1, -1, -1, 1, 1, 1, -1],
        [1, 1, -1, -1, -1, 1, 1],
        [-1, -1, 1, 1, 1, -1, -1]
    ], dtype=np.int8)
    state = Connect4State(board=board)

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
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0:3] = [1, -1, 1]
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is False
    assert value == 0.0


def test_terminal_non_terminal_three_in_a_row(game):
    """Three in a row should not be terminal (needs four)."""
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0:3] = 1  # Only 3 in a row
    state = Connect4State(board=board)

    done, value = game.terminal_value(state)
    assert done is False
    assert value == 0.0


# ===== Encoding Tests =====

def test_encode_returns_correct_shape(game, empty_state):
    """encode() should return float32 array of shape (6, 7)."""
    encoded = game.encode(empty_state)
    assert encoded.shape == (6, 7)
    assert encoded.dtype == np.float32


def test_encode_matches_board_values(game):
    """encode() should return board values as float32."""
    board = np.array([
        [1, -1, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 0, 0]
    ], dtype=np.int8)
    state = Connect4State(board=board)

    encoded = game.encode(state)
    expected = board.astype(np.float32)

    assert np.array_equal(encoded, expected)


# ===== State Key Tests =====

def test_key_returns_int64(game, empty_state):
    """key() should return np.int64."""
    key = game.key(empty_state)
    assert isinstance(key, np.int64), f"Expected np.int64, got {type(key)}"


def test_key_consistent_for_same_state(game, empty_state):
    """key() should return same value for same state."""
    key1 = game.key(empty_state)
    key2 = game.key(empty_state)
    assert key1 == key2
    assert isinstance(key1, np.int64)


def test_key_different_for_different_states(game, empty_state):
    """key() should return different values for different states."""
    key1 = game.key(empty_state)

    new_state = game.next_state(empty_state, 0)
    key2 = game.key(new_state)

    assert key1 != key2


def test_key_positive_value(game, empty_state):
    """key() should return positive value."""
    key = game.key(empty_state)
    assert key >= 0, f"Key must be non-negative, got {key}"


def test_key_fits_in_int64(game, empty_state):
    """key() should fit in signed 64-bit integer range."""
    key = game.key(empty_state)
    assert key <= np.iinfo(np.int64).max, f"Key exceeds int64 max"


def test_key_deterministic_across_calls(game):
    """key() should be deterministic (no randomness between calls)."""
    state = game.reset()
    keys = [game.key(state) for _ in range(100)]
    assert all(k == keys[0] for k in keys), "Keys must be identical for same state"


def test_key_different_for_symmetric_boards(game):
    """key() should distinguish left-right symmetric boards."""
    # Place piece in column 0
    state1 = game.next_state(game.reset(), 0)
    # Place piece in column 6
    state2 = game.next_state(game.reset(), 6)
    assert game.key(state1) != game.key(state2), "Symmetric boards must have different keys"


def test_key_usable_as_dict_key(game, empty_state):
    """key() should work as dictionary key for transposition tables."""
    key = game.key(empty_state)
    table = {key: "test_value"}
    assert table[key] == "test_value"
    assert key in table


def test_key_storable_in_numpy_array(game):
    """key() should be storable in np.int64 numpy array."""
    states = [game.reset()]
    for i in range(3):
        states.append(game.next_state(states[-1], i))

    keys = np.array([game.key(s) for s in states], dtype=np.int64)
    assert keys.dtype == np.int64
    assert len(keys) == 4


# ===== Symmetries Tests =====

def test_symmetries_returns_2_symmetries(game):
    """symmetries() should return 2 transformations (original + flip)."""
    x = np.zeros((6, 7), dtype=np.float32)
    pi = np.ones(7, dtype=np.float32)

    symmetries = game.symmetries(x, pi)
    assert len(symmetries) == 2


def test_symmetries_structure(game):
    """Each symmetry should be (board, policy) tuple."""
    x = np.zeros((6, 7), dtype=np.float32)
    pi = np.ones(7, dtype=np.float32)

    symmetries = game.symmetries(x, pi)
    for sym_x, sym_pi in symmetries:
        assert sym_x.shape == (6, 7)
        assert sym_x.dtype == np.float32
        assert sym_pi.shape == (7,)
        assert sym_pi.dtype == np.float32


def test_symmetries_first_is_original(game):
    """First symmetry should be original (no transformation)."""
    x = np.arange(42).reshape(6, 7).astype(np.float32)
    pi = np.arange(7).astype(np.float32)

    symmetries = game.symmetries(x, pi)
    sym_x_0, sym_pi_0 = symmetries[0]

    assert np.array_equal(sym_x_0, x)
    assert np.array_equal(sym_pi_0, pi)


def test_symmetries_second_is_flipped(game):
    """Second symmetry should be horizontally flipped."""
    x = np.arange(42).reshape(6, 7).astype(np.float32)
    pi = np.arange(7).astype(np.float32)

    symmetries = game.symmetries(x, pi)
    sym_x_1, sym_pi_1 = symmetries[1]

    expected_x = np.fliplr(x)
    expected_pi = np.flip(pi)

    assert np.array_equal(sym_x_1, expected_x)
    assert np.array_equal(sym_pi_1, expected_pi)


def test_symmetries_preserve_semantics(game):
    """Symmetries should preserve game semantics."""
    # Board with piece in left column
    x = np.zeros((6, 7), dtype=np.float32)
    x[5, 0] = 1.0  # Bottom-left

    # Policy favoring left column (action 0)
    pi = np.zeros(7, dtype=np.float32)
    pi[0] = 1.0

    symmetries = game.symmetries(x, pi)

    # Original should have piece at (5,0) and policy at 0
    sym_x_0, sym_pi_0 = symmetries[0]
    assert sym_x_0[5, 0] == 1.0
    assert sym_pi_0[0] == 1.0

    # Flipped should have piece at (5,6) and policy at 6
    sym_x_1, sym_pi_1 = symmetries[1]
    assert sym_x_1[5, 6] == 1.0
    assert sym_pi_1[6] == 1.0


# ===== Action Indexing Tests =====

def test_action_0_is_leftmost_column(game, empty_state):
    """Action 0 should correspond to leftmost column."""
    new_state = game.next_state(empty_state, 0)
    # Should be at bottom of column 0
    assert new_state.board[5, 0] == -1


def test_action_3_is_middle_column(game, empty_state):
    """Action 3 should correspond to middle column."""
    new_state = game.next_state(empty_state, 3)
    # Should be at bottom of column 3
    assert new_state.board[5, 3] == -1


def test_action_6_is_rightmost_column(game, empty_state):
    """Action 6 should correspond to rightmost column."""
    new_state = game.next_state(empty_state, 6)
    # Should be at bottom of column 6
    assert new_state.board[5, 6] == -1


def test_all_actions_map_to_correct_columns(game, empty_state):
    """Test that all 7 actions map to correct columns."""
    for action in range(7):
        new_state = game.next_state(empty_state, action)

        # Check that only the bottom cell of the target column is marked
        expected = np.zeros((6, 7), dtype=np.int8)
        expected[5, action] = -1
        assert np.array_equal(new_state.board, expected), \
            f"Action {action} should place piece in column {action}"
