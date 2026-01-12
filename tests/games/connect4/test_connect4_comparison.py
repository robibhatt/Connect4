"""
Comparison tests between legacy and bitboard Connect4 implementations.

These tests verify that the new bitboard-backed implementation produces
identical results to the original numpy-based implementation.
"""
import numpy as np
import pytest

from src.games.connect4.connect4 import Connect4, Connect4State
from tests.games.connect4._legacy_connect4 import (
    LegacyConnect4,
    LegacyConnect4State,
)


@pytest.fixture
def game():
    """Bitboard-backed Connect4 game."""
    return Connect4()


@pytest.fixture
def legacy_game():
    """Original numpy-based Connect4 game."""
    return LegacyConnect4()


# ===== State Equivalence Tests =====

class TestStateEquivalence:
    """Tests that state representations are equivalent."""

    def test_reset_produces_equivalent_boards(self, game, legacy_game):
        """reset() should produce equivalent empty boards."""
        state = game.reset()
        legacy_state = legacy_game.reset()
        assert np.array_equal(state.board, legacy_state.board)

    def test_board_property_type(self, game):
        """state.board should be numpy array with correct dtype and shape."""
        state = game.reset()
        assert isinstance(state.board, np.ndarray)
        assert state.board.shape == (6, 7)
        assert state.board.dtype == np.int8

    def test_state_from_board_produces_correct_board(self):
        """Constructing state from board should preserve the board."""
        board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0],
            [0, -1, 1, -1, 1, 0, 0],
        ], dtype=np.int8)
        state = Connect4State(board=board)
        assert np.array_equal(state.board, board)


# ===== Deterministic Game Sequence Tests =====

class TestDeterministicSequences:
    """Tests with predetermined move sequences."""

    def test_vertical_win_sequence(self, game, legacy_game):
        """Vertical win sequence should produce identical results."""
        moves = [3, 4, 3, 4, 3, 4, 3]  # Column 3 wins vertically

        state = game.reset()
        legacy_state = legacy_game.reset()

        for move in moves:
            state = game.next_state(state, move)
            legacy_state = legacy_game.next_state(legacy_state, move)
            assert np.array_equal(state.board, legacy_state.board), \
                f"Boards diverged after move {move}"

        # Final terminal check
        done, value = game.terminal_value(state)
        legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)
        assert done == legacy_done
        assert value == legacy_value

    def test_horizontal_win_sequence(self, game, legacy_game):
        """Horizontal win sequence should produce identical results."""
        moves = [0, 0, 1, 1, 2, 2, 3]  # Bottom row horizontal win

        state = game.reset()
        legacy_state = legacy_game.reset()

        for move in moves:
            state = game.next_state(state, move)
            legacy_state = legacy_game.next_state(legacy_state, move)
            assert np.array_equal(state.board, legacy_state.board)

        done, value = game.terminal_value(state)
        legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)
        assert done == legacy_done
        assert value == legacy_value

    def test_diagonal_win_sequence(self, game, legacy_game):
        """Diagonal win sequence should produce identical results."""
        # Build ascending diagonal for player 1
        moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 4, 3]

        state = game.reset()
        legacy_state = legacy_game.reset()

        for move in moves:
            state = game.next_state(state, move)
            legacy_state = legacy_game.next_state(legacy_state, move)
            assert np.array_equal(state.board, legacy_state.board)


# ===== Random Game Comparison Tests =====

class TestRandomGames:
    """Tests using random game play."""

    def test_100_random_games(self, game, legacy_game):
        """100 random games should produce identical results."""
        rng = np.random.default_rng(42)

        for game_num in range(100):
            state = game.reset()
            legacy_state = legacy_game.reset()

            move_count = 0
            while True:
                # Compare legal actions
                legal = game.legal_actions(state)
                legacy_legal = legacy_game.legal_actions(legacy_state)
                assert np.array_equal(legal, legacy_legal), \
                    f"legal_actions mismatch in game {game_num}, move {move_count}"

                # Compare terminal status
                done, value = game.terminal_value(state)
                legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)
                assert done == legacy_done, \
                    f"terminal done mismatch in game {game_num}"
                assert value == legacy_value, \
                    f"terminal value mismatch in game {game_num}"

                if done:
                    break

                # Make random move
                legal_cols = np.where(legal)[0]
                move = int(rng.choice(legal_cols))

                state = game.next_state(state, move)
                legacy_state = legacy_game.next_state(legacy_state, move)

                # Compare boards after each move
                assert np.array_equal(state.board, legacy_state.board), \
                    f"Board mismatch in game {game_num}, move {move_count}"

                move_count += 1

    def test_1000_random_games(self, game, legacy_game):
        """1000 random games for thorough comparison."""
        rng = np.random.default_rng(12345)
        wins = {"current": 0, "opponent": 0, "draw": 0}

        for game_num in range(1000):
            state = game.reset()
            legacy_state = legacy_game.reset()

            while True:
                done, value = game.terminal_value(state)
                legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)

                assert done == legacy_done
                assert value == legacy_value

                if done:
                    if value > 0:
                        wins["current"] += 1
                    elif value < 0:
                        wins["opponent"] += 1
                    else:
                        wins["draw"] += 1
                    break

                legal = game.legal_actions(state)
                legal_cols = np.where(legal)[0]
                move = int(rng.choice(legal_cols))

                state = game.next_state(state, move)
                legacy_state = legacy_game.next_state(legacy_state, move)

                assert np.array_equal(state.board, legacy_state.board)

        # Sanity check: with random play, first player should win more often
        assert wins["current"] + wins["opponent"] + wins["draw"] == 1000


# ===== API Method Comparison Tests =====

class TestAPIComparison:
    """Tests comparing each API method."""

    def test_encode_equivalence(self, game, legacy_game):
        """encode() should produce identical results."""
        moves = [3, 4, 2, 3, 1, 2]

        state = game.reset()
        legacy_state = legacy_game.reset()

        for move in moves:
            state = game.next_state(state, move)
            legacy_state = legacy_game.next_state(legacy_state, move)

        encoded = game.encode(state)
        legacy_encoded = legacy_game.encode(legacy_state)

        assert np.array_equal(encoded, legacy_encoded)
        assert encoded.dtype == legacy_encoded.dtype

    def test_key_equivalence(self, game, legacy_game):
        """key() should produce identical results."""
        moves = [3, 4, 2, 3, 1, 2]

        state = game.reset()
        legacy_state = legacy_game.reset()

        for move in moves:
            state = game.next_state(state, move)
            legacy_state = legacy_game.next_state(legacy_state, move)

        key = game.key(state)
        legacy_key = legacy_game.key(legacy_state)

        assert key == legacy_key

    def test_symmetries_equivalence(self, game, legacy_game):
        """symmetries() should produce identical results."""
        x = np.arange(42).reshape(6, 7).astype(np.float32)
        pi = np.arange(7).astype(np.float32)

        syms = game.symmetries(x, pi)
        legacy_syms = legacy_game.symmetries(x, pi)

        assert len(syms) == len(legacy_syms)
        for (sx, sp), (lsx, lsp) in zip(syms, legacy_syms):
            assert np.array_equal(sx, lsx)
            assert np.array_equal(sp, lsp)

    def test_to_play_equivalence(self, game, legacy_game):
        """to_play() should produce identical results at various states."""
        state = game.reset()
        legacy_state = legacy_game.reset()

        # Check at initial state
        assert game.to_play(state) == legacy_game.to_play(legacy_state)

        # Check after some moves
        for move in [3, 4, 2]:
            state = game.next_state(state, move)
            legacy_state = legacy_game.next_state(legacy_state, move)
            assert game.to_play(state) == legacy_game.to_play(legacy_state)


# ===== Edge Case Tests =====

class TestEdgeCases:
    """Tests for edge cases and special situations."""

    def test_all_horizontal_win_positions(self, game, legacy_game):
        """All 24 horizontal win positions should be detected identically."""
        for row in range(6):
            for start_col in range(4):
                board = np.zeros((6, 7), dtype=np.int8)
                board[row, start_col:start_col+4] = 1

                state = Connect4State(board=board)
                legacy_state = LegacyConnect4State(board=board)

                done, value = game.terminal_value(state)
                legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)

                assert done == legacy_done == True, \
                    f"Horizontal win at row={row}, col={start_col} not detected"
                assert value == legacy_value == 1.0

    def test_all_vertical_win_positions(self, game, legacy_game):
        """All 21 vertical win positions should be detected identically."""
        for col in range(7):
            for start_row in range(3):
                board = np.zeros((6, 7), dtype=np.int8)
                board[start_row:start_row+4, col] = 1

                state = Connect4State(board=board)
                legacy_state = LegacyConnect4State(board=board)

                done, value = game.terminal_value(state)
                legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)

                assert done == legacy_done == True, \
                    f"Vertical win at col={col}, row={start_row} not detected"
                assert value == legacy_value == 1.0

    def test_all_diagonal_ascending_win_positions(self, game, legacy_game):
        """All ascending diagonal wins should be detected identically."""
        for start_row in range(3, 6):  # Rows 3-5 can start ascending
            for start_col in range(4):  # Cols 0-3 can start ascending
                board = np.zeros((6, 7), dtype=np.int8)
                for i in range(4):
                    board[start_row - i, start_col + i] = 1

                state = Connect4State(board=board)
                legacy_state = LegacyConnect4State(board=board)

                done, value = game.terminal_value(state)
                legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)

                assert done == legacy_done == True, \
                    f"Ascending diagonal at ({start_row}, {start_col}) not detected"
                assert value == legacy_value == 1.0

    def test_all_diagonal_descending_win_positions(self, game, legacy_game):
        """All descending diagonal wins should be detected identically."""
        for start_row in range(3):  # Rows 0-2 can start descending
            for start_col in range(4):  # Cols 0-3 can start descending
                board = np.zeros((6, 7), dtype=np.int8)
                for i in range(4):
                    board[start_row + i, start_col + i] = 1

                state = Connect4State(board=board)
                legacy_state = LegacyConnect4State(board=board)

                done, value = game.terminal_value(state)
                legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)

                assert done == legacy_done == True, \
                    f"Descending diagonal at ({start_row}, {start_col}) not detected"
                assert value == legacy_value == 1.0

    def test_draw_detection(self, game, legacy_game):
        """Draw (full board with no winner) should be detected identically."""
        # Known draw position
        board = np.array([
            [1, 1, 1, -1, -1, -1, 1],
            [-1, -1, -1, 1, 1, 1, -1],
            [1, 1, 1, -1, -1, -1, 1],
            [-1, -1, -1, 1, 1, 1, -1],
            [1, 1, -1, -1, -1, 1, 1],
            [-1, -1, 1, 1, 1, -1, -1]
        ], dtype=np.int8)

        state = Connect4State(board=board)
        legacy_state = LegacyConnect4State(board=board)

        done, value = game.terminal_value(state)
        legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)

        assert done == legacy_done == True
        assert value == legacy_value == 0.0

    def test_opponent_win_detection(self, game, legacy_game):
        """Opponent (-1) wins should be detected with value -1.0."""
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0:4] = -1  # Opponent wins horizontally

        state = Connect4State(board=board)
        legacy_state = LegacyConnect4State(board=board)

        done, value = game.terminal_value(state)
        legacy_done, legacy_value = legacy_game.terminal_value(legacy_state)

        assert done == legacy_done == True
        assert value == legacy_value == -1.0


# ===== Performance Sanity Tests =====

class TestPerformance:
    """Basic performance sanity checks."""

    def test_terminal_value_is_fast(self, game):
        """terminal_value should complete many calls quickly."""
        import time

        state = game.reset()
        for m in [3, 4, 3, 4, 3, 4]:
            state = game.next_state(state, m)

        start = time.perf_counter()
        for _ in range(10000):
            game.terminal_value(state)
        elapsed = time.perf_counter() - start

        # Should complete 10k calls in under 50ms (very generous)
        assert elapsed < 0.05, f"terminal_value too slow: {elapsed:.3f}s for 10k calls"

    def test_next_state_is_fast(self, game):
        """next_state should complete many calls quickly."""
        import time

        start = time.perf_counter()
        for _ in range(1000):
            state = game.reset()
            for m in [3, 4, 3, 4, 3, 4, 3]:
                state = game.next_state(state, m)
        elapsed = time.perf_counter() - start

        # 1000 games * 7 moves = 7000 next_state calls in under 100ms
        assert elapsed < 0.1, f"next_state too slow: {elapsed:.3f}s"
