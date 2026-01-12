"""Tests for Connect4 bitboard utilities."""
import pytest
import numpy as np

from src.games.connect4.bitboard import (
    bit_index,
    numpy_to_bitboards,
    bitboards_to_numpy,
    has_won,
    get_legal_columns,
    drop_piece,
    column_height,
)


# ===== Coordinate Conversion Tests =====

class TestBitIndex:
    """Tests for bit_index coordinate conversion."""

    def test_bottom_left(self):
        """Bit index for bottom-left (row=5, col=0) should be 0."""
        assert bit_index(row=5, col=0) == 0

    def test_top_left(self):
        """Bit index for top-left (row=0, col=0) should be 5."""
        assert bit_index(row=0, col=0) == 5

    def test_bottom_right(self):
        """Bit index for bottom-right (row=5, col=6) should be 42."""
        assert bit_index(row=5, col=6) == 42

    def test_top_right(self):
        """Bit index for top-right (row=0, col=6) should be 47."""
        assert bit_index(row=0, col=6) == 47

    def test_all_positions_valid(self):
        """All 42 positions should map to unique bits in range [0, 48]."""
        bits = set()
        sentinel_bits = {6, 13, 20, 27, 34, 41, 48}

        for row in range(6):
            for col in range(7):
                bit = bit_index(row, col)
                assert 0 <= bit <= 47, f"Bit {bit} out of range for ({row}, {col})"
                assert bit not in sentinel_bits, f"Hit sentinel bit {bit} for ({row}, {col})"
                assert bit not in bits, f"Duplicate bit {bit} for ({row}, {col})"
                bits.add(bit)

        assert len(bits) == 42, "Should have exactly 42 unique bit positions"


# ===== Numpy <-> Bitboard Conversion Tests =====

class TestNumpyToBitboards:
    """Tests for numpy_to_bitboards conversion."""

    def test_empty_board(self):
        """Empty numpy board should produce zero bitboards."""
        board = np.zeros((6, 7), dtype=np.int8)
        bb_current, bb_opponent = numpy_to_bitboards(board)
        assert bb_current == 0
        assert bb_opponent == 0

    def test_single_current_player_piece(self):
        """Single +1 piece should set correct bit in bb_current."""
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0] = 1  # Bottom-left
        bb_current, bb_opponent = numpy_to_bitboards(board)
        assert bb_current == 1  # bit 0
        assert bb_opponent == 0

    def test_single_opponent_piece(self):
        """Single -1 piece should set correct bit in bb_opponent."""
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0] = -1  # Bottom-left
        bb_current, bb_opponent = numpy_to_bitboards(board)
        assert bb_current == 0
        assert bb_opponent == 1  # bit 0

    def test_multiple_pieces(self):
        """Multiple pieces should set correct bits."""
        board = np.zeros((6, 7), dtype=np.int8)
        board[5, 0] = 1   # bit 0
        board[4, 0] = -1  # bit 1
        board[5, 1] = 1   # bit 7
        bb_current, bb_opponent = numpy_to_bitboards(board)
        assert bb_current == (1 << 0) | (1 << 7)  # bits 0 and 7
        assert bb_opponent == (1 << 1)  # bit 1


class TestBitboardsToNumpy:
    """Tests for bitboards_to_numpy conversion."""

    def test_empty_bitboards(self):
        """Zero bitboards should produce empty numpy board."""
        board = bitboards_to_numpy(0, 0)
        assert board.shape == (6, 7)
        assert board.dtype == np.int8
        assert (board == 0).all()

    def test_single_current_piece(self):
        """Single bit in bb_current should produce +1 in correct position."""
        board = bitboards_to_numpy(bb_current=1, bb_opponent=0)  # bit 0 = (5, 0)
        assert board[5, 0] == 1
        assert np.sum(board != 0) == 1

    def test_single_opponent_piece(self):
        """Single bit in bb_opponent should produce -1 in correct position."""
        board = bitboards_to_numpy(bb_current=0, bb_opponent=1)  # bit 0 = (5, 0)
        assert board[5, 0] == -1
        assert np.sum(board != 0) == 1


class TestRoundtripConversion:
    """Tests for roundtrip conversion consistency."""

    def test_roundtrip_empty(self):
        """Empty board roundtrip should work."""
        board = np.zeros((6, 7), dtype=np.int8)
        bb_curr, bb_opp = numpy_to_bitboards(board)
        board_back = bitboards_to_numpy(bb_curr, bb_opp)
        assert np.array_equal(board, board_back)

    def test_roundtrip_complex_position(self):
        """Complex position roundtrip should preserve board exactly."""
        board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0],
            [0, -1, 1, -1, 1, 0, 0],
        ], dtype=np.int8)
        bb_curr, bb_opp = numpy_to_bitboards(board)
        board_back = bitboards_to_numpy(bb_curr, bb_opp)
        assert np.array_equal(board, board_back)

    def test_roundtrip_all_42_positions(self):
        """Each of the 42 positions should roundtrip correctly for both players."""
        for row in range(6):
            for col in range(7):
                for player in [1, -1]:
                    board = np.zeros((6, 7), dtype=np.int8)
                    board[row, col] = player
                    bb_curr, bb_opp = numpy_to_bitboards(board)
                    board_back = bitboards_to_numpy(bb_curr, bb_opp)
                    assert np.array_equal(board, board_back), \
                        f"Roundtrip failed for ({row}, {col}) player={player}"


# ===== Win Detection Tests =====

class TestHasWonEmpty:
    """Tests for win detection on empty/partial boards."""

    def test_empty_no_win(self):
        """Empty bitboard should not be a win."""
        assert has_won(0) is False

    def test_single_piece_no_win(self):
        """Single piece should not be a win."""
        assert has_won(1) is False

    def test_three_in_row_no_win(self):
        """Three in a row should NOT be a win."""
        # Bits 0, 7, 14 (only 3 horizontal)
        bb = (1 << 0) | (1 << 7) | (1 << 14)
        assert has_won(bb) is False


class TestHasWonHorizontal:
    """Tests for horizontal win detection."""

    def test_horizontal_bottom_row_left(self):
        """Horizontal 4-in-a-row at bottom row, columns 0-3."""
        # Bits 0, 7, 14, 21 (columns 0-3, row 5)
        bb = (1 << 0) | (1 << 7) | (1 << 14) | (1 << 21)
        assert has_won(bb) is True

    def test_horizontal_bottom_row_right(self):
        """Horizontal 4-in-a-row at bottom row, columns 3-6."""
        # Bits 21, 28, 35, 42 (columns 3-6, row 5)
        bb = (1 << 21) | (1 << 28) | (1 << 35) | (1 << 42)
        assert has_won(bb) is True

    def test_horizontal_top_row(self):
        """Horizontal 4-in-a-row at top row."""
        # Bits 5, 12, 19, 26 (columns 0-3, row 0)
        bb = (1 << 5) | (1 << 12) | (1 << 19) | (1 << 26)
        assert has_won(bb) is True

    def test_horizontal_all_positions(self):
        """Test horizontal win at every valid starting position."""
        for row in range(6):
            for start_col in range(4):  # Columns 0-3 can start horizontal wins
                bb = 0
                for c in range(4):
                    bb |= (1 << bit_index(row, start_col + c))
                assert has_won(bb), \
                    f"Horizontal at row={row}, start_col={start_col} not detected"


class TestHasWonVertical:
    """Tests for vertical win detection."""

    def test_vertical_column_0_bottom(self):
        """Vertical 4-in-a-row in column 0, rows 2-5."""
        # Bits 0, 1, 2, 3 (column 0, rows 5-2)
        bb = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
        assert has_won(bb) is True

    def test_vertical_column_0_top(self):
        """Vertical 4-in-a-row in column 0, rows 0-3."""
        # Bits 2, 3, 4, 5 (column 0, rows 3-0)
        bb = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        assert has_won(bb) is True

    def test_vertical_all_positions(self):
        """Test vertical win at every valid starting position."""
        for col in range(7):
            for start_row in range(3):  # Rows 0-2 can start vertical wins
                bb = 0
                for r in range(4):
                    bb |= (1 << bit_index(start_row + r, col))
                assert has_won(bb), \
                    f"Vertical at col={col}, start_row={start_row} not detected"


class TestHasWonDiagonal:
    """Tests for diagonal win detection."""

    def test_diagonal_ascending(self):
        """Ascending diagonal (/) from (5,0) to (2,3)."""
        # (5,0)=0, (4,1)=8, (3,2)=16, (2,3)=24
        bb = (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24)
        assert has_won(bb) is True

    def test_diagonal_descending(self):
        """Descending diagonal (\\) from (2,0) to (5,3)."""
        # (2,0)=3, (3,1)=9, (4,2)=15, (5,3)=21
        bb = (1 << 3) | (1 << 9) | (1 << 15) | (1 << 21)
        assert has_won(bb) is True

    def test_diagonal_ascending_corner(self):
        """Ascending diagonal in bottom-right corner."""
        # (5,3)=21, (4,4)=29, (3,5)=37, (2,6)=45
        bb = (1 << 21) | (1 << 29) | (1 << 37) | (1 << 45)
        assert has_won(bb) is True

    def test_diagonal_all_ascending_positions(self):
        """Test ascending diagonal win at every valid starting position."""
        # Ascending: row decreases, col increases
        for start_row in range(3, 6):  # Rows 3-5 can start ascending
            for start_col in range(4):  # Cols 0-3 can start ascending
                bb = 0
                for i in range(4):
                    bb |= (1 << bit_index(start_row - i, start_col + i))
                assert has_won(bb), \
                    f"Ascending diagonal at ({start_row}, {start_col}) not detected"

    def test_diagonal_all_descending_positions(self):
        """Test descending diagonal win at every valid starting position."""
        # Descending: row increases, col increases
        for start_row in range(3):  # Rows 0-2 can start descending
            for start_col in range(4):  # Cols 0-3 can start descending
                bb = 0
                for i in range(4):
                    bb |= (1 << bit_index(start_row + i, start_col + i))
                assert has_won(bb), \
                    f"Descending diagonal at ({start_row}, {start_col}) not detected"


# ===== Legal Actions Tests =====

class TestGetLegalColumns:
    """Tests for legal column detection."""

    def test_empty_board_all_legal(self):
        """All columns should be legal on empty board."""
        legal = get_legal_columns(bb_combined=0)
        assert legal.shape == (7,)
        assert legal.dtype == np.bool_
        assert legal.all()

    def test_single_full_column(self):
        """Full column should be illegal, others legal."""
        # Fill column 0 (bits 0-5)
        bb = 0b111111  # bits 0-5
        legal = get_legal_columns(bb)
        assert not legal[0], "Column 0 should be illegal when full"
        assert legal[1:].all(), "Other columns should be legal"

    def test_almost_full_column_still_legal(self):
        """Column with 5 pieces should still be legal."""
        # Fill column 0 except top (bits 0-4)
        bb = 0b11111  # bits 0-4
        legal = get_legal_columns(bb)
        assert legal[0], "Column 0 should be legal with one space"

    def test_all_columns_full(self):
        """No columns should be legal when all full."""
        # Fill all columns (each column has bits set at positions 0-5)
        bb = 0
        for col in range(7):
            for row in range(6):
                bb |= (1 << bit_index(row, col))
        legal = get_legal_columns(bb)
        assert not legal.any(), "No columns should be legal on full board"


# ===== Column Height Tests =====

class TestColumnHeight:
    """Tests for column height calculation."""

    def test_empty_column(self):
        """Empty column should have height 0."""
        assert column_height(0, col=0) == 0

    def test_single_piece(self):
        """Column with one piece should have height 1."""
        bb = 1  # bit 0 = (5, 0)
        assert column_height(bb, col=0) == 1

    def test_full_column(self):
        """Full column should have height 6."""
        bb = 0b111111  # bits 0-5
        assert column_height(bb, col=0) == 6

    def test_partial_column(self):
        """Column with 3 pieces should have height 3."""
        bb = 0b111  # bits 0-2
        assert column_height(bb, col=0) == 3


# ===== Drop Piece Tests =====

class TestDropPiece:
    """Tests for piece dropping."""

    def test_drop_in_empty_column(self):
        """Dropping in empty column should place at bottom (bit 0 for col 0)."""
        new_bb = drop_piece(bb_player=0, bb_combined=0, col=0)
        assert new_bb == 1  # bit 0

    def test_drop_stacks_on_existing(self):
        """Dropping in column with piece should stack on top."""
        bb_combined = 1  # bit 0 = (5, 0)
        new_bb = drop_piece(bb_player=0, bb_combined=bb_combined, col=0)
        assert new_bb == 2  # bit 1 = (4, 0)

    def test_drop_in_column_with_multiple(self):
        """Dropping in column with multiple pieces stacks correctly."""
        bb_combined = 0b111  # bits 0-2, column 0 has 3 pieces
        new_bb = drop_piece(bb_player=0, bb_combined=bb_combined, col=0)
        assert new_bb == (1 << 3)  # bit 3 = (2, 0)

    def test_drop_in_different_columns(self):
        """Dropping in different columns places at correct positions."""
        for col in range(7):
            new_bb = drop_piece(bb_player=0, bb_combined=0, col=col)
            expected_bit = col * 7  # bottom of each column
            assert new_bb == (1 << expected_bit), \
                f"Drop in col {col} should set bit {expected_bit}"

    def test_drop_adds_to_existing_player_pieces(self):
        """Dropping should OR with existing player bitboard."""
        bb_player = 1  # already have piece at bit 0
        bb_combined = 1  # same combined state
        new_bb = drop_piece(bb_player=bb_player, bb_combined=bb_combined, col=0)
        # Should have bit 0 (existing) and bit 1 (new)
        assert new_bb == 0b11


# ===== Integration: Win After Drop =====

class TestWinAfterDrop:
    """Integration tests for winning after dropping a piece."""

    def test_vertical_win_after_drop(self):
        """Should detect vertical win after dropping 4th piece."""
        bb = 0b111  # 3 pieces in column 0 (bits 0, 1, 2)
        new_bb = drop_piece(bb_player=bb, bb_combined=bb, col=0)
        assert has_won(new_bb), "Should have vertical win after 4th drop"

    def test_horizontal_win_after_drop(self):
        """Should detect horizontal win after connecting 4 horizontally."""
        # Place 3 pieces at bottom of columns 0, 1, 2
        bb = (1 << 0) | (1 << 7) | (1 << 14)  # bits 0, 7, 14
        bb_combined = bb
        # Drop in column 3
        new_bb = drop_piece(bb_player=bb, bb_combined=bb_combined, col=3)
        assert has_won(new_bb), "Should have horizontal win after connecting 4"
