"""
Bitboard utilities for Connect4.

Bitboard layout uses 49 bits (7 columns x 7 rows, where row 6 is sentinel):

  Column layout (each column is 7 bits, bit 6 is sentinel):
    5 12 19 26 33 40 47    <- row 0 (top)
    4 11 18 25 32 39 46    <- row 1
    3 10 17 24 31 38 45    <- row 2
    2  9 16 23 30 37 44    <- row 3
    1  8 15 22 29 36 43    <- row 4
    0  7 14 21 28 35 42    <- row 5 (bottom)
    ----------------------
    sentinel bits: 6, 13, 20, 27, 34, 41, 48 (always 0)

Bit index formula: bit = col * 7 + (5 - row)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bit_index(row: int, col: int) -> int:
    """
    Convert board coordinates to bit index.

    Args:
        row: Row index (0 = top, 5 = bottom)
        col: Column index (0 = left, 6 = right)

    Returns:
        Bit index in the bitboard (0-47, excluding sentinel bits)
    """
    return col * 7 + (5 - row)


def numpy_to_bitboards(board: NDArray[np.int8]) -> tuple[int, int]:
    """
    Convert numpy board to bitboards.

    Args:
        board: 6x7 numpy array with +1 (current player), -1 (opponent), 0 (empty)

    Returns:
        Tuple of (bb_current, bb_opponent) bitboards
    """
    bb_current = 0
    bb_opponent = 0

    for row in range(6):
        for col in range(7):
            bit = bit_index(row, col)
            cell = board[row, col]
            if cell == 1:
                bb_current |= (1 << bit)
            elif cell == -1:
                bb_opponent |= (1 << bit)

    return bb_current, bb_opponent


def bitboards_to_numpy(bb_current: int, bb_opponent: int) -> NDArray[np.int8]:
    """
    Convert bitboards to numpy board.

    Args:
        bb_current: Bitboard for current player (+1 pieces)
        bb_opponent: Bitboard for opponent (-1 pieces)

    Returns:
        6x7 numpy array with +1, -1, 0 values
    """
    board = np.zeros((6, 7), dtype=np.int8)

    for row in range(6):
        for col in range(7):
            bit = bit_index(row, col)
            if bb_current & (1 << bit):
                board[row, col] = 1
            elif bb_opponent & (1 << bit):
                board[row, col] = -1

    return board


def has_won(bb: int) -> bool:
    """
    Check if the given bitboard has a winning 4-in-a-row.

    Uses O(1) bitwise operations instead of O(n) loops.

    Args:
        bb: Bitboard for one player

    Returns:
        True if this player has 4-in-a-row
    """
    # Horizontal (shift by 7 = one column)
    m = bb & (bb >> 7)
    if m & (m >> 14):
        return True

    # Vertical (shift by 1 = one row within column)
    m = bb & (bb >> 1)
    if m & (m >> 2):
        return True

    # Diagonal \ (descending: shift by 6 = down-left, i.e., row+1, col-1)
    # Actually for our layout: moving down-right means row+1, col+1
    # row+1 means -1 in bit position within column
    # col+1 means +7 in bit position
    # So down-right diagonal is shift by 7-1 = 6
    m = bb & (bb >> 6)
    if m & (m >> 12):
        return True

    # Diagonal / (ascending: shift by 8 = down-right, i.e., row+1, col+1)
    # Actually: up-right means row-1, col+1
    # row-1 means +1 in bit position, col+1 means +7
    # So up-right diagonal is shift by 7+1 = 8
    m = bb & (bb >> 8)
    if m & (m >> 16):
        return True

    return False


def get_legal_columns(bb_combined: int) -> NDArray[np.bool_]:
    """
    Get array of legal columns (those not full).

    Args:
        bb_combined: OR of both players' bitboards

    Returns:
        Boolean array of shape (7,) indicating legal columns
    """
    legal = np.zeros(7, dtype=np.bool_)

    for col in range(7):
        # Top row of each column is at bit position: col * 7 + 5
        top_bit = col * 7 + 5
        if not (bb_combined & (1 << top_bit)):
            legal[col] = True

    return legal


def column_height(bb_combined: int, col: int) -> int:
    """
    Get the number of pieces in a column.

    Args:
        bb_combined: OR of both players' bitboards
        col: Column index (0-6)

    Returns:
        Number of pieces in the column (0-6)
    """
    # Extract the 6 bits for this column (excluding sentinel)
    col_bits = (bb_combined >> (col * 7)) & 0b111111
    return bin(col_bits).count('1')


def drop_piece(bb_player: int, bb_combined: int, col: int) -> int:
    """
    Drop a piece in the given column and return updated player bitboard.

    Args:
        bb_player: Current player's bitboard
        bb_combined: OR of both players' bitboards
        col: Column to drop in (0-6)

    Returns:
        Updated player bitboard with new piece added

    Note:
        Does not validate if column is full - caller must check.
    """
    # Find the lowest empty row in this column
    # Column starts at bit col*7, rows go from bit 0 (row 5) to bit 5 (row 0)
    height = column_height(bb_combined, col)
    new_bit = col * 7 + height  # Next available position
    return bb_player | (1 << new_bit)
