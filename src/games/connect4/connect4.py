"""Connect4 game implementation with bitboard optimization."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from jaxtyping import Bool, Float, Int

from src.games.core.game import Game, State
from src.games.connect4.bitboard import (
    numpy_to_bitboards,
    bitboards_to_numpy,
    has_won,
    get_legal_columns,
    drop_piece,
    column_height,
)


class Connect4State(State):
    """
    Connect4 game state backed by bitboards for O(1) win detection.

    Relative / canonical board representation:
      +1 = current player's discs
      -1 = opponent's discs
       0 = empty
    Always from the perspective of the player-to-move.

    Construction:
        - Connect4State(board=np_array) - from numpy array (backward compatible)
        - Connect4State._from_bitboards(bb_c, bb_o) - from bitboards (internal use)
    """

    __slots__ = ('_bb_current', '_bb_opponent', '_board_cache')

    def __init__(self, *, board: Int[np.ndarray, "6 7"]):
        """
        Create state from numpy board array.

        Args:
            board: 6x7 numpy array with +1 (current), -1 (opponent), 0 (empty)
        """
        object.__setattr__(self, '_bb_current', 0)
        object.__setattr__(self, '_bb_opponent', 0)
        object.__setattr__(self, '_board_cache', board)

        bb_c, bb_o = numpy_to_bitboards(board)
        object.__setattr__(self, '_bb_current', bb_c)
        object.__setattr__(self, '_bb_opponent', bb_o)

    @classmethod
    def _from_bitboards(
        cls,
        bb_current: int,
        bb_opponent: int,
    ) -> Connect4State:
        """
        Create state from bitboards (internal use for performance).

        Args:
            bb_current: Bitboard for current player's pieces
            bb_opponent: Bitboard for opponent's pieces

        Returns:
            New Connect4State instance
        """
        instance = object.__new__(cls)
        object.__setattr__(instance, '_bb_current', bb_current)
        object.__setattr__(instance, '_bb_opponent', bb_opponent)
        object.__setattr__(instance, '_board_cache', None)
        return instance

    @property
    def board(self) -> Int[np.ndarray, "6 7"]:
        """
        Get the numpy board representation.

        Lazily computed from bitboards if not cached.
        """
        if self._board_cache is None:
            board = bitboards_to_numpy(self._bb_current, self._bb_opponent)
            object.__setattr__(self, '_board_cache', board)
        return self._board_cache

    def __hash__(self) -> int:
        return hash((self._bb_current, self._bb_opponent))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connect4State):
            return NotImplemented
        return (
            self._bb_current == other._bb_current
            and self._bb_opponent == other._bb_opponent
        )

    def __repr__(self) -> str:
        return f"Connect4State(bb_current={self._bb_current}, bb_opponent={self._bb_opponent})"

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Connect4State is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Connect4State is immutable")


class Connect4(Game):
    """Connect4 game with O(1) win detection using bitboards."""

    action_size: int = 7

    def reset(self) -> Connect4State:
        """Return initial empty board state."""
        return Connect4State._from_bitboards(0, 0)

    def to_play(self, s: Connect4State) -> int:
        """Return +1 (current player is always +1 in canonical form)."""
        return +1

    def legal_actions(self, s: Connect4State) -> Bool[np.ndarray, "7"]:
        """Return boolean array of legal columns (not full)."""
        bb_combined = s._bb_current | s._bb_opponent
        return get_legal_columns(bb_combined)

    def next_state(self, s: Connect4State, a: int) -> Connect4State:
        """
        Apply action and return new state with flipped perspective.

        Args:
            s: Current state
            a: Column to drop piece in (0-6)

        Returns:
            New state after move, with perspective flipped

        Raises:
            ValueError: If action is out of range or column is full
        """
        if a < 0 or a >= self.action_size:
            raise ValueError(f"Action {a} out of range [0, {self.action_size}).")

        bb_combined = s._bb_current | s._bb_opponent

        # Check if column is full
        top_bit = a * 7 + 5
        if bb_combined & (1 << top_bit):
            raise ValueError(f"Illegal move: column {a} is full.")

        # Drop piece for current player
        new_bb_current = drop_piece(s._bb_current, bb_combined, a)

        # Flip perspective: opponent becomes current, current becomes opponent
        return Connect4State._from_bitboards(
            bb_current=s._bb_opponent,
            bb_opponent=new_bb_current,
        )

    def terminal_value(self, s: Connect4State) -> tuple[bool, float]:
        """
        Check if game is over and return value.

        Returns:
            (done, value) where value is from viewpoint of player-to-move:
            +1.0 = current player wins
            -1.0 = opponent wins
            0.0 = draw or ongoing
        """
        # Check if current player has won
        if has_won(s._bb_current):
            return True, +1.0

        # Check if opponent has won
        if has_won(s._bb_opponent):
            return True, -1.0

        # Check for draw (board full)
        bb_combined = s._bb_current | s._bb_opponent
        # Board is full if all top row bits are set
        TOP_ROW_MASK = (
            (1 << 5) | (1 << 12) | (1 << 19) | (1 << 26) |
            (1 << 33) | (1 << 40) | (1 << 47)
        )
        if (bb_combined & TOP_ROW_MASK) == TOP_ROW_MASK:
            return True, 0.0

        return False, 0.0

    def encode(self, s: Connect4State) -> Float[np.ndarray, "6 7"]:
        """
        Encode state for neural network input.

        Returns (6, 7) float32 array with +1, -1, 0 values.
        """
        return s.board.astype(np.float32, copy=False)

    def key(self, s: Connect4State) -> bytes:
        """
        Return hashable key for transposition tables.

        Uses board bytes for compatibility with existing caches.
        """
        return s.board.tobytes()

    def symmetries(
        self,
        x: Float[np.ndarray, "6 7"],
        pi: Float[np.ndarray, "7"],
    ) -> List[Tuple[Float[np.ndarray, "6 7"], Float[np.ndarray, "7"]]]:
        """
        Return symmetry transformations for data augmentation.

        Connect4 has left-right reflection symmetry.
        """
        xf = np.fliplr(x).astype(np.float32, copy=False)
        pf = np.flip(pi).astype(np.float32, copy=False)
        return [
            (x.astype(np.float32, copy=False), pi.astype(np.float32, copy=False)),
            (xf, pf),
        ]
