from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from jaxtyping import Bool, Float, Int

from src.games.core.game import Game, State


@dataclass(frozen=True)
class Connect4State(State):
    """
    Relative / canonical board:
      +1 = current player's discs
      -1 = opponent's discs
       0 = empty
    Always from the perspective of the player-to-move.
    """

    board: Int[np.ndarray, "6 7"]  # dtype int8 recommended


class Connect4(Game):
    action_size: int = 7

    def reset(self) -> Connect4State:
        board: Int[np.ndarray, "6 7"] = np.zeros((6, 7), dtype=np.int8)
        return Connect4State(board=board)

    def to_play(self, s: Connect4State) -> int:
        # In this representation, the player-to-move is always the +1 player.
        return +1

    def legal_actions(self, s: Connect4State) -> Bool[np.ndarray, "7"]:
        # A move is legal if the column is not full (top cell is empty).
        legal = (s.board[0, :] == 0)
        return legal.astype(np.bool_, copy=False)

    def next_state(self, s: Connect4State, a: int) -> Connect4State:
        if a < 0 or a >= self.action_size:
            raise ValueError(f"Action {a} out of range [0, {self.action_size}).")

        column = s.board[:, a]
        empty_rows = np.nonzero(column == 0)[0]
        if len(empty_rows) == 0:
            raise ValueError(f"Illegal move: column {a} is full.")

        r = int(empty_rows[-1])  # lowest available row
        board2 = np.array(s.board, copy=True)
        board2[r, a] = np.int8(+1)  # current player plays as +1
        board2 *= np.int8(-1)       # flip perspective for the next player
        return Connect4State(board=board2)

    def terminal_value(self, s: Connect4State) -> tuple[bool, float]:
        """
        Returns (done, v) where v is from viewpoint of player-to-move.
        Since state is canonical, "player-to-move" == +1 markers.
        """
        b = s.board

        if self._has_four(b, +1):
            return True, +1.0
        if self._has_four(b, -1):
            return True, -1.0

        if not bool((b == 0).any()):
            return True, 0.0

        return False, 0.0

    def encode(self, s: Connect4State) -> Float[np.ndarray, "6 7"]:
        """
        NN input: already canonical; return (6,7) float32.
        """
        x = s.board.astype(np.float32, copy=False)
        return x

    def key(self, s: Connect4State) -> bytes:
        # Player-to-move is implicit; board alone is enough.
        return s.board.tobytes()

    def symmetries(
        self,
        x: Float[np.ndarray, "6 7"],
        pi: Float[np.ndarray, "7"],
    ) -> List[Tuple[Float[np.ndarray, "6 7"], Float[np.ndarray, "7"]]]:
        """
        Left-right reflection symmetry.
        x:  (6,7)
        pi: (7,) corresponding to columns 0..6.
        """
        xf = np.fliplr(x).astype(np.float32, copy=False)
        pf = np.flip(pi).astype(np.float32, copy=False)
        return [
            (x.astype(np.float32, copy=False), pi.astype(np.float32, copy=False)),
            (xf, pf),
        ]

    def _has_four(self, b: Int[np.ndarray, "6 7"], player: int) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            for r in range(6):
                for c in range(7):
                    if int(b[r, c]) != player:
                        continue
                    if self._check_direction(b, r, c, dr, dc, player):
                        return True
        return False

    def _check_direction(
        self,
        b: Int[np.ndarray, "6 7"],
        r: int,
        c: int,
        dr: int,
        dc: int,
        player: int,
    ) -> bool:
        for k in range(1, 4):
            rr = r + dr * k
            cc = c + dc * k
            if rr < 0 or rr >= 6 or cc < 0 or cc >= 7:
                return False
            if int(b[rr, cc]) != player:
                return False
        return True
