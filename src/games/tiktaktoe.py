from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from jaxtyping import Bool, Float, Int

from src.games.game import Game, State


@dataclass(frozen=True)
class TicTacToeState(State):
    """
    Relative / canonical board:
      +1 = current player's marks
      -1 = opponent's marks
       0 = empty
    Always from the perspective of the player-to-move.
    """
    board: Int[np.ndarray, "3 3"]  # dtype int8 recommended


class TicTacToe(Game):
    action_size: int = 9

    def reset(self) -> TicTacToeState:
        board: Int[np.ndarray, "3 3"] = np.zeros((3, 3), dtype=np.int8)
        return TicTacToeState(board=board)

    def to_play(self, s: TicTacToeState) -> int:
        # In this representation, the player-to-move is always the +1 player.
        return +1

    def legal_actions(self, s: TicTacToeState) -> Bool[np.ndarray, "9"]:
        # legal_grid: Bool[np.ndarray, "3 3"]
        legal_grid = (s.board == 0)
        # legal: Bool[np.ndarray, "9"]  (flatten row-major: 0..8 left->right, top->bottom)
        legal = legal_grid.reshape(9)
        return legal.astype(np.bool_, copy=False)

    def next_state(self, s: TicTacToeState, a: int) -> TicTacToeState:
        if a < 0 or a >= self.action_size:
            raise ValueError(f"Action {a} out of range [0, {self.action_size}).")

        r, c = divmod(a, 3)
        if int(s.board[r, c]) != 0:
            raise ValueError(f"Illegal move: square {a} (r={r}, c={c}) is not empty.")

        # board2: Int[np.ndarray, "3 3"]
        board2 = np.array(s.board, copy=True)
        board2[r, c] = np.int8(+1)    # current player plays as +1
        board2 *= np.int8(-1)         # flip perspective for the next player
        return TicTacToeState(board=board2)

    def terminal_value(self, s: TicTacToeState) -> tuple[bool, float]:
        """
        Returns (done, v) where v is from viewpoint of player-to-move.
        Since state is canonical, "player-to-move" == +1 markers.
        """
        # b: Int[np.ndarray, "3 3"]
        b = s.board

        # row/col checks
        row_sums = b.sum(axis=1)  # Int[np.ndarray, "3"]
        col_sums = b.sum(axis=0)  # Int[np.ndarray, "3"]

        if (row_sums == 3).any() or (col_sums == 3).any():
            return True, +1.0
        if (row_sums == -3).any() or (col_sums == -3).any():
            return True, -1.0

        # diagonal checks
        diag1 = int(b[0, 0] + b[1, 1] + b[2, 2])
        diag2 = int(b[0, 2] + b[1, 1] + b[2, 0])
        if diag1 == 3 or diag2 == 3:
            return True, +1.0
        if diag1 == -3 or diag2 == -3:
            return True, -1.0

        # draw if no empty squares
        if not bool((b == 0).any()):
            return True, 0.0

        return False, 0.0

    def encode(self, s: TicTacToeState) -> Float[np.ndarray, "3 3"]:
        """
        NN input: already canonical; return (3,3) float32.
        """
        # x: Float[np.ndarray, "3 3"]
        x = s.board.astype(np.float32, copy=False)
        return x

    def key(self, s: TicTacToeState) -> bytes:
        # Player-to-move is implicit; board alone is enough.
        return s.board.tobytes()

    def symmetries(
        self,
        x: Float[np.ndarray, "3 3"],
        pi: Float[np.ndarray, "9"],
    ) -> List[Tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "9"]]]:
        """
        8 symmetries: 4 rotations x {no flip, horizontal flip}.
        x:  (3,3)
        pi: (9,) flattened row-major corresponding to squares 0..8.
        """
        # x3: Float[np.ndarray, "3 3"]
        x3 = x.reshape(3, 3)
        # p3: Float[np.ndarray, "3 3"]
        p3 = pi.reshape(3, 3)

        out: List[Tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "9"]]] = []
        for k in range(4):
            # xr: Float[np.ndarray, "3 3"]
            xr = np.rot90(x3, k).astype(np.float32, copy=False)
            # pr: Float[np.ndarray, "3 3"]
            pr = np.rot90(p3, k).astype(np.float32, copy=False)
            out.append((xr, pr.reshape(9).astype(np.float32, copy=False)))

            # xf: Float[np.ndarray, "3 3"]
            xf = np.fliplr(xr).astype(np.float32, copy=False)
            # pf: Float[np.ndarray, "3 3"]
            pf = np.fliplr(pr).astype(np.float32, copy=False)
            out.append((xf, pf.reshape(9).astype(np.float32, copy=False)))

        return out
