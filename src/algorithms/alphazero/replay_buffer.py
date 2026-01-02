from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class AZSample:
    x: np.ndarray   # float32, shape e.g. (C,H,W)
    pi: np.ndarray  # float32, shape (A,)
    z: float        # -1.0, 0.0, +1.0


class ReplayBuffer:
    """
    AlphaZero replay buffer storing (x, pi, z).

    - x: encoded state (player-to-move perspective)
    - pi: MCTS visit distribution at that state
    - z: final outcome from player-to-move perspective at that state
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._data: Deque[AZSample] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, x: np.ndarray, pi: np.ndarray, z: float) -> None:
        self._data.append(
            AZSample(
                x=np.asarray(x, dtype=np.float32),
                pi=np.asarray(pi, dtype=np.float32),
                z=float(z),
            )
        )

    def add_game(
        self,
        game_samples: list[Tuple[np.ndarray, np.ndarray, float]],
        augment_fn=None,
    ) -> None:
        """
        Add a whole game's samples.
        game_samples: list of (x, pi, z)
        augment_fn: optional callable (x, pi) -> list[(x2, pi2)] for symmetries
        """
        if augment_fn is None:
            for x, pi, z in game_samples:
                self.add(x, pi, z)
        else:
            for x, pi, z in game_samples:
                for x2, pi2 in augment_fn(x, pi):
                    self.add(x2, pi2, z)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          X: float32 (B, ...)  stacked
          PI: float32 (B, A)
          Z: float32 (B,)
        """
        rng = rng or np.random.default_rng()
        B = min(int(batch_size), len(self._data))
        idx = rng.choice(len(self._data), size=B, replace=False)

        xs = [self._data[i].x for i in idx]
        pis = [self._data[i].pi for i in idx]
        zs = [self._data[i].z for i in idx]

        X = np.stack(xs, axis=0).astype(np.float32, copy=False)
        PI = np.stack(pis, axis=0).astype(np.float32, copy=False)
        Z = np.asarray(zs, dtype=np.float32)
        return X, PI, Z
