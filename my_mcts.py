from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from games.game import Game, State
from evaluator import Evaluator


@dataclass
class MCTSConfig:
    num_sims: int = 200
    c_puct: float = 1.5

    # Root exploration noise (AlphaZero)
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # Numerical safety
    illegal_action_penalty: float = 1e9


class MCTS:
    """
    AlphaZero-style MCTS:
      - selection via PUCT
      - leaf evaluation via Evaluator (policy priors + value)
      - backup with sign flip each ply

    IMPORTANT CONVENTION:
      game.terminal_value(s) returns value from viewpoint of player-to-move at s.
      evaluator.evaluate(game, s) returns value from viewpoint of player-to-move at s.
      Therefore backup uses v = -v each step.
    """

    def __init__(
        self,
        game: Game,
        evaluator: Evaluator,
        cfg: Optional[MCTSConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        raise NotImplementedError

    # -------------------------
    # Public API
    # -------------------------

    def run(
        self,
        root: State,
        add_dirichlet_noise: bool = True,
        tau: float = 1.0,
    ) -> np.ndarray:
        """
        Run MCTS simulations from root and return improved policy pi over actions.

        pi is derived from root visit counts:
          pi[a] ∝ N(root,a)^(1/tau), masked to legal actions.

        tau:
          - tau=1.0: proportional to visits
          - tau->0: argmax
        """
        raise NotImplementedError

    def select_action(self, pi: np.ndarray, deterministic: bool = False) -> int:
        """
        Convert a policy distribution pi into an action choice.
        - deterministic: choose argmax
        - otherwise: sample from pi
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Drop the transposition table (fresh search memory)."""
        raise NotImplementedError

    # Optional: reuse subtree memory across real moves
    def advance_root(self, new_root: State) -> None:
        """
        Keeps table as-is. You can call this between moves if you want to reuse stats.
        (No-op in this transposition-table design; included for semantic clarity.)
        """
        raise NotImplementedError

    # -------------------------
    # Test time Mover
    # -------------------------

    @torch.inference_mode()
    def play_move(
        self,
        s: State,
        num_sims: Optional[int] = None,
        deterministic: bool = True,
    ) -> int:
        """
        Choose an action for actual play:
        - run MCTS from state s (no Dirichlet noise)
        - pick action from the improved policy (usually deterministically)

        num_sims: override cfg.num_sims for this call (optional).
        deterministic: if True, choose argmax; else sample from pi.
        """
        raise NotImplementedError
