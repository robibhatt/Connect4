from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from src.games.game import Game, State


@dataclass
class MCTSConfig:
    num_sims: int = 200
    c_puct: float = 1.5

    # Root exploration noise (AlphaZero)
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # Numerical safety
    illegal_action_penalty: float = 1e9


class Node:
    """
    Search-time bookkeeping for a single state key.
    Stores per-action stats: P, N, W, Q.
    """
    __slots__ = ("P", "N", "W", "Q", "expanded")

    def __init__(self, action_size: int):
        self.P = np.zeros(action_size, dtype=np.float32)       # priors
        self.N = np.zeros(action_size, dtype=np.int32)         # visit counts
        self.W = np.zeros(action_size, dtype=np.float32)       # total value
        self.Q = np.zeros(action_size, dtype=np.float32)       # mean value (W/N)
        self.expanded = False


class MCTS:
    """
    AlphaZero-style MCTS:
      - selection via PUCT
      - leaf evaluation via neural network model (policy priors + value)
      - backup with sign flip each ply

    IMPORTANT CONVENTION:
      game.terminal_value(s) returns value from viewpoint of player-to-move at s.
      model(x) returns (logits, value) from viewpoint of player-to-move at s.
      Therefore backup uses v = -v each step.
    """

    def __init__(
        self,
        game: Game,
        model: nn.Module,
        device: Optional[torch.device] = None,
        cfg: Optional[MCTSConfig] = None,
        rng: Optional[np.random.Generator] = None
    ):
        self.game = game
        self.model = model
        self.device = device or torch.device('cpu')
        self.cfg = cfg or MCTSConfig()
        self.rng = rng or np.random.default_rng()

        # Transposition table: state_key -> Node stats
        self._table: Dict[object, Node] = {}

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
        root_node = self._get_node(root)

        # Expand root if needed, injecting Dirichlet noise if requested.
        if not root_node.expanded:
            self._expand(root, root_node, add_dirichlet_noise=add_dirichlet_noise)
        elif add_dirichlet_noise:
            # If already expanded but you still want noise at the root (common in self-play),
            # we can re-apply noise to its P in-place safely.
            self._apply_dirichlet_noise_inplace(root, root_node)

        for _ in range(self.cfg.num_sims):
            self._simulate(root)

        return self._visits_to_policy(root, root_node, tau=tau)

    def select_action(self, pi: np.ndarray, deterministic: bool = False) -> int:
        """
        Convert a policy distribution pi into an action choice.
        - deterministic: choose argmax
        - otherwise: sample from pi
        """
        if deterministic:
            return int(np.argmax(pi))
        return int(self.rng.choice(len(pi), p=pi))

    def clear(self) -> None:
        """Drop the transposition table (fresh search memory)."""
        self._table.clear()

    # Optional: reuse subtree memory across real moves
    def advance_root(self, new_root: State) -> None:
        """
        Keeps table as-is. You can call this between moves if you want to reuse stats.
        (No-op in this transposition-table design; included for semantic clarity.)
        """
        # With a transposition table, "advancing root" is just using a different root state.
        # We don't need to restructure anything.
        _ = new_root

    # -------------------------
    # Core MCTS routines
    # -------------------------

    def _simulate(self, root: State) -> None:
        """
        One MCTS simulation:
          - selection down the tree with PUCT
          - expand/evaluate leaf
          - backup along the path with sign flips
        """
        path: List[Tuple[State, int]] = []
        s = root

        while True:
            done, tv = self.game.terminal_value(s)
            if done:
                v = float(tv)
                break

            node = self._get_node(s)

            if not node.expanded:
                v = self._expand(s, node, add_dirichlet_noise=False)
                break

            a = self._select_action_puct(s, node)
            path.append((s, a))
            s = self.game.next_state(s, a)

        # Backup: update edge stats along path; flip viewpoint each ply.
        for s_parent, a in reversed(path):
            v = -v  # <-- flip BEFORE using it at the parent
            parent_node = self._get_node(s_parent)
            parent_node.N[a] += 1
            parent_node.W[a] += v
            parent_node.Q[a] = parent_node.W[a] / parent_node.N[a]

    def _select_action_puct(self, s: State, node: Node) -> int:
        legal = self.game.legal_actions(s)  # bool[action_size]
        if legal.dtype != np.bool_:
            legal = legal.astype(bool)

        # Total visits from this node across legal actions
        n_sum = int(node.N[legal].sum())
        if n_sum <= 0:
            n_sum = 1

        # PUCT score
        # U(a) = Q + c_puct * P * sqrt(n_sum) / (1 + N)
        u = node.Q + (self.cfg.c_puct * node.P * (np.sqrt(n_sum) / (1.0 + node.N)))

        # mask illegal actions to -inf-ish
        u = u.copy()
        u[~legal] = -self.cfg.illegal_action_penalty

        return int(np.argmax(u))

    def _expand(self, s: State, node: Node, add_dirichlet_noise: bool) -> float:
        """
        Expand a leaf node:
          - query model for priors + value at s
          - mask+normalize priors to legal actions
          - optionally add Dirichlet noise (root only)
          - store priors in node and mark expanded
          - return value v (from viewpoint of player-to-move at s)
        """
        # Encode state -> torch tensor
        x_np = self.game.encode(s).astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)

        # Run model
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits_b, value_b = self.model(x)
        finally:
            self.model.train(was_training)

        # Extract outputs
        logits = logits_b.squeeze(0)
        value = value_b.squeeze(0).squeeze(0)

        # Masked softmax over legal actions
        legal = self.game.legal_actions(s)
        if legal.dtype != np.bool_:
            legal = legal.astype(bool)

        legal_t = torch.from_numpy(legal).to(logits.device)
        masked_logits = logits.masked_fill(~legal_t, -1e9)
        priors_t = torch.softmax(masked_logits, dim=0)

        # Convert to numpy
        priors = priors_t.cpu().numpy().astype(np.float32)
        v = float(value.cpu().item())

        # Apply noise if needed
        p = priors
        if add_dirichlet_noise:
            p = self._with_dirichlet_noise(p, legal)

        node.P[:] = p
        node.expanded = True
        return v

    # -------------------------
    # Policy extraction
    # -------------------------

    def _visits_to_policy(self, s: State, node: Node, tau: float) -> np.ndarray:
        legal = self.game.legal_actions(s)
        if legal.dtype != np.bool_:
            legal = legal.astype(bool)

        visits = node.N.astype(np.float32)
        visits[~legal] = 0.0

        if tau <= 1e-8:
            pi = np.zeros_like(visits, dtype=np.float32)
            pi[int(np.argmax(visits))] = 1.0
            return pi

        # pi[a] ∝ N[a]^(1/tau)
        inv_tau = 1.0 / tau
        weights = np.power(visits, inv_tau, dtype=np.float32)
        weights[~legal] = 0.0
        z = float(weights.sum())
        if z <= 0:
            pi = legal.astype(np.float32)
            pi /= pi.sum()
            return pi
        return (weights / z).astype(np.float32)

    # -------------------------
    # Dirichlet noise helpers
    # -------------------------

    def _with_dirichlet_noise(self, p: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """
        AlphaZero root noise:
          p <- (1-eps)*p + eps*Dir(alpha)
        Noise is applied over the full action space, then masked/renormalized.
        """
        eps = self.cfg.dirichlet_eps
        alpha = self.cfg.dirichlet_alpha

        # Sample noise for all actions; mask illegal later.
        noise = self.rng.dirichlet([alpha] * self.game.action_size).astype(np.float32)
        p2 = (1.0 - eps) * p + eps * noise
        return self._mask_and_normalize(p2, legal)

    def _apply_dirichlet_noise_inplace(self, s: State, node: Node) -> None:
        legal = self.game.legal_actions(s)
        node.P[:] = self._with_dirichlet_noise(node.P, legal)

    # -------------------------
    # Table / node helpers
    # -------------------------

    def _get_node(self, s: State) -> Node:
        k = self.game.key(s)
        n = self._table.get(k)
        if n is None:
            n = Node(self.game.action_size)
            self._table[k] = n
        return n

    # -------------------------
    # Probability helpers
    # -------------------------

    @staticmethod
    def _mask_and_normalize(p: np.ndarray, legal: np.ndarray) -> np.ndarray:
        p = p.astype(np.float32, copy=True)
        p[~legal] = 0.0
        s = float(p.sum())
        if s <= 0.0:
            # If evaluator gave junk (or all illegal), fall back to uniform legal
            p = legal.astype(np.float32)
            p /= float(p.sum())
            return p
        p /= s
        return p
    
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
        old_num_sims = self.cfg.num_sims
        if num_sims is not None:
            self.cfg.num_sims = int(num_sims)

        try:
            pi = self.run(root=s, add_dirichlet_noise=False, tau=1e-8)  # tau≈0 => argmax visits
            return self.select_action(pi, deterministic=deterministic)
        finally:
            self.cfg.num_sims = old_num_sims