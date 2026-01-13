"""
MCGS (Monte Carlo Graph Search) algorithm with UCB1 selection and random rollouts.

This module provides the API specification for MCGS implementation.
USER MUST IMPLEMENT all methods to create a working MCGS algorithm.

Public API:
    - MCGS: Core MCGS algorithm class
    - MCGSCoreConfig: Configuration dataclass for MCGS hyperparameters
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import math

from src.games.core.game import Game, State
from src.algorithms.mcgs.player import Player
from src.algorithms.mcgs.table import Table, Node
from src.algorithms.mcgs.evaluator import Evaluator



@dataclass
class MCGSCoreConfig:

    num_sims: int = 1000
    c_exploration: float = 1.414       # UCB1 constant (sqrt(2))
    max_rollout_depth: int | None = None
    rollout_seed: int | None = None
    illegal_action_penalty: float = 1e9
    max_nodes: int | None = 100_000       # Maximum number of nodes we store in our graph
    batch_size: int = 1                # Batch size for node evaluations (1 = no batching)


class MCGS:

    def __init__(
        self,
        game: Game,
        cfg: Optional[MCGSCoreConfig] = None,
        rng: Optional[np.random.Generator] = None
    ):

        # store the game
        self.game = game

        # store the config
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = MCGSCoreConfig()

        # set the rng
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        # create the table, player, evaluator
        self.table = Table(num_states=self.cfg.max_nodes)
        self.evaluator = Evaluator(
                            game=self.game, 
                            max_depth=self.cfg.max_rollout_depth, 
                            seed=self.rng.integers(1000)
                        )
        self.player = Player(
                            table=self.table,
                            game=self.game,
                            c=self.cfg.c_exploration
                        )


    # -------------------------
    # Public API
    # -------------------------

    def run(self, root: State) -> np.ndarray:

        # reset table for fresh search
        self.clear()

        # get children states
        legalities = self.game.legal_actions(s=root)
        legal_actions = np.where(legalities)[0]
        children_states = [self.game.next_state(root, a) for a in legal_actions]

        # do rollouts until we have enough
        total_rollouts = 0
        while total_rollouts < self.cfg.num_sims:
            total_rollouts += self._add_rollouts(state=root)

        # get N values from table for each child
        counts = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child_state in zip(legal_actions, children_states):
            key = self.game.key(child_state)
            node = self.table.get_node(state_hash=key)
            counts[action] = node.n

        return counts
    

    def _add_rollouts(
        self,
        state: State
    ) -> int:
        # generate rollouts from player
        rollouts, valued_rollouts = self.player.generate_rollouts(
            target_batch_size=self.cfg.batch_size,
            state=state
        )

        # evaluate unvalued rollouts using last state
        if rollouts:
            last_states = [rollout[-1] for rollout in rollouts]
            values = self.evaluator.evaluate(last_states)
            for rollout, value in zip(rollouts, values):
                valued_rollouts.append((rollout, value))

        # back propagate all valued rollouts
        self._back_prop(valued_rollouts)

        return len(valued_rollouts)

    def _back_prop(
        self,
        valued_rollouts: list[tuple[list[State], float]]
    ):

        ns = {}
        ws = {}

        for (rollout, value) in valued_rollouts:
            sign = 1.0
            for state in reversed(rollout):
                key = self.game.key(state)
                if key not in ns:
                    node = self.table.get_node(state_hash=key)
                    ns[key] = node.n
                    ws[key] = node.w
                ns[key] += 1
                ws[key] += value * sign
                sign *= -1.

        for key in ns:
            node = Node(
                n = ns[key],
                w = ws[key],
                hash = key
            )
            self.table.add_state(node)


    def select_action(self, pi: np.ndarray, deterministic: bool = False) -> int:

        if deterministic:
            return np.argmax(pi)
        else:
            probs = pi / np.sum(pi)
            return self.rng.choice(len(pi), p=probs)

    def play_move(self, s: State, deterministic: bool = True) -> int:

        pi = self.run(root=s)
        legal_actions = self.game.legal_actions(s=s)
        pi = np.where(legal_actions, pi, 0)
        return self.select_action(pi, deterministic=deterministic)

    def clear(self) -> None:
        self.table.clear()


