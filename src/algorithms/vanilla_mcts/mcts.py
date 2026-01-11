"""
Vanilla MCTS algorithm with UCB1 selection and random rollouts.

This module provides the API specification for vanilla MCTS implementation.
USER MUST IMPLEMENT all methods to create a working MCTS algorithm.

Public API:
    - VanillaMCTS: Core MCTS algorithm class
    - MCTSConfig: Configuration dataclass for MCTS hyperparameters
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import math

from src.games.core.game import Game, State


class Node:

    def __init__(
        self,
        state: State,
        num_actions: int
    ):
        
        self.state = state
        self.children = [None for action in range(num_actions)]
        self.is_expanded = False
        self.checked_actions = [False for action in range(num_actions)]

        # stuff for UCT
        self.N = 0
        self.W = 0
        self.Q = 0


@dataclass
class MCTSConfig:

    num_sims: int = 1000
    c_exploration: float = 1.414       # UCB1 constant (sqrt(2))
    max_rollout_depth: int | None = None
    rollout_seed: int | None = None
    illegal_action_penalty: float = 1e9


class VanillaMCTS:

    def __init__(
        self,
        game: Game,
        cfg: Optional[MCTSConfig] = None,
        rng: Optional[np.random.Generator] = None
    ):

        # store the game
        self.game = game

        # store the config
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = MCTSConfig()

        # set the rng
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        # a dictionary of a small set of nodes that we keep referencing
        # if we don't find a node in here, we basically start from scratch.
        self.nodes = {}


    # -------------------------
    # Public API
    # -------------------------

    def run(self, root: State) -> np.ndarray:

        # first, we check if we have a starting point
        hash = self.game.key(s=root)
        if hash in self.nodes:
            root_node = self.nodes[hash]
            self.clear()
            self.nodes[hash] = root_node

        else:

            # well all our savings are a moo point
            self.clear()
            root_node = Node(state=root, num_actions=self.game.action_size)


        for sim in range(self.cfg.num_sims):

            # do all the work of getting the statistacs
            self._rollout(node=root_node)

        # get them visit frequencies
        counts = [child.N if child else 0. for child in root_node.children]
        return np.array(counts)
        
    
    def _select_child(self, node: Node) -> Node:
        """
        uses UCT to select the best child assuming the node has already
        been expanded. 
        """

        assert(node.is_expanded)

        # returns child with best score
        best_score = None
        best_child = None
        for child in node.children:
            if child is not None:
                child_score = - 1.0 * child.Q + self.cfg.c_exploration * math.sqrt(math.log(node.N) / child.N)
                if best_score is None or child_score > best_score:
                    best_score = child_score
                    best_child = child
        return best_child

    def _random_play(self, state: State) -> float:

        current_state = state
        sign = 1.0
        counter = 0
        while self.cfg.max_rollout_depth is None or counter < self.cfg.max_rollout_depth:
            counter += 1
            done, v = self.game.terminal_value(s=current_state)
            if done:
                return sign * v
            else:

                # flip the sign
                sign *= -1.0

                # get a random legal action
                legal_actions = self.game.legal_actions(s=current_state)
                legal_indices = np.flatnonzero(legal_actions)
                action = self.rng.choice(legal_indices)

                # get the next state
                current_state = self.game.next_state(s=current_state, a=action)

        return 0.0

            
    def _rollout(self, node: Node) -> None:

        visited_nodes = []
        current_node = node
        final_reward = None

        while final_reward is None:

            current_node.N += 1

            # check if it is terminal
            (done, value) = self.game.terminal_value(current_node.state)
            if done:
                current_node.W += value
                current_node.Q = current_node.W / current_node.N
                final_reward = -1.0 * value
                break

            visited_nodes.append(current_node)

            if not current_node.is_expanded:

                # pick a brand new action
                new_action = None
                legal_actions = self.game.legal_actions(s=current_node.state)
                for action in range(self.game.action_size):
                    if not current_node.checked_actions[action]:
                        if not legal_actions[action]:
                            current_node.checked_actions[action] = True
                        else:
                            new_action = action

                # check if it is illegal
                assert(legal_actions[new_action])

                # get the next state
                next_state = self.game.next_state(s=current_node.state, a=new_action)

                # create a new node
                new_node = Node(state=next_state, num_actions=self.game.action_size)

                # set it as a child of the current_node
                current_node.children[new_action] = new_node

                # update current_nodes status
                current_node.checked_actions[new_action] = True
                if sum(current_node.checked_actions) == len(current_node.checked_actions):
                    current_node.is_expanded = True

                # random play
                reward = self._random_play(state=next_state)
                new_node.N += 1
                new_node.W += reward
                new_node.Q = new_node.W / new_node.N
                final_reward = -1.0 * reward

            else:

                # get the best child
                current_node = self._select_child(node=current_node)

        # now update all the nodes
        for path_node in reversed(visited_nodes):
            path_node.W += final_reward
            path_node.Q = path_node.W / path_node.N
            final_reward *= -1

        # update dictionary
        for path_node in visited_nodes[:3]:
            hash = self.game.key(path_node.state)
            if hash not in self.nodes:
                self.nodes[hash] = path_node


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

        self.nodes = {}

    
