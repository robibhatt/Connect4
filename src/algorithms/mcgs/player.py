from dataclasses import dataclass
from src.algorithms.mcgs.table import Table, Node
from src.games.core.game import Game, State
from collections import Counter
import numpy as np
import math




class Player:

    def __init__(
        self,
        table: Table,
        game: Game,
        c: float
    ):
        
        self.table = table
        self.game = game
        self.c = c # exploration constant

    def _children(
        self,
        state: State
    )-> list[State]:
        
        legalities = self.game.legal_actions(s=state)
        legal_actions = np.where(legalities)[0]
        children =  [self.game.next_state(state, a) for a in legal_actions]
        return children
        
    def _next_node(
        self,
        state: State,
        n: int,
        virtual_ns: Counter
    )-> tuple[Node, State]:
        
        """
        Use UCB1 to select the next node
        """

        children_states = self._children(state)
        best_node = None
        best_score = None
        best_state = None
        for child_state in children_states:

            key = self.game.key(child_state)
            node = self.table.get_node(state_hash=key)
            
            virtual_n = max(virtual_ns[key], node.n)

            # we grab a node if it has never been seen
            if virtual_n == 0:
                return (node, child_state)
            
            else:
                score = -1.0 * node.w / virtual_n + self.c * math.sqrt(math.log(n) / virtual_n)
                if best_score is None or score > best_score:
                    best_score = score
                    best_node = node
                    best_state = child_state

        return (best_node, best_state)
                    

    def generate_rollouts(
        self,
        target_batch_size: int,
        state: State
    ) -> tuple[list[list[State]], list[tuple[list[State], float]]]:
        
        """
        Here is the key mcgs logic.
        Returns a list of rollouts and a list of valued rollouts.
        """

        rollouts = []
        valued_rollouts = []
        virtual_ns = Counter()

        while len(rollouts) + len(valued_rollouts) < target_batch_size:

            # we generate another set of rollouts
            current_state = state
            found_leaf = False
            rollout = []
            current_node = None
            while not found_leaf:

                # check the table for our state
                if current_node is None:
                    key = self.game.key(current_state)
                    current_node = self.table.get_node(state_hash=key)

                # put our state into the rollout
                rollout.append(current_state)

                # start by checking terminal
                (done , v) = self.game.terminal_value(current_state)
                if done:

                    # value is always from pov of last guy in rollout
                    valued_rollouts.append((rollout, v))
                    found_leaf = True

                else:

                    # otherwise we check if our node is a leaf!

                    virtual_ns[current_node.hash] = max(virtual_ns[current_node.hash],  current_node.n)
                    if virtual_ns[current_node.hash] == 0:
                        # we found a leaf
                        rollouts.append(rollout)
                        found_leaf = True

                    else:
                        # no leaf
                        current_node, current_state = self._next_node(
                                                                state=current_state, 
                                                                n=virtual_ns[current_node.hash],
                                                                virtual_ns=virtual_ns)

            # increment the virtual counter
            for state in rollout:
                virtual_ns[self.game.key(state)] += 1

        return (rollouts, valued_rollouts)