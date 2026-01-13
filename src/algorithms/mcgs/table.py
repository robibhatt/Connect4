import numpy as np
from typing import NamedTuple


class Node(NamedTuple):

    n: int
    w: float
    hash: int


class Table:

    def __init__(
        self,
        num_states: int
    ):
        
        # store num states
        self.num_states = num_states

        # bools for whether we use a slot
        self.in_use = np.zeros(num_states, dtype=np.bool_)

        # necessary for checking which state is stored here
        self.hashes = np.zeros(num_states, dtype=np.int64)

        # Ws used for UCT1
        self.W = np.zeros(num_states, dtype=np.float32)

        # Ns used for UCT1
        self.N = np.zeros(num_states, dtype=np.int64)

    def add_state(
        self,
        node: Node
    ):
        """
        We always keep the more visited state
        """

        loc = node.hash % self.num_states
        if (not self.in_use[loc]) or (self.N[loc] <= node.n):
            # we slot our information in if the slot was empty or if we have a more visited guy
            self.hashes[loc] = node.hash
            self.W[loc] = node.w
            self.N[loc] = node.n
            self.in_use[loc] = True

    def get_node(
        self,
        state_hash: int
    ) -> Node:
        """
        Just grab the N and W
        """

        loc = state_hash % self.num_states
        if self.in_use[loc] and self.hashes[loc] == state_hash:
            # we only grab if the hash actually matches
            return Node(n=int(self.N[loc]),
                        w=float(self.W[loc]),
                        hash=state_hash)

        else:
            return Node(n=0,
                        w=0.,
                        hash=state_hash)

    def clear(self):
        """Reset all table state."""
        self.in_use.fill(False)
        self.hashes.fill(0)
        self.W.fill(0)
        self.N.fill(0)