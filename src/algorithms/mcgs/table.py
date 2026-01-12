import numpy as np


class Table:

    def __init__(
        self,
        num_states: int
    ):
        
        # helpful for deleting old nodes
        self.generations = np.zeros(num_states, dtype=np.int64)
        self.current_generation = 1

        # necessary for checking which state is stored here
        self.hashes = np.zeros(num_states, dtype=np.int64)

        # Ws used for UCT1
        self.W = np.zeros(num_states, dtype=np.float32)

        # Ns used for UCT1
        self.N = np.zeros(num_states, dtype=np.int64)


    def add_state(
        self,
        state_hash: np.int64,
        generation: int,
        w: float,
        n: int
    ):

        pass
        

