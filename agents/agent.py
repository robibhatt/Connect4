from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from games.game import Game, State
from mcts.alphazero_mcts import MCTS

import numpy as np


class Agent(ABC):
    """
    Base agent interface.
    Maps a game state to an action index.
    """

    def __init__(self, game: Game):
        self.game = game

    @abstractmethod
    def act(self, s: State) -> int:
        """
        Choose an action at state s.
        Must return a valid action index.
        """
        raise NotImplementedError
    
    def start(self):
        """
        Used at the beginning of a game (i.e. make sure no game trees for 
        certain agents.)
        """
    

class RandomAgent(Agent):
    """
    Picks a uniformly-random legal action (legality defined by self.game).
    """

    def __init__(self, game: Game, rng: Optional[np.random.Generator] = None):
        super().__init__(game)
        self.rng = rng or np.random.default_rng()

    def act(self, s: State) -> int:
        legal = self.game.legal_actions(s)
        if legal.dtype != np.bool_:
            legal = legal.astype(bool)

        legal_idxs = np.flatnonzero(legal)
        if legal_idxs.size == 0:
            raise RuntimeError("No legal actions available at this state.")

        return int(self.rng.choice(legal_idxs))
    

class MCTSAgent(Agent):
    """
    Agent that selects actions using a pre-configured MCTS instance.
    Assumes the MCTS (and its model) are already trained / ready.
    """

    def __init__(self, game: Game, mcts: MCTS):
        super().__init__(game)
        self.mcts = mcts

    def start(self) -> None:
        """
        Called at the beginning of a new game.
        Clears the MCTS search tree.
        """
        self.mcts.clear()

    def act(self, s: State) -> int:
        """
        Choose an action by running MCTS from state s.
        """
        return self.mcts.play_move(s)