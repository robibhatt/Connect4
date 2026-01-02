from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.games.core.game import Game, State

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