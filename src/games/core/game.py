import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    pass


class Game:
    """
    Abstract base class for turn-based, zero-sum, discrete-action games.
    """

    action_size: int  # total number of discrete actions

    # --- state lifecycle ---

    def reset(self) -> State:
        """Return the initial game state."""
        raise NotImplementedError

    def next_state(self, s: State, a: int) -> State:
        """Apply action a to state s and return the new state."""
        raise NotImplementedError

    # --- rules & queries ---

    def to_play(self, s: State) -> int:
        """
        Return +1 or -1 indicating which player is to move.
        """
        raise NotImplementedError

    def legal_actions(self, s: State) -> np.ndarray:
        """
        Return bool array of shape (action_size,)
        indicating which actions are legal.
        """
        raise NotImplementedError

    def terminal_value(self, s: State) -> tuple[bool, float]:
        """
        Return (done, v).

        done: True if state is terminal.
        v: outcome from viewpoint of the player-to-move at s:
           +1 win, 0 draw, -1 loss.
        """
        raise NotImplementedError

    # --- learning / search hooks ---

    def encode(self, s: State) -> np.ndarray:
        """
        Encode state into a neural-network input tensor,
        canonicalized from the viewpoint of the player-to-move.
        """
        raise NotImplementedError

    def key(self, s: State):
        """
        Return a hashable, deterministic key for the state.
        Used for transpositions in MCTS.
        """
        raise NotImplementedError

    def symmetries(self, x: np.ndarray, pi: np.ndarray):
        """
        Optional data augmentation.
        Return list of (x', pi') pairs.
        """
        return [(x, pi)]