from src.games.core.game import Game, State
import numpy as np


class Evaluator:

    """
    We evaluate states with random play
    """

    def __init__(
        self,
        game: Game,
        max_depth: int | None,
        seed: int | None,
    ):
        
        self.game = game
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)
    

    def evaluate(
        self,
        states: list[State]
    )-> list[float]:

        evals = []

        for initial_state in states:
            current_state = initial_state
            sign = 1.0
            counter = 0
            found_value = False
            while self.max_depth is None or counter < self.max_depth:
                counter += 1
                done, v = self.game.terminal_value(s=current_state)
                if done:
                    evals.append(sign * v)
                    found_value = True
                    break
                else:

                    # flip the sign
                    sign *= -1.0

                    # get a random legal action
                    legal_actions = self.game.legal_actions(s=current_state)
                    legal_indices = np.flatnonzero(legal_actions)
                    action = self.rng.choice(legal_indices)

                    # get the next state
                    current_state = self.game.next_state(s=current_state, a=action)
            if not found_value:
                evals.append(0.0)

        return evals
            


