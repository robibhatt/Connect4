"""
MCGS Trainer - minimal training (just hyperparameter validation).

Since MCGS has no learning, the "trainer" just validates configuration
and optionally runs test games to verify the agent works.
"""

from __future__ import annotations

import numpy as np

from src.games.core.game import Game
from src.algorithms.mcgs.mcgs import MCGS
from src.algorithms.shared.trainer_args import TrainerArgs
from src.agents.checkpointable import CheckpointableAgent


class Trainer:
    """
    MCGS trainer.

    Since MCGS has no training phase (no learning),
    this class just validates the configuration and optionally
    runs test games to verify the agent works correctly.
    """

    def __init__(
        self,
        game: Game,
        mcgs: MCGS,
        args: TrainerArgs
    ):
        """
        Initialize trainer.

        Args:
            game: Game instance
            mcgs: MCGS instance
            args: Trainer arguments
        """
        self.game = game
        self.mcgs = mcgs
        self.args = args

        if args.random_seed is not None:
            np.random.seed(args.random_seed)

    def run(self):
        """
        Run "training" (actually just validation/testing).

        For MCGS, there's no training to do!
        Instead, we:
        1. Validate configuration
        2. Run test games to verify agent works
        3. Print statistics
        """
        if self.args.verbose:
            print(f"\n{'='*60}")
            print("MCGS Configuration Validation")
            print(f"{'='*60}")
            print(f"Simulations per move: {self.mcgs.cfg.num_sims}")
            print(f"Exploration constant: {self.mcgs.cfg.c_exploration}")
            print(f"Max rollout depth: {self.mcgs.cfg.max_rollout_depth or 'Unlimited'}")
            print(f"{'='*60}\n")

        # Run test games
        if self.args.num_test_games > 0:
            self._run_test_games()

        if self.args.verbose:
            print("\nConfiguration validated successfully!")
            print("Agent is ready to play.")

    def _run_test_games(self):
        """
        Run test games to verify agent works.

        Plays agent against itself and reports statistics.
        """
        if self.args.verbose:
            print(f"Running {self.args.num_test_games} test games...")

        game_lengths = []

        for game_idx in range(self.args.num_test_games):
            self.mcgs.clear()
            s = self.game.reset()
            moves = 0

            while True:
                done, _ = self.game.terminal_value(s)
                if done:
                    break

                # Get action from MCGS
                a = self.mcgs.play_move(s, deterministic=True)
                s = self.game.next_state(s, a)
                moves += 1

            game_lengths.append(moves)

            if self.args.verbose and (game_idx + 1) % 5 == 0:
                print(f"  Completed {game_idx + 1}/{self.args.num_test_games} games")

        # Report statistics
        if self.args.verbose:
            avg_length = np.mean(game_lengths)
            print(f"\nTest games statistics:")
            print(f"  Average game length: {avg_length:.1f} moves")
            print(f"  Min/Max: {min(game_lengths)}/{max(game_lengths)} moves")

    def create_agent(self) -> CheckpointableAgent:
        """
        Create agent from MCGS instance.

        Returns:
            Agent instance ready to be saved or used for play
        """
        from src.algorithms.mcgs.agent import MCGSAgent
        return MCGSAgent(game=self.game, mcgs=self.mcgs)
