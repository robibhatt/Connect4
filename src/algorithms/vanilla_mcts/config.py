"""
Unified configuration for Vanilla MCTS algorithm.

Similar to AlphaZeroConfig but without neural network parameters.
"""

from __future__ import annotations
from dataclasses import dataclass

from src.algorithms.vanilla_mcts.mcts import MCTSConfig


@dataclass
class TrainerArgs:
    """Arguments for Vanilla MCTS trainer."""
    num_test_games: int = 10
    device: str = "cpu"  # Kept for consistency, not used
    random_seed: int | None = None
    verbose: bool = True


@dataclass
class VanillaMCTSConfig:
    """
    Complete Vanilla MCTS algorithm configuration.

    Combines:
    - TrainerArgs (4 fields): minimal training loop configuration
    - MCTSConfig (5 fields): MCTS search configuration

    Total: 9 fields (no model fields needed)

    Required Fields:
        None (all have defaults)

    Optional Fields:
        All fields have sensible defaults
    """

    # ===== Trainer Configuration (4 fields from TrainerArgs) =====
    num_test_games: int = 10           # Number of test games to run
    device: str = "cpu"                # Device (unused but kept for consistency)
    random_seed: int | None = None     # Random seed for reproducibility
    verbose: bool = True               # Print progress during testing

    # ===== MCTS Configuration (5 fields from MCTSConfig) =====
    num_sims: int = 1000               # Number of MCTS simulations per move
    c_exploration: float = 1.414       # UCB1 exploration constant (sqrt(2) by default)
    max_rollout_depth: int | None = None  # Max depth for rollouts (None = unlimited)
    rollout_seed: int | None = None    # Random seed for rollout policy
    illegal_action_penalty: float = 1e9   # Penalty for illegal actions

    def to_trainer_args(self) -> TrainerArgs:
        """Extract TrainerArgs from this config."""
        return TrainerArgs(
            num_test_games=self.num_test_games,
            device=self.device,
            random_seed=self.random_seed,
            verbose=self.verbose,
        )

    def to_mcts_config(self) -> MCTSConfig:
        """Extract MCTSConfig from this config."""
        return MCTSConfig(
            num_sims=self.num_sims,
            c_exploration=self.c_exploration,
            max_rollout_depth=self.max_rollout_depth,
            rollout_seed=self.rollout_seed,
            illegal_action_penalty=self.illegal_action_penalty,
        )
