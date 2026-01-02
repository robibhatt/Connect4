"""
Unified configuration for AlphaZero algorithm.

Merges all configuration from TrainerArgs, MCTSConfig, and model settings
into a single dataclass for cleaner config management.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any

from src.algorithms.alphazero.trainer import TrainerArgs
from src.algorithms.alphazero.mcts import MCTSConfig


@dataclass
class AlphaZeroConfig:
    """
    Complete AlphaZero algorithm configuration.

    Combines all parameters from:
    - TrainerArgs (14 fields): training loop configuration
    - MCTSConfig (5 fields): MCTS search configuration
    - Model configuration (2 fields): model class and kwargs

    Total: 21 fields

    Required Fields:
        model_class: Full model class name (e.g., 'TicTacToeMLPNet')

    Optional Fields (with defaults from TrainerArgs/MCTSConfig):
        All other fields have defaults matching the original classes
    """

    # ===== Model Configuration (2 fields) =====
    model_class: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ===== Trainer Configuration (14 fields from TrainerArgs) =====
    # Self-play
    iterations: int = 100
    games_per_iteration: int = 25
    temp_moves: int = 10
    tau: float = 1.0
    deterministic_after_temp: bool = True
    add_dirichlet_noise: bool = True

    # Training
    batch_size: int = 128
    train_steps_per_iteration: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_coef: float = 1.0

    # Buffer
    buffer_capacity: int = 100_000

    # Misc
    device: str = "mps"
    clear_mcts_each_game: bool = True

    # ===== MCTS Configuration (5 fields from MCTSConfig) =====
    num_sims: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    illegal_action_penalty: float = 1e9

    def to_trainer_args(self) -> TrainerArgs:
        """
        Extract TrainerArgs from this config.

        Returns subset of fields that belong to TrainerArgs.
        """
        return TrainerArgs(
            iterations=self.iterations,
            games_per_iteration=self.games_per_iteration,
            temp_moves=self.temp_moves,
            tau=self.tau,
            deterministic_after_temp=self.deterministic_after_temp,
            add_dirichlet_noise=self.add_dirichlet_noise,
            batch_size=self.batch_size,
            train_steps_per_iteration=self.train_steps_per_iteration,
            lr=self.lr,
            weight_decay=self.weight_decay,
            value_loss_coef=self.value_loss_coef,
            buffer_capacity=self.buffer_capacity,
            device=self.device,
            clear_mcts_each_game=self.clear_mcts_each_game,
        )

    def to_mcts_config(self) -> MCTSConfig:
        """
        Extract MCTSConfig from this config.

        Returns subset of fields that belong to MCTSConfig.
        """
        return MCTSConfig(
            num_sims=self.num_sims,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_eps=self.dirichlet_eps,
            illegal_action_penalty=self.illegal_action_penalty,
        )
