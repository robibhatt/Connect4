"""Base class for game-specific neural networks."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional


class GameNet(nn.Module):
    """
    Abstract base class for game neural networks.

    All game-specific models should inherit from this class to ensure
    consistent interface for the training pipeline.

    Subclasses should override:
        game_name: str - Unique identifier for the game (e.g., "tictactoe")
        input_shape: Tuple[int, ...] - Expected input dimensions (e.g., (3, 3))
        action_size: int - Number of possible actions (e.g., 9)
    """

    # Subclasses should override these
    game_name: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    action_size: Optional[int] = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Canonical board representation [B, *input_shape]
               (+1 for current player, -1 for opponent, 0 for empty)

        Returns:
            logits: Policy logits [B, action_size]
            value: Value prediction [B, 1] in range [-1, 1]
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def validate_input(self, x: torch.Tensor) -> None:
        """
        Validate input tensor shape.

        Args:
            x: Input tensor to validate

        Raises:
            ValueError: If input shape doesn't match expected input_shape
        """
        if self.input_shape is None:
            return

        expected_shape = (x.shape[0],) + self.input_shape
        if x.shape != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, got {x.shape}"
            )

    def validate_output(
        self,
        logits: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Validate output tensor shapes.

        Args:
            logits: Policy logits tensor
            value: Value prediction tensor

        Raises:
            ValueError: If output shapes don't match expected dimensions
        """
        if self.action_size is None:
            return

        batch_size = logits.shape[0]

        if logits.shape != (batch_size, self.action_size):
            raise ValueError(
                f"Expected logits shape ({batch_size}, {self.action_size}), "
                f"got {logits.shape}"
            )

        if value.shape not in [(batch_size, 1), (batch_size,)]:
            raise ValueError(
                f"Expected value shape ({batch_size}, 1) or ({batch_size},), "
                f"got {value.shape}"
            )
