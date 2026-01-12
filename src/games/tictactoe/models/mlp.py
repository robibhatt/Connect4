from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.models.base import GameNet
from src.models.registry import ModelRegistry


class TicTacToeMLPNet(GameNet):
    """
    MLP neural network for TicTacToe game playing.

    Architecture:
      - Input: [B, 3, 3] canonical board (+1 me, -1 opponent, 0 empty)
      - Trunk: 2-layer MLP with ReLU
      - Policy head: Linear -> logits [B, 9]
      - Value head: Linear -> tanh -> value [B, 1]

    Notes:
      - Does NOT mask illegal moves (handled by MCTS)
      - Value uses tanh to output in [-1, 1]
    """

    game_name = "tictactoe"
    input_shape = (3, 3)
    action_size = 9

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden

        # Feature trunk
        self.fc1 = nn.Linear(9, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # Policy head
        self.policy = nn.Linear(hidden, 9)

        # Value head
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.validate_input(x)

        # Flatten board: [B, 3, 3] -> [B, 9]
        h = x.reshape(x.shape[0], 9).to(dtype=torch.float32)

        # Trunk
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # Heads
        logits = self.policy(h)  # [B, 9]
        value_raw = self.value(h)  # [B, 1]
        value = torch.tanh(value_raw)  # [B, 1] in [-1, 1]

        self.validate_output(logits, value)
        return logits, value


# Auto-register this model
ModelRegistry.register(TicTacToeMLPNet)
