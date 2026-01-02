from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.models.base import GameNet


class Connect4Net(GameNet):
    """
    Neural network for Connect4 AlphaZero training.

    Architecture:
      - Input: [B, 6, 7] canonical board (+1 me, -1 opponent, 0 empty)
      - Trunk: 2-layer MLP with ReLU
      - Policy head: Linear -> logits [B, 7]
      - Value head: Linear -> tanh -> value [B, 1]

    Notes:
      - Does NOT mask illegal moves (handled by MCTS)
      - Value uses tanh to output in [-1, 1]
      - Larger hidden size than TicTacToe due to bigger state space
    """

    game_name = "connect4"
    input_shape = (6, 7)
    action_size = 7

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = hidden

        # Feature trunk
        self.fc1 = nn.Linear(42, hidden)  # 6*7 = 42
        self.fc2 = nn.Linear(hidden, hidden)

        # Policy head
        self.policy = nn.Linear(hidden, 7)

        # Value head
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.validate_input(x)

        # Flatten board: [B, 6, 7] -> [B, 42]
        h = x.reshape(x.shape[0], 42).to(dtype=torch.float32)

        # Trunk
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # Heads
        logits = self.policy(h)  # [B, 7]
        value_raw = self.value(h)  # [B, 1]
        value = torch.tanh(value_raw)  # [B, 1] in [-1, 1]

        self.validate_output(logits, value)
        return logits, value
