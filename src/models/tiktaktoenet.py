from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TicTacToeNet(nn.Module):
    """
    Inputs / outputs (jaxtyping-style comments):

      x:      Float[Tensor, "B 3 3"]   (canonical board: +1 me, -1 them, 0 empty)
      logits: Float[Tensor, "B 9"]     (one logit per action 0..8)
      value:  Float[Tensor, "B 1"]     (scalar value per state)

    Notes:
      - We DON'T mask illegal moves here; do it in the Evaluator / MCTS.
      - Value head uses tanh so it naturally lives in [-1, 1].
    """

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: Float[Tensor, "B 3 3"]
        if x.ndim != 3 or x.shape[-2:] != (3, 3):
            raise ValueError(f"Expected x of shape [B,3,3], got {tuple(x.shape)}")

        # Flatten board
        # h: Float[Tensor, "B 9"]
        h = x.reshape(x.shape[0], 9).to(dtype=torch.float32)

        # trunk
        # h: Float[Tensor, "B H"]
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # logits: Float[Tensor, "B 9"]
        logits = self.policy(h)

        # value_raw: Float[Tensor, "B 1"]
        value_raw = self.value(h)

        # value: Float[Tensor, "B 1"] in [-1,1]
        value = torch.tanh(value_raw)

        return logits, value
