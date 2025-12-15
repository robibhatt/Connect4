from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from games.game import Game, State


class Evaluator:
    """
    Abstract interface used by MCTS.

    evaluate(game, s) -> (logits[action_size], value_scalar)
      - logits are raw scores (no softmax)
      - value is in [-1, 1], from viewpoint of player-to-move
    """
    def evaluate(self, game: Game, s: State) -> tuple[np.ndarray, float]:
        raise NotImplementedError


class TorchEvaluator(Evaluator):
    """
    model(x) -> (logits, value)

    Shapes (jaxtyping-style comments):
      x:      Float[Tensor, "B ..."]
      logits: Float[Tensor, "B A"]
      value:  Float[Tensor, "B 1"]   # REQUIRED
    """

    def __init__(self, model: nn.Module, device: torch.device | None = None):
        self.model = model
        self.device = device
        if self.device is not None:
            self.model.to(self.device)

    def evaluate(self, game: Game, s: State) -> tuple[np.ndarray, float]:
        # encode state -> torch, batch size = 1
        # x: Float[np.ndarray, "..."]
        x_np = game.encode(s).astype(np.float32, copy=False)

        # x: Float[Tensor, "1 ..."]
        x = torch.from_numpy(x_np).unsqueeze(0)
        if self.device is not None:
            x = x.to(self.device)

        # run model in eval mode, but restore afterwards
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                # logits_b: Float[Tensor, "1 A"]
                # value_b:  Float[Tensor, "1 1"]   (required)
                logits_b, value_b = self.model(x)
        finally:
            self.model.train(was_training)

        # ---- enforce value shape contract ----
        if value_b.ndim != 2 or value_b.shape[1] != 1:
            raise ValueError(
                f"Model must return value of shape [B, 1], got {tuple(value_b.shape)}"
            )

        # remove batch dimension
        # logits: Float[Tensor, "A"]
        logits = logits_b.squeeze(0)

        # value: Float[Tensor, "1"] -> scalar
        value = value_b.squeeze(0).squeeze(0)

        # ---- masked softmax over legal actions ----
        # legal: Bool[np.ndarray, "A"]
        legal = game.legal_actions(s)
        if legal.dtype != np.bool_:
            legal = legal.astype(bool)

        # legal_t: Bool[Tensor, "A"]
        legal_t = torch.from_numpy(legal).to(logits.device)

        # masked_logits: Float[Tensor, "A"]
        masked_logits = logits.masked_fill(~legal_t, -1e9)

        # priors_t: Float[Tensor, "A"]
        priors_t = torch.softmax(masked_logits, dim=0)

        # priors: Float[np.ndarray, "A"]
        priors = priors_t.detach().cpu().numpy().astype(np.float32, copy=False)

        # v: float
        v = float(value.detach().cpu().item())

        return priors, v
