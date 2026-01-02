from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import yaml

from src.games.core.game import Game, State
from src.mcts import MCTS, MCTSConfig


class AlphaZeroMCTSAgent:
    """
    Agent that uses AlphaZero-style MCTS with a neural network.

    Can be instantiated from a trained model checkpoint directory,
    which contains model.pt and train.yaml with MCTS hyperparameters.
    """

    def __init__(self, game: Game, mcts: MCTS):
        """
        Direct construction (for advanced use or backwards compatibility).

        Args:
            game: Game instance
            mcts: Pre-configured MCTS instance
        """
        self.game = game
        self.mcts = mcts

    def start(self) -> None:
        """Called at the beginning of a new game. Clears MCTS tree."""
        self.mcts.clear()

    def act(self, s: State) -> int:
        """Choose an action by running MCTS from state s."""
        return self.mcts.play_move(s)

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        game: Game,
        model_cls: type[nn.Module],
        device: Optional[str | torch.device] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        mcts_params: Optional[Dict[str, Any]] = None,
    ) -> AlphaZeroMCTSAgent:
        """
        Create agent from a trained model checkpoint directory.

        Args:
            model_dir: Path to directory containing model.pt and train.yaml
            game: Game instance
            model_cls: Model class to instantiate (e.g., TicTacToeNet)
            device: Device for inference ('cpu', 'cuda', 'mps', etc.)
            model_kwargs: Optional kwargs for model construction
            mcts_params: Optional MCTS hyperparameter overrides

        Returns:
            AlphaZeroMCTSAgent ready to play

        Example:
            agent = AlphaZeroMCTSAgent.from_checkpoint(
                model_dir='trained_models/20251215_144209_TicTacToeNet',
                game=TicTacToe(),
                model_cls=TicTacToeNet,
                device='mps'
            )
        """
        model_dir = Path(model_dir)
        device = torch.device(device) if device else torch.device('cpu')

        # Load model
        model = _load_model(
            model_cls=model_cls,
            model_dir=model_dir,
            model_kwargs=model_kwargs,
            device=device
        )

        # Load MCTS config from train.yaml
        cfg_path = model_dir / "train.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No train.yaml found in {model_dir}")

        with cfg_path.open("r") as f:
            saved_cfg = yaml.safe_load(f) or {}

        # Merge saved config with overrides
        mcts_config_dict = saved_cfg.get("mcts", {})
        if mcts_params:
            mcts_config_dict.update(mcts_params)

        mcts_cfg = MCTSConfig(**mcts_config_dict)

        # Create MCTS
        mcts = MCTS(
            game=game,
            model=model,
            device=device,
            cfg=mcts_cfg
        )

        return cls(game=game, mcts=mcts)


def _load_model(
    model_cls: type[nn.Module],
    model_dir: Path,
    model_kwargs: Optional[Dict[str, Any]],
    device: torch.device,
) -> nn.Module:
    """
    Load a model from a checkpoint directory.

    Helper function extracted from scripts/play.py for reuse.

    Args:
        model_cls: Model class to instantiate
        model_dir: Directory containing model.pt
        model_kwargs: Optional kwargs for model construction
        device: Device to load model onto

    Returns:
        Loaded model in eval mode
    """
    ckpt_path = model_dir / "model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No model.pt found in {model_dir}")

    # Construct model
    if model_kwargs is None:
        model = model_cls()
    else:
        model = model_cls(**model_kwargs)

    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model
