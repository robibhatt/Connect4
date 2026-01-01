from agents.agent import RandomAgent, MCTSAgent
from games.tiktaktoe import TicTacToe
from models.tiktaktoenet import TicTacToeNet
from games.game_play import simulate_match
from mcts.alphazero_mcts import MCTS, MCTSConfig
import torch
from pathlib import Path
from games.tiktaktoe_vs_human import play_ttt_human_vs_agent_click
import yaml


def load_model_from_dir(
    model_cls,
    model_dir,
    model_kwargs=None,
    device="cpu",
    strict=True,
):
    """
    model_cls: nn.Module class (e.g. TicTacToeNet)
    model_dir: str or Path to folder containing model.pt
    model_kwargs: dict or None
    device: torch device or string
    strict: passed to load_state_dict
    """
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No model.pt found in {model_dir}")

    # Construct model
    try:
        if model_kwargs is None:
            model = model_cls()
        else:
            model = model_cls(**model_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to construct {model_cls.__name__} "
            f"with kwargs={model_kwargs}"
        ) from e

    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        if missing:
            print(f"[warn] missing keys: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def main():

    game = TicTacToe()

    model_dir = Path('trained_models/20251215_144209_TicTacToeNet')

    model = load_model_from_dir(
        model_cls=TicTacToeNet,
        model_kwargs=None,
        model_dir=model_dir,
        device='mps'
    )


        
    random_agent = RandomAgent(game=game)

    cfg_path = model_dir / "train.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No train.yaml found in {model_dir}")

    with cfg_path.open("r") as f:
        saved_cfg = yaml.safe_load(f) or {}

    mcts_cfg = MCTSConfig(**saved_cfg.get("mcts", {}))

    mcts = MCTS(game=game,
                model=model,
                device=next(model.parameters()).device,
                cfg=mcts_cfg)

    mcts_agent = MCTSAgent(game= game, mcts=mcts)
    simulate_match(
        game=game,
        agent1=random_agent,
        agent2=mcts_agent,
        num_games=100
    )


    play_ttt_human_vs_agent_click(
        game=game,
        agent=mcts_agent,
        pause_seconds=0.4
    )





if __name__ == '__main__':
    main()