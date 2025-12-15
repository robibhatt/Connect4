from agent import RandomAgent
from games.tiktaktoe import TicTacToe
from models.tiktaktoenet import TicTacToeNet
from games.game_play import simulate_match
from agent import MCTSAgent
from mcts import MCTS, MCTSConfig
import torch
from pathlib import Path
from evaluator import TorchEvaluator
from games.tiktaktoe_vs_human import play_ttt_human_vs_agent_click


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

    model = load_model_from_dir(
        model_cls=TicTacToeNet,
        model_kwargs=None,
        model_dir='trained_models/20251215_121528_TicTacToeNet',
        device='mps'
    )


        
    random_agent = RandomAgent(game=game)

    mcts_cfg = MCTSConfig(
        num_sims=50,
        c_puct=1.25,
        dirichlet_alpha=0.6,   # higher alpha for small action space
        dirichlet_eps=0.20,    # slightly gentler than 0.25
        illegal_action_penalty=1e9,
    )

    evaluator = TorchEvaluator(model=model,
                               device=next(model.parameters()).device)
    
    mcts = MCTS(game=game,
                evaluator=evaluator,
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