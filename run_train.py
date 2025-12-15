import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import yaml

from agent import RandomAgent, MCTSAgent
from evaluator import TorchEvaluator
from games.game_play import simulate_match
from games.tiktaktoe import TicTacToe
from mcts import MCTSConfig, MCTS
from models.tiktaktoenet import TicTacToeNet
from trainer import TrainerArgs, Trainer

def save_model_with_timestamp(model, root_dir="trained_models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.__class__.__name__
    folder_name = f"{timestamp}_{model_name}"

    save_dir = Path(root_dir) / folder_name
    save_dir.mkdir(parents=True, exist_ok=False)

    save_path = save_dir / "model.pt"
    torch.save(model.state_dict(), save_path)

    return save_dir


def load_config(config_path: Path) -> Tuple[TrainerArgs, MCTSConfig, dict]:
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    train_cfg = config.get("train", {})
    mcts_cfg = config.get("mcts", {})

    trainer_args = TrainerArgs(**train_cfg)
    mcts_config = MCTSConfig(**mcts_cfg)

    return trainer_args, mcts_config, config


def main():

    config_path = Path("train.yaml")
    args, mcts_cfg, _ = load_config(config_path)

    game = TicTacToe()
    model = TicTacToeNet()
    evaluator = TorchEvaluator(model=model,
                               device=torch.device(args.device))

    mcts = MCTS(game=game,
                evaluator=evaluator,
                cfg=mcts_cfg)
    
    trainer = Trainer(
        game=game,
        model=model,
        mcts=mcts,
        args=args
    )

    trainer.run()

    save_dir = save_model_with_timestamp(model)
    shutil.copy2(config_path, save_dir / config_path.name)

    # testing code
    # random_agent = RandomAgent(game=game)
    # mcts_agent = MCTSAgent(game= game, mcts=mcts)
    # simulate_match(
    #     game=game,
    #     agent1=random_agent,
    #     agent2=mcts_agent,
    #     num_games=100
    # )


if __name__ == '__main__':
    main()






