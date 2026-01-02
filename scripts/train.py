import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import yaml

from src.games.tiktaktoe import TicTacToe
from src.mcts.alphazero_mcts import MCTSConfig, MCTS
from src.models.tiktaktoenet import TicTacToeNet
from src.training.trainer import TrainerArgs, Trainer

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

    mcts = MCTS(game=game,
                model=model,
                device=torch.device(args.device),
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


if __name__ == '__main__':
    main()






