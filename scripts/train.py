from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import yaml

from src.games.tictactoe import TicTacToe
from src.games.connect4 import Connect4
from src.models.registry import get_model_for_game, ModelRegistry
from src.mcts import MCTS, MCTSConfig
from src.training import Trainer, TrainerArgs


# Game registry
GAMES = {
    'tictactoe': TicTacToe,
    'connect4': Connect4,
}


def save_model_with_metadata(model, game_name: str, config: dict, root_dir="trained_models"):
    """
    Save model with timestamp and game name in folder name.

    Args:
        model: Trained model
        game_name: Name of the game (e.g., 'tictactoe')
        config: Full config dict to save alongside model
        root_dir: Root directory for saved models

    Returns:
        Path to saved model directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_class_name = model.__class__.__name__

    # Include game name in folder for easy identification
    folder_name = f"{timestamp}_{game_name}_{model_class_name}"

    save_dir = Path(root_dir) / folder_name
    save_dir.mkdir(parents=True, exist_ok=False)

    # Save model weights
    model_path = save_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    # Save config with game metadata
    config_with_metadata = {
        **config,
        'metadata': {
            'game': game_name,
            'model_class': model_class_name,
            'timestamp': timestamp,
        }
    }
    config_path = save_dir / "train.yaml"
    with config_path.open('w') as f:
        yaml.dump(config_with_metadata, f, default_flow_style=False)

    print(f"\nModel saved to: {save_dir}")
    print(f"  - Game: {game_name}")
    print(f"  - Model: {model_class_name}")

    return save_dir


def load_config(config_path: Path) -> Tuple[str, dict, TrainerArgs, MCTSConfig, dict]:
    """
    Load configuration from YAML file.

    Returns:
        game_name: Name of the game to train
        model_config: Model hyperparameters
        trainer_args: Training configuration
        mcts_config: MCTS configuration
        full_config: Complete config dict for saving
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    # Extract game name
    game_config = config.get("game", {})
    if isinstance(game_config, dict):
        game_name = game_config.get("name", "tictactoe")
    else:
        game_name = game_config  # backward compatibility

    # Extract model config
    model_config = config.get("model", {})

    # Extract training and MCTS configs
    train_cfg = config.get("train", {})
    mcts_cfg = config.get("mcts", {})

    trainer_args = TrainerArgs(**train_cfg)
    mcts_config = MCTSConfig(**mcts_cfg)

    return game_name, model_config, trainer_args, mcts_config, config


def main():
    config_path = Path(__file__).parent / "train.yaml"
    game_name, model_config, args, mcts_cfg, full_config = load_config(config_path)

    # Instantiate game
    if game_name not in GAMES:
        raise ValueError(
            f"Unknown game '{game_name}'. Available: {list(GAMES.keys())}"
        )
    game = GAMES[game_name]()

    # Get model class from registry and instantiate
    ModelClass = get_model_for_game(game)
    model = ModelClass(**model_config)

    print(f"\n{'='*60}")
    print(f"Training {ModelClass.__name__} for {game_name}")
    print(f"{'='*60}\n")

    # Validate compatibility
    ModelRegistry.validate_compatibility(game, model)

    # Create MCTS
    mcts = MCTS(
        game=game,
        model=model,
        device=torch.device(args.device),
        cfg=mcts_cfg
    )

    # Create trainer
    trainer = Trainer(
        game=game,
        model=model,
        mcts=mcts,
        args=args
    )

    # Run training
    trainer.run()

    # Save with metadata
    save_dir = save_model_with_metadata(
        model=model,
        game_name=game_name,
        config=full_config
    )


if __name__ == '__main__':
    main()






