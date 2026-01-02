from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import yaml

from src.games.core.registry import GameRegistry
from src.models.registry import ModelRegistry
from src.algorithms.alphazero import MCTS, MCTSConfig, Trainer, TrainerArgs
from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.alphazero.config import AlphaZeroConfig


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


def load_config(config_path: Path) -> Tuple[str, AlphaZeroConfig, dict]:
    """
    Load configuration from YAML file.

    Supports new algorithm-based format:
        game: tictactoe
        algorithm:
          name: alphazero
          device: mps
          model:
            class: TicTacToeMLPNet
            hidden: 64
          iterations: 400
          num_sims: 50
          ...

    Returns:
        game_name: Name of the game to train
        config: AlphaZeroConfig instance
        full_config: Complete config dict for saving
    """
    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f) or {}

    # Extract game name
    game_config = config_dict.get("game", {})
    if isinstance(game_config, dict):
        game_name = game_config.get("name", "tictactoe")
    else:
        game_name = game_config  # Simple string format: game: tictactoe

    # Extract algorithm section
    algo_config = config_dict.get("algorithm", {})
    if not algo_config:
        raise ValueError(
            "Config must include 'algorithm' section with 'name' field. "
            "Example: algorithm: { name: alphazero, ... }"
        )

    algo_name = algo_config.get("name")
    if not algo_name:
        raise ValueError("Algorithm section must include 'name' field")

    # Get config class from registry
    ConfigClass = AlgorithmRegistry.get_config_class(algo_name)

    # Extract model config from nested structure
    model_config = algo_config.get("model", {}).copy()
    if 'class' not in model_config:
        raise ValueError(
            "Algorithm.model must include 'class' field. "
            "Example: model: { class: TicTacToeMLPNet, hidden: 64 }"
        )

    model_class_name = model_config.pop('class')
    model_kwargs = model_config  # Remaining fields are kwargs

    # Build config dict (flatten algorithm section)
    config_params = {
        'model_class': model_class_name,
        'model_kwargs': model_kwargs,
        **{k: v for k, v in algo_config.items() if k not in ['name', 'model']}
    }

    # Instantiate config
    config = ConfigClass(**config_params)

    return game_name, config, config_dict


def main():
    config_path = Path(__file__).parent / "train.yaml"
    game_name, config, full_config = load_config(config_path)

    # Instantiate game using GameRegistry
    GameClass = GameRegistry.get_game(game_name)
    game = GameClass()

    # Get model class from ModelRegistry and instantiate
    ModelClass = ModelRegistry.get_model(config.model_class)
    model = ModelClass(**config.model_kwargs)

    print(f"\n{'='*60}")
    print(f"Training {ModelClass.__name__} for {game_name}")
    print(f"Algorithm: AlphaZero")
    print(f"{'='*60}\n")

    # Validate compatibility
    ModelRegistry.validate_compatibility(game, model)

    # Create trainer using registry factory
    factory = AlgorithmRegistry.get_trainer_factory('alphazero')
    trainer = factory(game, model, config)

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






