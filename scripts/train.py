from pathlib import Path
from typing import Tuple, Any

import yaml

from src.games.core.registry import GameRegistry
from src.models.registry import ModelRegistry
from src.algorithms.registry import AlgorithmRegistry
from src.agents.checkpoint_utils import save_agent_checkpoint


def load_config(config_path: Path) -> Tuple[str, str, Any, dict]:
    """
    Load configuration from YAML file.

    Supports algorithm-based format:
        game: tictactoe
        algorithm:
          name: alphazero  # or vanilla_mcts
          device: mps
          model:           # optional, only for algorithms that need it
            class: TicTacToeMLPNet
            hidden: 64
          iterations: 400
          num_sims: 50
          ...

    Returns:
        game_name: Name of the game to train
        algo_name: Name of the algorithm
        config: Algorithm config instance (AlphaZeroConfig, VanillaMCTSConfig, etc.)
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

    # Build config params from algorithm section (excluding 'name' and 'model')
    config_params = {k: v for k, v in algo_config.items() if k not in ['name', 'model']}

    # Extract model config if present (only some algorithms need it)
    model_config = algo_config.get("model", {})
    if model_config:
        model_config = model_config.copy()
        if 'class' in model_config:
            model_class_name = model_config.pop('class')
            config_params['model_class'] = model_class_name
            config_params['model_kwargs'] = model_config  # Remaining fields are kwargs

    # Instantiate config
    config = ConfigClass(**config_params)

    return game_name, algo_name, config, config_dict


def main():
    config_path = Path(__file__).parent / "train.yaml"
    game_name, algo_name, config, full_config = load_config(config_path)

    # Instantiate game using GameRegistry
    GameClass = GameRegistry.get_game(game_name)
    game = GameClass()

    # Conditionally create model (only if config specifies one)
    model = None
    if hasattr(config, 'model_class') and config.model_class:
        ModelClass = ModelRegistry.get_model(config.model_class)
        model = ModelClass(**config.model_kwargs)
        ModelRegistry.validate_compatibility(game, model)
        model_name = ModelClass.__name__
    else:
        model_name = "N/A"

    print(f"\n{'='*60}")
    print(f"Training for {game_name}")
    print(f"Algorithm: {algo_name}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    # Create trainer using registry factory (uses algo_name, not hardcoded!)
    factory = AlgorithmRegistry.get_trainer_factory(algo_name)
    trainer = factory(game, model, config)

    # Run training
    trainer.run()

    # Create agent from trained model
    agent = trainer.create_agent()

    # Build agent config using registry factory
    agent_config_factory = AlgorithmRegistry.get_agent_config_factory(algo_name)
    agent_config = agent_config_factory(config)

    # Save agent checkpoint
    agent_class_name = agent.__class__.__name__
    save_dir = save_agent_checkpoint(
        agent=agent,
        agent_class_name=agent_class_name,
        game_name=game_name,
        config=agent_config,
        training_config=full_config,
        root_dir="saved_agents"
    )

    print(f"\nAgent saved to: {save_dir}")
    print(f"  - Game: {game_name}")
    print(f"  - Agent: {agent_class_name}")


if __name__ == '__main__':
    main()






