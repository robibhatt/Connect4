from pathlib import Path
from typing import Optional, Tuple, Type
import yaml

from src.agents import RandomAgent, AlphaZeroMCTSAgent
from src.games.core.registry import GameRegistry
from src.games.core.game_play import simulate_match
from src.games.tictactoe.ui import TicTacToeUI
from src.games.connect4.ui import Connect4UI
from src.models.registry import ModelRegistry
import torch.nn as nn


# UI registry (keep for now - UI registry is out of scope)
UIS = {
    'tictactoe': TicTacToeUI,
    'connect4': Connect4UI,
}


def load_game_from_checkpoint(model_dir: Path) -> Tuple[str, str]:
    """
    Extract game name and model class name from checkpoint directory.

    Args:
        model_dir: Path to model checkpoint directory

    Returns:
        (game_name, model_class_name)
    """
    config_path = model_dir / "train.yaml"

    # Legacy model name mapping (for old checkpoints)
    LEGACY_MODEL_MAP = {
        'TicTacToeNet': 'TicTacToeMLPNet',
        'Connect4Net': 'Connect4MLPNet',
    }

    if config_path.exists():
        with config_path.open('r') as f:
            config = yaml.safe_load(f) or {}

        # Try to get model class from model config (new format)
        model_config = config.get('model', {})
        if 'class' in model_config:
            model_class_name = model_config['class']
            # Get game name from metadata or game config
            game_name = config.get('metadata', {}).get('game') or config.get('game', {}).get('name', 'tictactoe')
            return game_name, model_class_name

        # Try to get from metadata (old format - needs legacy mapping)
        metadata = config.get('metadata', {})
        if 'game' in metadata and 'model_class' in metadata:
            game_name = metadata['game']
            model_class_name = metadata['model_class']
            # Apply legacy mapping
            model_class_name = LEGACY_MODEL_MAP.get(model_class_name, model_class_name)
            return game_name, model_class_name

        # Try to get game name from config
        game_config = config.get('game', {})
        if isinstance(game_config, dict) and 'name' in game_config:
            game_name = game_config['name']
            # Infer model name from game name (legacy default)
            model_class_name = f"{game_name.capitalize()}MLPNet" if game_name == 'tictactoe' else f"{game_name.capitalize()}MLPNet"
            model_class_name = LEGACY_MODEL_MAP.get(f"{game_name.capitalize()}Net", model_class_name)
            return game_name, model_class_name

    # Fallback: infer from directory name
    # Format: YYYYMMDD_HHMMSS_<game>_<ModelClass>
    dir_name = model_dir.name
    parts = dir_name.split('_')

    if len(parts) >= 4:
        # Extract model class name from directory
        model_class_name = parts[-1]  # Last part is model class name
        # Apply legacy mapping
        model_class_name = LEGACY_MODEL_MAP.get(model_class_name, model_class_name)

        # Find game name
        for available_game in GameRegistry.list_games():
            if available_game in dir_name.lower():
                return available_game, model_class_name

    # Final fallback: assume TicTacToe with MLP
    print(f"Warning: Could not determine game/model from {model_dir}. Assuming tictactoe/TicTacToeMLPNet.")
    return 'tictactoe', 'TicTacToeMLPNet'


def main():
    # Configuration
    model_dir = Path('trained_models/20251215_144209_TicTacToeNet')

    # Auto-detect game and model from checkpoint
    game_name, model_class_name = load_game_from_checkpoint(model_dir)

    print(f"\n{'='*60}")
    print(f"Playing {game_name} with {model_class_name}")
    print(f"{'='*60}\n")

    # Instantiate game using GameRegistry
    game_cls = GameRegistry.get_game(game_name)
    game = game_cls()

    # Get model class from ModelRegistry
    model_cls = ModelRegistry.get_model(model_class_name)

    # Create agents
    random_agent = RandomAgent(game=game)

    mcts_agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=model_dir,
        game=game,
        model_cls=model_cls,
        device='mps'
    )

    # Validate compatibility
    ModelRegistry.validate_compatibility(game, mcts_agent.mcts.model)

    # Simulate matches
    simulate_match(
        game=game,
        agent1=random_agent,
        agent2=mcts_agent,
        num_games=100
    )

    # Play interactively with UI
    ui_cls = UIS.get(game_name)
    if ui_cls:
        ui = ui_cls(game=game, agent=mcts_agent, pause_seconds=0.4)
        ui.run()
    else:
        print(f"No UI available for {game_name}")


if __name__ == '__main__':
    main()