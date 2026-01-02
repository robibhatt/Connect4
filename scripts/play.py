from pathlib import Path
from typing import Optional, Tuple, Type
import yaml

from src.agents import RandomAgent, AlphaZeroMCTSAgent
from src.games.tictactoe import TicTacToe, TicTacToeNet
from src.games.connect4 import Connect4, Connect4Net
from src.games.core.game_play import simulate_match
from src.games.tictactoe.ui import TicTacToeUI
from src.games.connect4.ui import Connect4UI
from src.models.registry import ModelRegistry
import torch.nn as nn


# Game/Model/UI registries
GAMES = {
    'tictactoe': TicTacToe,
    'connect4': Connect4,
}

MODELS = {
    'tictactoe': TicTacToeNet,
    'connect4': Connect4Net,
}

UIS = {
    'tictactoe': TicTacToeUI,
    'connect4': Connect4UI,
}


def load_game_from_checkpoint(model_dir: Path) -> Tuple[str, Type[nn.Module]]:
    """
    Extract game name and model class from checkpoint directory.

    Args:
        model_dir: Path to model checkpoint directory

    Returns:
        (game_name, model_class)
    """
    config_path = model_dir / "train.yaml"

    if config_path.exists():
        with config_path.open('r') as f:
            config = yaml.safe_load(f) or {}

        # Try to get game from metadata (new format)
        metadata = config.get('metadata', {})
        if 'game' in metadata:
            game_name = metadata['game']
            model_cls = MODELS.get(game_name)
            if model_cls:
                return game_name, model_cls

        # Try to get from game config (also new format)
        game_config = config.get('game', {})
        if isinstance(game_config, dict) and 'name' in game_config:
            game_name = game_config['name']
            model_cls = MODELS.get(game_name)
            if model_cls:
                return game_name, model_cls

    # Fallback: infer from directory name
    # Format: YYYYMMDD_HHMMSS_<game>_<ModelClass>
    dir_name = model_dir.name
    parts = dir_name.split('_')

    if len(parts) >= 3:
        # Try to find game name in directory name
        for game_name in GAMES.keys():
            if game_name in dir_name.lower():
                return game_name, MODELS[game_name]

    # Final fallback: assume TicTacToe (backward compatibility)
    print(f"Warning: Could not determine game from {model_dir}. Assuming tictactoe.")
    return 'tictactoe', TicTacToeNet


def main():
    # Configuration
    model_dir = Path('trained_models/20251215_144209_TicTacToeNet')

    # Auto-detect game and model from checkpoint
    game_name, model_cls = load_game_from_checkpoint(model_dir)

    print(f"\n{'='*60}")
    print(f"Playing {game_name} with {model_cls.__name__}")
    print(f"{'='*60}\n")

    # Instantiate game
    game = GAMES[game_name]()

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