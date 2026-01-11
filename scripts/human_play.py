"""
Human play script with YAML configuration.

Allows playing games against AI agents (AlphaZero or Random) with configuration
automatically loaded from scripts/human_play.yaml.

Usage:
    python -m scripts.human_play

Configuration:
    Edit scripts/human_play.yaml to choose your game and opponent.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from src.agents import RandomAgent, load_agent_checkpoint
from src.agents.agent import Agent
from src.algorithms.registry import AlgorithmRegistry
from src.games.core.game import Game
from src.games.core.registry import GameRegistry
from src.games.tictactoe.ui import TicTacToeUI
from src.games.connect4.ui import Connect4UI


# UI registry (keep for now - UI registry is out of scope)
UIS = {
    'tictactoe': TicTacToeUI,
    'connect4': Connect4UI,
}


@dataclass
class PlayConfig:
    """Validated play configuration."""
    game_name: str
    agent_type: str  # 'alphazero' or 'random'
    checkpoint_dir: Optional[Path]  # None if agent_type='random'
    pause_seconds: float = 0.4


def print_error(message: str) -> None:
    """Print formatted error message."""
    print("=" * 60)
    print("Configuration Error")
    print("=" * 60)
    print()
    print(message)
    print()
    print("=" * 60)


def validate_config_schema(config_dict: dict) -> None:
    """
    Validate that required fields exist in config.

    Args:
        config_dict: Loaded YAML configuration

    Raises:
        ValueError: If required fields are missing
    """
    # Check for game field
    if 'game' not in config_dict:
        raise ValueError(
            "Problem: Missing required field 'game'\n\n"
            "Details:\n"
            "  The configuration must specify which game to play.\n\n"
            "To fix:\n"
            "  Add a 'game' field to your config:\n\n"
            "  game: tictactoe  # or 'connect4'\n\n"
            "Available games: tictactoe, connect4"
        )

    # Check for agent field
    if 'agent' not in config_dict:
        raise ValueError(
            "Problem: Missing required field 'agent'\n\n"
            "Details:\n"
            "  The configuration must specify which agent to play against.\n\n"
            "To fix:\n"
            "  Add an 'agent' section to your config:\n\n"
            "  agent:\n"
            "    type: random  # or 'alphazero'\n\n"
            "For AlphaZero agents, also add checkpoint_dir:\n"
            "  agent:\n"
            "    type: alphazero\n"
            "    checkpoint_dir: saved_agents/YOUR_CHECKPOINT"
        )

    # Check for agent.type field
    if 'type' not in config_dict['agent']:
        raise ValueError(
            "Problem: Missing required field 'agent.type'\n\n"
            "Details:\n"
            "  The agent configuration must specify the agent type.\n\n"
            "To fix:\n"
            "  Add a 'type' field under 'agent':\n\n"
            "  agent:\n"
            "    type: random  # Options: 'random' or 'alphazero'"
        )


def validate_game_exists(game_name: str) -> None:
    """
    Validate that game is registered.

    Args:
        game_name: Game identifier

    Raises:
        ValueError: If game is not registered
    """
    try:
        GameRegistry.get_game(game_name)
    except KeyError:
        available_games = ['tictactoe', 'connect4']
        raise ValueError(
            f"Problem: Unknown game '{game_name}'\n\n"
            f"Available games: {', '.join(available_games)}\n\n"
            "To fix:\n"
            "  Set 'game' to one of the available game names:\n\n"
            "  game: tictactoe  # or 'connect4'"
        )


def validate_agent_type(agent_type: str) -> None:
    """
    Validate that agent type is registered.

    Args:
        agent_type: Agent type string

    Raises:
        ValueError: If agent type is not registered
    """
    valid_types = AlgorithmRegistry.get_all_algorithms()
    if agent_type not in valid_types:
        raise ValueError(
            f"Invalid agent type '{agent_type}'. Valid: {', '.join(sorted(valid_types))}"
        )


def validate_checkpoint_exists(checkpoint_dir: Path, agent_type: str) -> None:
    """
    Validate that checkpoint directory and required files exist.

    Args:
        checkpoint_dir: Path to checkpoint directory
        agent_type: Agent type (determines required files via metadata)

    Raises:
        ValueError: If checkpoint directory or files don't exist
    """
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_dir}")

    metadata = AlgorithmRegistry.get_metadata(agent_type)
    for required_file in metadata.checkpoint_files:
        if not (checkpoint_dir / required_file).exists():
            raise ValueError(f"Checkpoint missing {required_file}: {checkpoint_dir}")


def validate_game_match(config_game: str, checkpoint_dir: Path) -> None:
    """
    Validate that config game matches checkpoint game.

    Args:
        config_game: Game name from config
        checkpoint_dir: Path to checkpoint directory

    Raises:
        ValueError: If games don't match
    """
    agent_yaml_path = checkpoint_dir / "agent.yaml"
    with agent_yaml_path.open('r') as f:
        agent_yaml = yaml.safe_load(f)

    checkpoint_game = agent_yaml.get('game')

    if config_game != checkpoint_game:
        raise ValueError(
            f"Problem: Game mismatch between config and checkpoint\n\n"
            f"Details:\n"
            f"  Config specifies: {config_game}\n"
            f"  Checkpoint contains: {checkpoint_game}\n\n"
            "To fix:\n"
            f"  Option 1: Change 'game' in config to '{checkpoint_game}'\n"
            f"  Option 2: Use a checkpoint trained for '{config_game}'"
        )


def load_play_config(config_path: Path) -> PlayConfig:
    """
    Load and validate play configuration from YAML.

    Args:
        config_path: Path to YAML config file

    Returns:
        PlayConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Check config file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Problem: Configuration file not found\n\n"
            f"Details:\n"
            f"  Path: {config_path}\n\n"
            "To fix:\n"
            "  Ensure the config file exists and the path is correct.\n\n"
            "Example usage:\n"
            "  python scripts/human_play.py configs/play/tictactoe_random.yaml"
        )

    # Load YAML
    try:
        with config_path.open('r') as f:
            config_dict = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(
            f"Problem: Invalid YAML syntax\n\n"
            f"Details:\n"
            f"  File: {config_path}\n"
            f"  Error: {e}\n\n"
            "To fix:\n"
            "  Check YAML syntax using a validator."
        )

    # Validate schema
    validate_config_schema(config_dict)

    # Extract fields
    game_name = config_dict['game']
    agent_config = config_dict['agent']
    agent_type = agent_config['type']
    checkpoint_dir_str = agent_config.get('checkpoint_dir')
    ui_config = config_dict.get('ui', {})
    pause_seconds = ui_config.get('pause_seconds', 0.4)

    # Validate game
    validate_game_exists(game_name)

    # Validate agent type
    validate_agent_type(agent_type)

    # Handle checkpoint-based agents using metadata
    checkpoint_dir = None
    metadata = AlgorithmRegistry.get_metadata(agent_type)
    if metadata.requires_checkpoint:
        if not checkpoint_dir_str:
            raise ValueError(f"{agent_type} requires checkpoint_dir")
        checkpoint_dir = Path(checkpoint_dir_str)
        validate_checkpoint_exists(checkpoint_dir, agent_type)
        validate_game_match(game_name, checkpoint_dir)

    return PlayConfig(
        game_name=game_name,
        agent_type=agent_type,
        checkpoint_dir=checkpoint_dir,
        pause_seconds=pause_seconds
    )




def create_game(game_name: str) -> Game:
    """
    Create game instance from GameRegistry.

    Args:
        game_name: Game identifier

    Returns:
        Game: Instantiated game
    """
    GameClass = GameRegistry.get_game(game_name)
    return GameClass()


def create_agent(config: PlayConfig, game: Game) -> Agent:
    """
    Create agent based on config type using AlgorithmRegistry metadata.

    Args:
        config: Play configuration
        game: Game instance

    Returns:
        Agent: Configured agent ready to play
    """
    metadata = AlgorithmRegistry.get_metadata(config.agent_type)
    if metadata.requires_checkpoint:
        return load_agent_checkpoint(config.checkpoint_dir)
    return RandomAgent(game=game)


def print_game_info(config: PlayConfig, agent: Agent, game: Game) -> None:
    """
    Print game and agent information.

    Args:
        config: Play configuration
        agent: Agent instance
        game: Game instance
    """
    print(f"\n{'='*60}")
    print(f"Playing {config.game_name.upper()}")
    print(f"Opponent: {agent.__class__.__name__}")
    print(f"{'='*60}\n")


def launch_ui(config: PlayConfig, game: Game, agent: Agent) -> None:
    """
    Create and launch game UI.

    Args:
        config: Play configuration
        game: Game instance
        agent: Agent instance
    """
    game_name = game.__class__.__name__.lower()
    ui_cls = UIS.get(game_name)

    if not ui_cls:
        raise ValueError(f"No UI available for {game_name}")

    ui = ui_cls(
        game=game,
        agent=agent,
        pause_seconds=config.pause_seconds
    )
    ui.run()


def main() -> None:
    """Main entry point."""
    try:
        # Automatically load config from same directory as script
        script_dir = Path(__file__).parent
        config_path = script_dir / "human_play.yaml"

        # Load and validate configuration
        config = load_play_config(config_path)

        # Create game instance
        game = create_game(config.game_name)

        # Create agent instance
        agent = create_agent(config, game)

        # Print game info
        print_game_info(config, agent, game)

        # Launch UI
        launch_ui(config, game, agent)

    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()
