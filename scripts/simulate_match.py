"""
Simulate match between two agents with YAML configuration.

Runs two agents against each other for multiple games and prints detailed
statistics including wins, losses, draws, and performance as first/second player.

Usage:
    python -m scripts.simulate_match

Configuration:
    Edit scripts/simulate_match.yaml to configure game, agents, and number of games.
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
from src.games.core.game_play import simulate_match
from src.games.core.registry import GameRegistry


@dataclass
class MatchConfig:
    """Validated match configuration."""
    game_name: str
    num_games: int
    agent1_type: str
    agent1_checkpoint_dir: Optional[Path]
    agent2_type: str
    agent2_checkpoint_dir: Optional[Path]


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

    if 'match' not in config_dict:
        raise ValueError(
            "Problem: Missing required field 'match'\n\n"
            "Details:\n"
            "  The configuration must specify match settings.\n\n"
            "To fix:\n"
            "  Add a 'match' section to your config:\n\n"
            "  match:\n"
            "    num_games: 100"
        )

    if 'num_games' not in config_dict['match']:
        raise ValueError(
            "Problem: Missing required field 'match.num_games'\n\n"
            "Details:\n"
            "  The match configuration must specify number of games.\n\n"
            "To fix:\n"
            "  Add num_games to your match section:\n\n"
            "  match:\n"
            "    num_games: 100"
        )

    for agent_num in [1, 2]:
        agent_key = f'agent{agent_num}'
        if agent_key not in config_dict:
            raise ValueError(
                f"Problem: Missing required field '{agent_key}'\n\n"
                f"Details:\n"
                f"  The configuration must specify both agent1 and agent2.\n\n"
                f"To fix:\n"
                f"  Add an '{agent_key}' section to your config:\n\n"
                f"  {agent_key}:\n"
                f"    type: random  # or 'alphazero' or 'vanilla_mcts'"
            )

        if 'type' not in config_dict[agent_key]:
            raise ValueError(
                f"Problem: Missing required field '{agent_key}.type'\n\n"
                f"Details:\n"
                f"  Agent configuration must specify the agent type.\n\n"
                f"To fix:\n"
                f"  Add a 'type' field under '{agent_key}':\n\n"
                f"  {agent_key}:\n"
                f"    type: random  # Options: 'random', 'alphazero', or 'vanilla_mcts'"
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
        available_games = GameRegistry.list_games()
        raise ValueError(
            f"Problem: Unknown game '{game_name}'\n\n"
            f"Available games: {', '.join(available_games)}\n\n"
            "To fix:\n"
            f"  Set 'game' to one of the available game names:\n\n"
            f"  game: {available_games[0] if available_games else 'your_game'}"
        )


def validate_agent_type(agent_type: str, agent_num: int) -> None:
    """
    Validate that agent type is valid.

    Args:
        agent_type: Agent type string
        agent_num: Agent number (1 or 2)

    Raises:
        ValueError: If agent type is invalid
    """
    valid_types = AlgorithmRegistry.get_all_algorithms()
    if agent_type not in valid_types:
        # Build description of each algorithm type
        type_descriptions = []
        for algo in valid_types:
            metadata = AlgorithmRegistry.get_metadata(algo)
            if metadata.requires_checkpoint:
                type_descriptions.append(f"  - {algo}: requires checkpoint_dir")
            else:
                type_descriptions.append(f"  - {algo}: no checkpoint needed")

        raise ValueError(
            f"Problem: Invalid agent{agent_num} type '{agent_type}'\n\n"
            f"Valid types:\n"
            f"{chr(10).join(type_descriptions)}\n\n"
            "To fix:\n"
            f"  Set agent{agent_num}.type to one of the valid types:\n\n"
            f"  agent{agent_num}:\n"
            f"    type: {valid_types[0] if valid_types else 'your_algorithm'}"
        )


def validate_checkpoint_exists(checkpoint_dir: Path, agent_num: int, agent_type: str) -> None:
    """
    Validate that checkpoint directory and required files exist.

    Args:
        checkpoint_dir: Path to checkpoint directory
        agent_num: Agent number (1 or 2)
        agent_type: Agent type (e.g., 'alphazero', 'vanilla_mcts')

    Raises:
        ValueError: If checkpoint directory or files don't exist
    """
    if not checkpoint_dir.exists():
        raise ValueError(
            f"Problem: Agent{agent_num} checkpoint directory not found\n\n"
            f"Details:\n"
            f"  Path: {checkpoint_dir}\n\n"
            "This usually means:\n"
            "  1. You haven't trained an agent yet, or\n"
            "  2. The checkpoint path in your config is incorrect\n\n"
            "To fix:\n"
            "  Option 1: Train an agent first:\n"
            "    python scripts/train.py scripts/train.yaml\n\n"
            "  Option 2: Update your config with the correct checkpoint path:\n"
            f"    agent{agent_num}:\n"
            f"      checkpoint_dir: saved_agents/[your_checkpoint_dir]\n\n"
            "  Option 3: Use a random agent instead:\n"
            f"    agent{agent_num}:\n"
            f"      type: random"
        )

    # Get required checkpoint files from algorithm metadata
    metadata = AlgorithmRegistry.get_metadata(agent_type)
    for required_file in metadata.checkpoint_files:
        file_path = checkpoint_dir / required_file
        if not file_path.exists():
            raise ValueError(
                f"Problem: Agent{agent_num} checkpoint missing {required_file}\n\n"
                f"Details:\n"
                f"  Path: {checkpoint_dir}\n"
                f"  Missing: {required_file}\n\n"
                "To fix:\n"
                f"  Ensure this is a valid {agent_type} checkpoint directory.\n"
                f"  A valid {agent_type} checkpoint should contain:\n"
                f"    {', '.join(metadata.checkpoint_files)}"
            )


def load_match_config(config_path: Path) -> MatchConfig:
    """
    Load and validate match configuration from YAML.

    Args:
        config_path: Path to YAML config file

    Returns:
        MatchConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Problem: Configuration file not found\n\n"
            f"Details:\n"
            f"  Path: {config_path}\n\n"
            "To fix:\n"
            "  Create the config file at scripts/simulate_match.yaml\n\n"
            "Example config:\n"
            "  game: tictactoe\n"
            "  match:\n"
            "    num_games: 100\n"
            "  agent1:\n"
            "    type: random\n"
            "  agent2:\n"
            "    type: random"
        )

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

    validate_config_schema(config_dict)

    game_name = config_dict['game']
    num_games = config_dict['match']['num_games']

    validate_game_exists(game_name)

    if not isinstance(num_games, int) or num_games <= 0:
        raise ValueError(
            f"Problem: Invalid num_games value '{num_games}'\n\n"
            "Details:\n"
            "  num_games must be a positive integer.\n\n"
            "To fix:\n"
            "  match:\n"
            "    num_games: 100"
        )

    # Process both agents
    agent_configs = []
    for agent_num in [1, 2]:
        agent_key = f'agent{agent_num}'
        agent_dict = config_dict[agent_key]
        agent_type = agent_dict['type']

        validate_agent_type(agent_type, agent_num)

        checkpoint_dir = None
        metadata = AlgorithmRegistry.get_metadata(agent_type)
        if metadata.requires_checkpoint:
            checkpoint_dir_str = agent_dict.get('checkpoint_dir')
            if not checkpoint_dir_str:
                raise ValueError(
                    f"Problem: Missing checkpoint_dir for agent{agent_num}\n\n"
                    "Details:\n"
                    f"  {agent_type} agents require a checkpoint directory.\n\n"
                    "To fix:\n"
                    f"  Add checkpoint_dir to agent{agent_num} config:\n\n"
                    f"  agent{agent_num}:\n"
                    f"    type: {agent_type}\n"
                    f"    checkpoint_dir: saved_agents/YOUR_CHECKPOINT"
                )

            checkpoint_dir = Path(checkpoint_dir_str)
            validate_checkpoint_exists(checkpoint_dir, agent_num, agent_type)

        agent_configs.append((agent_type, checkpoint_dir))

    return MatchConfig(
        game_name=game_name,
        num_games=num_games,
        agent1_type=agent_configs[0][0],
        agent1_checkpoint_dir=agent_configs[0][1],
        agent2_type=agent_configs[1][0],
        agent2_checkpoint_dir=agent_configs[1][1]
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


def create_agent(agent_type: str, checkpoint_dir: Optional[Path], game: Game) -> Agent:
    """
    Create agent based on type.

    Args:
        agent_type: Agent type (e.g., 'alphazero', 'vanilla_mcts', 'random')
        checkpoint_dir: Path to checkpoint (required for algorithms that need it)
        game: Game instance

    Returns:
        Agent: Configured agent ready to play
    """
    metadata = AlgorithmRegistry.get_metadata(agent_type)
    if metadata.requires_checkpoint:
        return load_agent_checkpoint(checkpoint_dir)
    else:
        # No checkpoint required = random agent
        return RandomAgent(game=game)


def main() -> None:
    """Main entry point."""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / "simulate_match.yaml"

        config = load_match_config(config_path)

        game = create_game(config.game_name)

        agent1 = create_agent(config.agent1_type, config.agent1_checkpoint_dir, game)
        agent2 = create_agent(config.agent2_type, config.agent2_checkpoint_dir, game)

        print(f"\n{'='*60}")
        print(f"Match Simulation: {config.game_name.upper()}")
        print(f"Agent 1: {agent1.__class__.__name__}")
        print(f"Agent 2: {agent2.__class__.__name__}")
        print(f"Games: {config.num_games}")
        print(f"{'='*60}\n")

        simulate_match(game, agent1, agent2, config.num_games)

    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()
