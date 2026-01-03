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
                f"    type: random  # or 'alphazero'"
            )

        if 'type' not in config_dict[agent_key]:
            raise ValueError(
                f"Problem: Missing required field '{agent_key}.type'\n\n"
                f"Details:\n"
                f"  Agent configuration must specify the agent type.\n\n"
                f"To fix:\n"
                f"  Add a 'type' field under '{agent_key}':\n\n"
                f"  {agent_key}:\n"
                f"    type: random  # Options: 'random' or 'alphazero'"
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


def validate_agent_type(agent_type: str, agent_num: int) -> None:
    """
    Validate that agent type is valid.

    Args:
        agent_type: Agent type string
        agent_num: Agent number (1 or 2)

    Raises:
        ValueError: If agent type is invalid
    """
    valid_types = ['alphazero', 'random']
    if agent_type not in valid_types:
        raise ValueError(
            f"Problem: Invalid agent{agent_num} type '{agent_type}'\n\n"
            f"Valid types:\n"
            f"  - alphazero: Trained AlphaZero agent (requires checkpoint_dir)\n"
            f"  - random: Random move agent (no checkpoint needed)\n\n"
            "To fix:\n"
            f"  Set agent{agent_num}.type to 'alphazero' or 'random':\n\n"
            f"  agent{agent_num}:\n"
            f"    type: random  # or 'alphazero'"
        )


def validate_checkpoint_exists(checkpoint_dir: Path, agent_num: int) -> None:
    """
    Validate that checkpoint directory and required files exist.

    Args:
        checkpoint_dir: Path to checkpoint directory
        agent_num: Agent number (1 or 2)

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

    agent_yaml_path = checkpoint_dir / "agent.yaml"
    if not agent_yaml_path.exists():
        raise ValueError(
            f"Problem: Agent{agent_num} checkpoint missing agent.yaml\n\n"
            f"Details:\n"
            f"  Path: {checkpoint_dir}\n"
            f"  Missing: agent.yaml\n\n"
            "To fix:\n"
            "  Ensure this is a valid agent checkpoint directory.\n"
            "  A valid checkpoint should contain:\n"
            "    - agent.yaml (agent configuration)\n"
            "    - model.pt (model weights)"
        )

    model_path = checkpoint_dir / "model.pt"
    if not model_path.exists():
        raise ValueError(
            f"Problem: Agent{agent_num} checkpoint missing model.pt\n\n"
            f"Details:\n"
            f"  Path: {checkpoint_dir}\n"
            f"  Missing: model.pt\n\n"
            "To fix:\n"
            "  Ensure this is a valid agent checkpoint directory.\n"
            "  A valid checkpoint should contain:\n"
            "    - agent.yaml (agent configuration)\n"
            "    - model.pt (model weights)"
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
        if agent_type == 'alphazero':
            checkpoint_dir_str = agent_dict.get('checkpoint_dir')
            if not checkpoint_dir_str:
                raise ValueError(
                    f"Problem: Missing checkpoint_dir for agent{agent_num}\n\n"
                    "Details:\n"
                    "  AlphaZero agents require a checkpoint directory.\n\n"
                    "To fix:\n"
                    f"  Add checkpoint_dir to agent{agent_num} config:\n\n"
                    f"  agent{agent_num}:\n"
                    f"    type: alphazero\n"
                    f"    checkpoint_dir: saved_agents/YOUR_CHECKPOINT"
                )

            checkpoint_dir = Path(checkpoint_dir_str)
            validate_checkpoint_exists(checkpoint_dir, agent_num)

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
        agent_type: Agent type ('alphazero' or 'random')
        checkpoint_dir: Path to checkpoint (required for alphazero)
        game: Game instance

    Returns:
        Agent: Configured agent ready to play
    """
    if agent_type == 'alphazero':
        return load_agent_checkpoint(checkpoint_dir)
    elif agent_type == 'random':
        return RandomAgent(game=game)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


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
