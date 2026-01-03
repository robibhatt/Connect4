"""
View agent vs agent match with manual stepping controls.

Loads configuration from view_match.yaml and launches a pygame UI where you can
watch two AI agents play against each other. Press SPACE to advance each move.
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
from src.games.core.registry import GameRegistry
from src.games.core.ui.game_ui import UIConfig
from src.games.tictactoe.ui.tictactoe_match_ui import TicTacToeMatchUI
from src.games.connect4.ui.connect4_match_ui import Connect4MatchUI


@dataclass
class ViewMatchConfig:
    """Validated view_match configuration."""
    game_name: str
    agent1_type: str
    agent1_name: str
    agent1_checkpoint_dir: Optional[Path]
    agent2_type: str
    agent2_name: str
    agent2_checkpoint_dir: Optional[Path]
    pause_after_move: float
    window_size: int


def print_error(message: str) -> None:
    """Print formatted error message."""
    print("=" * 60)
    print("Configuration Error")
    print("=" * 60)
    print()
    print(message)
    print()
    print("=" * 60)


def load_view_match_config(config_path: Path) -> ViewMatchConfig:
    """
    Load and validate view_match config from YAML.

    Validates:
    - File exists
    - YAML syntax
    - Required fields present
    - Game name valid
    - Agent types valid ('random' or 'alphazero')
    - Checkpoint dirs exist (for alphazero agents)

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated ViewMatchConfig

    Raises:
        ValueError: For any validation error
    """
    # 1. Check file exists
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    # 2. Load YAML
    with config_path.open('r') as f:
        try:
            config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")

    # 3. Validate schema (required fields)
    if 'game' not in config_dict:
        raise ValueError("Missing required field: 'game'")
    if 'agent1' not in config_dict:
        raise ValueError("Missing required field: 'agent1'")
    if 'agent2' not in config_dict:
        raise ValueError("Missing required field: 'agent2'")

    game_name = config_dict['game']

    # 4. Validate game exists
    try:
        GameRegistry.get_game(game_name)
    except (ValueError, KeyError):
        raise ValueError(
            f"Invalid game: '{game_name}'\n"
            f"Valid games: {GameRegistry.list_games()}"
        )

    # 5. Extract agent configs
    agent1_dict = config_dict['agent1']
    agent2_dict = config_dict['agent2']

    # Agent 1
    agent1_type = agent1_dict.get('type')
    if not agent1_type:
        raise ValueError("Missing 'type' in agent1 config")
    if agent1_type not in ['random', 'alphazero']:
        raise ValueError(
            f"Invalid agent1 type: '{agent1_type}'. "
            "Must be 'random' or 'alphazero'"
        )

    agent1_name = agent1_dict.get('name', agent1_type.capitalize())

    agent1_checkpoint_dir = None
    if agent1_type == 'alphazero':
        checkpoint_str = agent1_dict.get('checkpoint_dir')
        if not checkpoint_str:
            raise ValueError("Missing 'checkpoint_dir' for agent1 (alphazero)")
        agent1_checkpoint_dir = Path(checkpoint_str)
        if not agent1_checkpoint_dir.exists():
            raise ValueError(
                f"agent1 checkpoint directory not found: {agent1_checkpoint_dir}"
            )

    # Agent 2
    agent2_type = agent2_dict.get('type')
    if not agent2_type:
        raise ValueError("Missing 'type' in agent2 config")
    if agent2_type not in ['random', 'alphazero']:
        raise ValueError(
            f"Invalid agent2 type: '{agent2_type}'. "
            "Must be 'random' or 'alphazero'"
        )

    agent2_name = agent2_dict.get('name', agent2_type.capitalize())

    agent2_checkpoint_dir = None
    if agent2_type == 'alphazero':
        checkpoint_str = agent2_dict.get('checkpoint_dir')
        if not checkpoint_str:
            raise ValueError("Missing 'checkpoint_dir' for agent2 (alphazero)")
        agent2_checkpoint_dir = Path(checkpoint_str)
        if not agent2_checkpoint_dir.exists():
            raise ValueError(
                f"agent2 checkpoint directory not found: {agent2_checkpoint_dir}"
            )

    # 6. Extract UI config (with defaults)
    ui_dict = config_dict.get('ui', {})
    pause_after_move = ui_dict.get('pause_after_move', 0.3)
    window_size = ui_dict.get('window_size', 800)

    return ViewMatchConfig(
        game_name=game_name,
        agent1_type=agent1_type,
        agent1_name=agent1_name,
        agent1_checkpoint_dir=agent1_checkpoint_dir,
        agent2_type=agent2_type,
        agent2_name=agent2_name,
        agent2_checkpoint_dir=agent2_checkpoint_dir,
        pause_after_move=pause_after_move,
        window_size=window_size,
    )


def create_agent(agent_type: str, checkpoint_dir: Optional[Path], game: Game) -> Agent:
    """
    Create agent from type.

    Args:
        agent_type: 'random' or 'alphazero'
        checkpoint_dir: Path to checkpoint (required for alphazero)
        game: Game instance

    Returns:
        Agent instance

    Raises:
        ValueError: If alphazero without checkpoint or checkpoint invalid
    """
    if agent_type == 'random':
        return RandomAgent(game=game)
    elif agent_type == 'alphazero':
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir required for alphazero agent")
        # load_agent_checkpoint will validate the checkpoint directory exists
        # and will load the game from the checkpoint config
        try:
            return load_agent_checkpoint(checkpoint_dir)
        except FileNotFoundError as e:
            # Convert to ValueError for consistent error handling
            raise ValueError(str(e))
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_match_ui(game: Game, agent1: Agent, agent2: Agent, config: ViewMatchConfig):
    """
    Create appropriate match UI for game type.

    Args:
        game: Game instance
        agent1: First agent
        agent2: Second agent
        config: View match configuration

    Returns:
        Match UI instance (TicTacToeMatchUI or Connect4MatchUI)

    Raises:
        ValueError: If no UI available for game type
    """
    ui_config = UIConfig(window_size=config.window_size)

    game_name = game.__class__.__name__.lower()

    if game_name == 'tictactoe':
        return TicTacToeMatchUI(
            game=game,
            agent1=agent1,
            agent2=agent2,
            agent1_name=config.agent1_name,
            agent2_name=config.agent2_name,
            pause_after_move=config.pause_after_move,
            cfg=ui_config,
        )
    elif game_name == 'connect4':
        return Connect4MatchUI(
            game=game,
            agent1=agent1,
            agent2=agent2,
            agent1_name=config.agent1_name,
            agent2_name=config.agent2_name,
            pause_after_move=config.pause_after_move,
            cfg=ui_config,
        )
    else:
        raise ValueError(f"No match UI available for game: {game_name}")


def main() -> None:
    """Main entry point for view_match script."""
    try:
        # 1. Load config
        script_dir = Path(__file__).parent
        config_path = script_dir / "view_match.yaml"
        config = load_view_match_config(config_path)

        # 2. Create game
        GameClass = GameRegistry.get_game(config.game_name)
        game = GameClass()

        # 3. Create agents
        agent1 = create_agent(config.agent1_type, config.agent1_checkpoint_dir, game)
        agent2 = create_agent(config.agent2_type, config.agent2_checkpoint_dir, game)

        # 4. Print header
        print(f"\n{'='*60}")
        print(f"Agent Match Viewer: {config.game_name.upper()}")
        print(f"Agent 1: {config.agent1_name}")
        print(f"Agent 2: {config.agent2_name}")
        print(f"{'='*60}\n")
        print("Controls:")
        print("  SPACE - Advance to next move / Start next game")
        print("  N     - Start new game (keeps score)")
        print("  Q/ESC - Quit")
        print(f"\n{'='*60}\n")

        # 5. Create and launch UI
        ui = create_match_ui(game, agent1, agent2, config)
        ui.run()

    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == '__main__':
    main()
