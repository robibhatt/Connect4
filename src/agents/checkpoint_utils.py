"""
Checkpoint utilities for saving and loading agents.

Provides high-level functions for agent persistence.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

import yaml

from src.agents.checkpointable import CheckpointableAgent
from src.agents.registry import AgentRegistry
from src.agents.agent import Agent
from src.games.core.registry import GameRegistry

if TYPE_CHECKING:
    from src.algorithms.vanilla_mcts import VanillaMCTSAgentConfig


def save_agent_checkpoint(
    agent: CheckpointableAgent,
    agent_class_name: str,
    game_name: str,
    config: VanillaMCTSAgentConfig,
    training_config: Optional[Dict] = None,
    root_dir: str = "saved_agents",
    custom_folder_name: Optional[str] = None
) -> Path:
    """
    Save agent checkpoint with metadata.

    Creates directory: {root_dir}/{custom_folder_name} if provided,
    otherwise {root_dir}/{timestamp}_{game}_{AgentClass}/
    Saves files:
        - model.pt: Model weights (via agent.to_checkpoint())
        - agent.yaml: Complete agent configuration

    Args:
        agent: Agent to save (must implement CheckpointableAgent)
        agent_class_name: Agent class name (e.g., 'TicTacToeAlphaZeroAgent')
        game_name: Game identifier (e.g., 'tictactoe')
        config: Agent configuration
        training_config: Optional training metadata
        root_dir: Root directory for saved agents
        custom_folder_name: Optional custom name for the checkpoint folder.
            If None or empty, uses auto-generated timestamped name.

    Returns:
        Path to saved agent directory

    Raises:
        FileExistsError: If custom_folder_name is provided and directory exists
        ValueError: If custom_folder_name contains path separators
    """
    # Determine folder name
    if custom_folder_name and custom_folder_name.strip():
        folder_name = custom_folder_name.strip()
        # Validate: no path separators allowed
        if '/' in folder_name or '\\' in folder_name:
            raise ValueError(
                f"custom_folder_name cannot contain path separator: '{folder_name}'"
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{game_name}_{agent_class_name}"

    save_dir = Path(root_dir) / folder_name

    # Check for existing directory with helpful error message
    if save_dir.exists():
        raise FileExistsError(
            f"Checkpoint directory already exists: '{save_dir}'. "
            "Choose a different custom_folder_name or remove existing directory."
        )

    save_dir.mkdir(parents=True, exist_ok=False)

    # Delegate to agent's checkpoint method (saves model.pt)
    agent.to_checkpoint(save_dir)

    # Build agent.yaml conditionally based on config type
    agent_yaml = {
        'agent_class': agent_class_name,
        'game': game_name,
        'timestamp': timestamp,
        'device': config.device,
    }

    # Add model section only if config has model fields
    if hasattr(config, 'model_class') and hasattr(config, 'model_kwargs'):
        agent_yaml['model'] = {
            'class': config.model_class,
            'kwargs': config.model_kwargs
        }

    # Build MCTS section
    mcts_config = {}
    # Common fields
    if hasattr(config, 'num_sims'):
        mcts_config['num_sims'] = config.num_sims
    if hasattr(config, 'illegal_action_penalty'):
        mcts_config['illegal_action_penalty'] = config.illegal_action_penalty
    # Vanilla MCTS fields
    if hasattr(config, 'c_exploration'):
        mcts_config['c_exploration'] = config.c_exploration
    if hasattr(config, 'max_rollout_depth'):
        mcts_config['max_rollout_depth'] = config.max_rollout_depth
    if hasattr(config, 'rollout_seed'):
        mcts_config['rollout_seed'] = config.rollout_seed

    agent_yaml['mcts'] = mcts_config

    if training_config:
        agent_yaml['training'] = training_config

    # Save agent.yaml
    with (save_dir / "agent.yaml").open('w') as f:
        yaml.dump(agent_yaml, f, default_flow_style=False)

    return save_dir


def load_agent_checkpoint(checkpoint_dir: Path | str) -> Agent:
    """
    Load agent from checkpoint directory.

    Auto-detects agent class and game from agent.yaml.

    Args:
        checkpoint_dir: Path to saved agent directory

    Returns:
        Loaded agent ready to play

    Raises:
        FileNotFoundError: If checkpoint directory or agent.yaml not found
        KeyError: If agent class not registered
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    agent_yaml_path = checkpoint_dir / "agent.yaml"
    if not agent_yaml_path.exists():
        raise FileNotFoundError(f"agent.yaml not found in {checkpoint_dir}")

    # Load agent.yaml
    with agent_yaml_path.open('r') as f:
        agent_yaml = yaml.safe_load(f)

    agent_class_name = agent_yaml['agent_class']
    game_name = agent_yaml['game']

    # Get agent class from registry
    AgentClass = AgentRegistry.get_agent(agent_class_name)

    # Get game from registry
    GameClass = GameRegistry.get_game(game_name)
    game = GameClass()

    # Use agent's class method to reconstruct
    # Device can be overridden from agent.yaml
    device = agent_yaml.get('device', 'cpu')
    return AgentClass.from_checkpoint(checkpoint_dir, game, device=device)
