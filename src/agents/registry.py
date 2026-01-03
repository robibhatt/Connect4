"""
Agent Registry for dynamic agent discovery and loading.

Provides centralized mapping between agent class names and agent classes,
following the same pattern as ModelRegistry.
"""

from __future__ import annotations
from typing import Type, Dict, Set
import re

from src.agents.agent import Agent


class AgentRegistry:
    """
    SINGLE centralized registry for ALL agents from ALL games.

    Supports multiple agent types per game (e.g., TicTacToeAlphaZeroAgent, TicTacToeRandomAgent).

    Naming convention: {Game}{Algorithm}Agent
    Examples:
        - TicTacToeAlphaZeroAgent -> game: tictactoe, algorithm: AlphaZero
        - Connect4AlphaZeroAgent -> game: connect4, algorithm: AlphaZero

    Usage:
        # Register an agent (auto-extracts game name from class name)
        AgentRegistry.register(TicTacToeAlphaZeroAgent)

        # Get agent class by full class name
        agent_cls = AgentRegistry.get_agent('TicTacToeAlphaZeroAgent')
        agent = agent_cls(game=game, mcts=mcts)

        # Get all agents for a game
        agents = AgentRegistry.get_agents_for_game('tictactoe')

        # List all registered agents
        all_agents = AgentRegistry.list_agents()
    """

    _registry: Dict[str, Type[Agent]] = {}  # agent_class_name -> agent_class
    _game_to_agents: Dict[str, Set[str]] = {}  # game_name -> set of agent_class_names

    @classmethod
    def register(cls, agent_class: Type[Agent]) -> None:
        """
        Register an agent class (auto-extracts game name from class name).

        Expects agent class names in format: {Game}{Algorithm}Agent
        E.g., TicTacToeAlphaZeroAgent -> extracts 'tictactoe'

        Args:
            agent_class: Agent class to register

        Raises:
            ValueError: If agent already registered or game name cannot be extracted
        """
        agent_class_name = agent_class.__name__

        # Extract game name from class name
        # Pattern: {Game}{Algorithm}Agent -> extract {Game}
        # E.g., TicTacToeAlphaZeroAgent -> TicTacToe
        # Game can contain digits (Connect4), Algorithm is mixed case (AlphaZero, Random, etc.)
        match = re.match(
            r'^([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)*)([A-Z][A-Za-z0-9]+)Agent$',
            agent_class_name
        )
        if not match:
            raise ValueError(
                f"Agent class name '{agent_class_name}' does not follow pattern "
                f"{{Game}}{{Algorithm}}Agent (e.g., TicTacToeAlphaZeroAgent)"
            )

        game_name_camel = match.group(1)  # E.g., "TicTacToe" or "Connect4"
        game_name = game_name_camel.lower()  # E.g., "tictactoe" or "connect4"

        # Register in main registry
        if agent_class_name in cls._registry:
            if cls._registry[agent_class_name] != agent_class:
                raise ValueError(
                    f"Agent '{agent_class_name}' already registered with "
                    f"{cls._registry[agent_class_name]}, "
                    f"cannot register {agent_class}"
                )
            # Already registered with same class, silently succeed
            return

        cls._registry[agent_class_name] = agent_class

        # Track in game-to-agents mapping
        if game_name not in cls._game_to_agents:
            cls._game_to_agents[game_name] = set()
        cls._game_to_agents[game_name].add(agent_class_name)

    @classmethod
    def get_agent(cls, agent_class_name: str) -> Type[Agent]:
        """
        Get the agent class by full class name.

        Args:
            agent_class_name: Full agent class name (e.g., 'TicTacToeAlphaZeroAgent')

        Returns:
            Agent class

        Raises:
            KeyError: If agent not registered
        """
        # Lazy registration - try to register on first access
        if agent_class_name not in cls._registry:
            cls._ensure_registered_by_class_name(agent_class_name)

        if agent_class_name not in cls._registry:
            raise KeyError(
                f"No agent registered with name '{agent_class_name}'. "
                f"Available agents: {list(cls._registry.keys())}"
            )
        return cls._registry[agent_class_name]

    @classmethod
    def get_agents_for_game(cls, game_name: str) -> list[str]:
        """
        Get all agent class names registered for a game.

        Args:
            game_name: Game identifier (e.g., 'tictactoe')

        Returns:
            List of agent class names (e.g., ['TicTacToeAlphaZeroAgent', 'TicTacToeRandomAgent'])
        """
        # Lazy registration attempt
        cls._ensure_registered(game_name)

        if game_name not in cls._game_to_agents:
            return []
        return sorted(cls._game_to_agents[game_name])

    @classmethod
    def _ensure_registered(cls, game_name: str) -> None:
        """Ensure a game's agents are registered (lazy registration by game name)."""
        # No game-specific agents to register anymore
        # AlphaZeroAgent is generic and works with all games
        pass

    @classmethod
    def _ensure_registered_by_class_name(cls, agent_class_name: str) -> None:
        """Ensure an agent is registered (lazy registration by agent class name)."""
        if agent_class_name == 'AlphaZeroAgent':
            try:
                from src.algorithms.alphazero import AlphaZeroAgent
                if 'AlphaZeroAgent' not in cls._registry:
                    cls._registry['AlphaZeroAgent'] = AlphaZeroAgent
            except (ImportError, ValueError):
                pass

    @classmethod
    def list_agents(cls) -> list[str]:
        """List all registered agent class names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_games(cls) -> list[str]:
        """List all games that have registered agents."""
        return sorted(cls._game_to_agents.keys())


# Auto-register known agents
def _auto_register():
    """Automatically register all known agents."""
    # Register the generic AlphaZeroAgent
    try:
        from src.algorithms.alphazero import AlphaZeroAgent
        if 'AlphaZeroAgent' not in AgentRegistry._registry:
            AgentRegistry._registry['AlphaZeroAgent'] = AlphaZeroAgent
    except ImportError:
        pass


_auto_register()
