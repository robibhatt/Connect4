"""
Algorithm Registry for training.

Provides centralized mapping of algorithm names to their config classes,
trainer factories, and agent config factories, enabling dynamic algorithm loading.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Type, Dict, Callable, Tuple, Any, Optional
import torch.nn as nn

from src.games.core.game import Game


@dataclass
class AlgorithmMetadata:
    """
    Metadata describing algorithm capabilities and requirements.

    Attributes:
        requires_model: Whether this algorithm needs a neural network model
        requires_checkpoint: Whether trained agents need checkpoint directories
        checkpoint_files: Tuple of required files in checkpoint directory
    """
    requires_model: bool = True
    requires_checkpoint: bool = True
    checkpoint_files: tuple[str, ...] = ('agent.yaml',)


class AlgorithmRegistry:
    """
    Registry mapping algorithm names to (ConfigClass, TrainerFactory, AgentConfigFactory).

    Each algorithm provides:
    - Config class: Dataclass defining all algorithm parameters
    - Trainer factory: Function that creates a trainer instance
    - Agent config factory: Function that creates agent config from training config

    Usage:
        # Get config class
        ConfigClass = AlgorithmRegistry.get_config_class('alphazero')
        config = ConfigClass(**yaml_dict)

        # Get trainer factory
        factory = AlgorithmRegistry.get_trainer_factory('alphazero')
        trainer = factory(game, model, config)

        # Get agent config factory
        agent_config_factory = AlgorithmRegistry.get_agent_config_factory('alphazero')
        agent_config = agent_config_factory(config)

        # Register new algorithm
        AlgorithmRegistry.register(
            'my_algo',
            MyAlgoConfig,
            my_algo_trainer_factory,
            my_algo_agent_config_factory
        )

        # List all algorithms
        algos = AlgorithmRegistry.list_algorithms()
    """

    # Registry maps: algorithm_name -> (ConfigClass, TrainerFactory)
    _registry: Dict[str, Tuple[Type, Callable]] = {}

    # Agent config factories: algorithm_name -> AgentConfigFactory
    _agent_config_factories: Dict[str, Callable] = {}

    # Metadata: algorithm_name -> AlgorithmMetadata
    _metadata: Dict[str, AlgorithmMetadata] = {}

    @classmethod
    def register(
        cls,
        algorithm_name: str,
        config_class: Optional[Type],
        trainer_factory: Optional[Callable[[Game, nn.Module, Any], Any]],
        agent_config_factory: Optional[Callable[[Any], Any]] = None,
        metadata: Optional[AlgorithmMetadata] = None
    ) -> None:
        """
        Register an algorithm.

        Args:
            algorithm_name: Unique algorithm identifier (e.g., 'alphazero')
            config_class: Config dataclass for this algorithm (None for pseudo-algorithms)
            trainer_factory: Factory function (game, model, config) -> trainer (None for pseudo-algorithms)
            agent_config_factory: Factory function (config) -> agent_config (optional)
            metadata: Algorithm metadata describing capabilities (optional)

        Raises:
            ValueError: If algorithm already registered with different class
        """
        if algorithm_name in cls._registry:
            existing_config, existing_factory = cls._registry[algorithm_name]
            if existing_config != config_class or existing_factory != trainer_factory:
                # Get names safely (handles Mock objects in tests)
                existing_config_name = getattr(existing_config, '__name__', repr(existing_config))
                existing_factory_name = getattr(existing_factory, '__name__', repr(existing_factory))
                new_config_name = getattr(config_class, '__name__', repr(config_class))
                new_factory_name = getattr(trainer_factory, '__name__', repr(trainer_factory))

                raise ValueError(
                    f"Algorithm '{algorithm_name}' already registered with "
                    f"({existing_config_name}, {existing_factory_name}), "
                    f"cannot register ({new_config_name}, {new_factory_name})"
                )
            # Already registered with same classes, silently succeed
            # But still update agent_config_factory and metadata if provided
            if agent_config_factory is not None:
                cls._agent_config_factories[algorithm_name] = agent_config_factory
            if metadata is not None:
                cls._metadata[algorithm_name] = metadata
            return

        cls._registry[algorithm_name] = (config_class, trainer_factory)
        if agent_config_factory is not None:
            cls._agent_config_factories[algorithm_name] = agent_config_factory
        if metadata is not None:
            cls._metadata[algorithm_name] = metadata

    @classmethod
    def get_config_class(cls, algorithm_name: str) -> Type:
        """
        Get the config class for an algorithm.

        Args:
            algorithm_name: Algorithm identifier

        Returns:
            Config class (e.g., AlphaZeroConfig)

        Raises:
            KeyError: If algorithm not registered
        """
        # Lazy registration - try to register on first access
        if algorithm_name not in cls._registry:
            cls._ensure_registered(algorithm_name)

        if algorithm_name not in cls._registry:
            raise KeyError(
                f"No algorithm registered for '{algorithm_name}'. "
                f"Available algorithms: {list(cls._registry.keys())}"
            )

        config_class, _ = cls._registry[algorithm_name]
        return config_class

    @classmethod
    def get_trainer_factory(cls, algorithm_name: str) -> Callable:
        """
        Get the trainer factory for an algorithm.

        Args:
            algorithm_name: Algorithm identifier

        Returns:
            Trainer factory function: (game, model, config) -> trainer

        Raises:
            KeyError: If algorithm not registered
        """
        # Lazy registration - try to register on first access
        if algorithm_name not in cls._registry:
            cls._ensure_registered(algorithm_name)

        if algorithm_name not in cls._registry:
            raise KeyError(
                f"No algorithm registered for '{algorithm_name}'. "
                f"Available algorithms: {list(cls._registry.keys())}"
            )

        _, trainer_factory = cls._registry[algorithm_name]
        return trainer_factory

    @classmethod
    def get_agent_config_factory(cls, algorithm_name: str) -> Callable:
        """
        Get the agent config factory for an algorithm.

        Args:
            algorithm_name: Algorithm identifier

        Returns:
            Agent config factory function: (config) -> agent_config

        Raises:
            KeyError: If algorithm not registered or has no agent config factory
        """
        # Lazy registration - try to register on first access
        if algorithm_name not in cls._agent_config_factories:
            cls._ensure_registered(algorithm_name)

        if algorithm_name not in cls._agent_config_factories:
            raise KeyError(
                f"No agent config factory registered for '{algorithm_name}'. "
                f"Available algorithms: {list(cls._agent_config_factories.keys())}"
            )

        return cls._agent_config_factories[algorithm_name]

    @classmethod
    def get_metadata(cls, algorithm_name: str) -> AlgorithmMetadata:
        """
        Get the metadata for an algorithm.

        Args:
            algorithm_name: Algorithm identifier

        Returns:
            AlgorithmMetadata describing algorithm capabilities

        Raises:
            KeyError: If algorithm not registered
        """
        # Lazy registration - try to discover on first access
        if algorithm_name not in cls._metadata:
            cls._discover_algorithms()

        if algorithm_name not in cls._metadata:
            raise KeyError(
                f"No metadata registered for '{algorithm_name}'. "
                f"Available algorithms: {list(cls._metadata.keys())}"
            )

        return cls._metadata[algorithm_name]

    @classmethod
    def get_all_algorithms(cls) -> list[str]:
        """
        Get list of all registered algorithm names.

        Returns:
            Sorted list of algorithm names
        """
        # Ensure all algorithms are discovered
        cls._discover_algorithms()
        return sorted(cls._registry.keys())

    @classmethod
    def _discover_algorithms(cls) -> None:
        """
        Discover and register algorithms by scanning src/algorithms/*/ directories.

        Each algorithm module should have a register_algorithm() function that
        handles its own registration.
        """
        import importlib
        from pathlib import Path

        algorithms_dir = Path(__file__).parent
        for subdir in algorithms_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('_'):
                try:
                    # Import the module
                    module = importlib.import_module(f'src.algorithms.{subdir.name}')
                    # Look for register_algorithm() function
                    if hasattr(module, 'register_algorithm'):
                        module.register_algorithm()
                except ImportError:
                    pass

    @classmethod
    def _ensure_registered(cls, algorithm_name: str) -> None:
        """Ensure an algorithm is registered (lazy registration)."""
        # First try discovery
        cls._discover_algorithms()

        # Fall back to hardcoded registration for backwards compatibility
        if algorithm_name == 'alphazero' and algorithm_name not in cls._registry:
            try:
                from src.algorithms.alphazero.config import AlphaZeroConfig
                from src.algorithms.alphazero.factories import (
                    create_alphazero_trainer,
                    create_alphazero_agent_config
                )
                cls.register(
                    'alphazero',
                    AlphaZeroConfig,
                    create_alphazero_trainer,
                    create_alphazero_agent_config
                )
            except (ImportError, ValueError):
                pass
        elif algorithm_name == 'vanilla_mcts' and algorithm_name not in cls._registry:
            try:
                from src.algorithms.vanilla_mcts.config import VanillaMCTSConfig
                from src.algorithms.vanilla_mcts.factories import (
                    create_vanilla_mcts_trainer,
                    create_vanilla_mcts_agent_config
                )
                cls.register(
                    'vanilla_mcts',
                    VanillaMCTSConfig,
                    create_vanilla_mcts_trainer,
                    create_vanilla_mcts_agent_config
                )
            except (ImportError, ValueError):
                pass

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all registered algorithms."""
        # Use discovery to find all algorithms
        cls._discover_algorithms()
        return sorted(cls._registry.keys())


# Auto-discover and register algorithms on module load
AlgorithmRegistry._discover_algorithms()
