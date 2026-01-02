"""
Algorithm Registry for AlphaZero training.

Provides centralized mapping of algorithm names to their config classes
and trainer factories, enabling dynamic algorithm loading.
"""

from __future__ import annotations
from typing import Type, Dict, Callable, Tuple, Any
import torch.nn as nn

from src.games.core.game import Game


class AlgorithmRegistry:
    """
    Registry mapping algorithm names to (ConfigClass, TrainerFactory).

    Each algorithm provides:
    - Config class: Dataclass defining all algorithm parameters
    - Trainer factory: Function that creates a trainer instance

    Usage:
        # Get config class
        ConfigClass = AlgorithmRegistry.get_config_class('alphazero')
        config = ConfigClass(**yaml_dict)

        # Get trainer factory
        factory = AlgorithmRegistry.get_trainer_factory('alphazero')
        trainer = factory(game, model, config)

        # Register new algorithm
        AlgorithmRegistry.register(
            'my_algo',
            MyAlgoConfig,
            my_algo_factory
        )

        # List all algorithms
        algos = AlgorithmRegistry.list_algorithms()
    """

    # Registry maps: algorithm_name -> (ConfigClass, TrainerFactory)
    _registry: Dict[str, Tuple[Type, Callable]] = {}

    @classmethod
    def register(
        cls,
        algorithm_name: str,
        config_class: Type,
        trainer_factory: Callable[[Game, nn.Module, Any], Any]
    ) -> None:
        """
        Register an algorithm.

        Args:
            algorithm_name: Unique algorithm identifier (e.g., 'alphazero')
            config_class: Config dataclass for this algorithm
            trainer_factory: Factory function (game, model, config) -> trainer

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
            return

        cls._registry[algorithm_name] = (config_class, trainer_factory)

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
    def _ensure_registered(cls, algorithm_name: str) -> None:
        """Ensure an algorithm is registered (lazy registration)."""
        if algorithm_name == 'alphazero' and algorithm_name not in cls._registry:
            try:
                from src.algorithms.alphazero.config import AlphaZeroConfig
                from src.algorithms.alphazero.factories import create_alphazero_trainer
                cls.register('alphazero', AlphaZeroConfig, create_alphazero_trainer)
            except (ImportError, ValueError):
                pass

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all registered algorithms."""
        # Ensure known algorithms are registered
        cls._ensure_registered('alphazero')
        return sorted(cls._registry.keys())


# Auto-register known algorithms
def _auto_register():
    """Automatically register all known algorithms."""
    try:
        from src.algorithms.alphazero.config import AlphaZeroConfig
        from src.algorithms.alphazero.factories import create_alphazero_trainer
        AlgorithmRegistry.register('alphazero', AlphaZeroConfig, create_alphazero_trainer)
    except ImportError:
        pass


_auto_register()
