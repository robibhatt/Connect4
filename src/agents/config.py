"""
Base configuration classes for agents.

Provides abstract interface for agent checkpoint serialization.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class AgentConfig(ABC):
    """
    Base configuration for all agents.

    Provides checkpoint serialization interface for saving/loading agent configs.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize config to dictionary for saving.

        Returns:
            Dictionary representation of config
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        """
        Deserialize config from dictionary.

        Args:
            data: Dictionary representation of config

        Returns:
            Config instance
        """
        pass
