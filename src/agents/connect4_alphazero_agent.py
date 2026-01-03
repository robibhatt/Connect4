"""DEPRECATED: Use AlphaZeroAgent instead."""

import warnings
from src.agents.alphazero_agent import AlphaZeroAgent


class Connect4AlphaZeroAgent(AlphaZeroAgent):
    """
    DEPRECATED: Use AlphaZeroAgent instead.

    Maintained for backward compatibility with existing checkpoints.
    Will be removed in a future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Connect4AlphaZeroAgent is deprecated. "
            "Use AlphaZeroAgent instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
