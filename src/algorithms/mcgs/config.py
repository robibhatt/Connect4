"""
Unified configuration for MCGS algorithm.

Uses composition: embeds MCGSCoreConfig and TrainerArgs rather than duplicating fields.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

from src.algorithms.mcgs.mcgs import MCGSCoreConfig
from src.algorithms.shared.trainer_args import TrainerArgs


@dataclass
class MCGSConfig:
    """
    Complete MCGS algorithm configuration using composition.

    Embeds:
    - core: MCGSCoreConfig (5 fields) - MCGS search configuration
    - trainer: TrainerArgs (4 fields) - trainer configuration

    Adding a new parameter to MCGSCoreConfig or TrainerArgs automatically
    makes it available here without any changes to this class.
    """

    core: MCGSCoreConfig = field(default_factory=MCGSCoreConfig)
    trainer: TrainerArgs = field(default_factory=TrainerArgs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to nested dict structure for YAML/JSON."""
        return {
            "core": asdict(self.core),
            "trainer": self.trainer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCGSConfig:
        """Deserialize from nested dict structure."""
        core_data = data.get("core", {})
        trainer_data = data.get("trainer", {})

        # Create sub-configs with defaults for missing fields
        core = MCGSCoreConfig(**core_data)
        trainer = TrainerArgs.from_dict(trainer_data) if trainer_data else TrainerArgs()

        return cls(core=core, trainer=trainer)
