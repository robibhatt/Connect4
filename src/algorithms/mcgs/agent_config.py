"""
Configuration for MCGS agents.

Uses composition: embeds MCGSCoreConfig rather than duplicating fields.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

from src.agents.config import AgentConfig
from src.algorithms.mcgs.mcgs import MCGSCoreConfig


@dataclass
class MCGSAgentConfig(AgentConfig):
    """
    Configuration for MCGS agents using composition.

    Embeds MCGSCoreConfig to avoid field duplication. Adding a new parameter
    to MCGSCoreConfig automatically makes it available here.
    """

    mcgs: MCGSCoreConfig
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to nested dict structure for YAML."""
        return {
            "mcgs": asdict(self.mcgs),
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCGSAgentConfig:
        """Deserialize from nested dict structure."""
        mcgs_data = data.get("mcgs", {})
        mcgs = MCGSCoreConfig(**mcgs_data)
        device = data.get("device", "cpu")
        return cls(mcgs=mcgs, device=device)
