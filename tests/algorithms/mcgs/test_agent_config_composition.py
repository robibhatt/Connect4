"""Tests for MCGSAgentConfig composition structure."""

import pytest
import yaml

from src.algorithms.mcgs.agent_config import MCGSAgentConfig
from src.algorithms.mcgs.mcgs import MCGSCoreConfig


class TestMCGSAgentConfigComposition:
    """Test that MCGSAgentConfig uses composition with embedded MCGSCoreConfig."""

    def test_agent_config_embeds_mcgs_core_config(self):
        """MCGSAgentConfig should embed MCGSCoreConfig."""
        core = MCGSCoreConfig(num_sims=500)
        config = MCGSAgentConfig(mcgs=core)
        assert hasattr(config, "mcgs")
        assert isinstance(config.mcgs, MCGSCoreConfig)
        assert config.mcgs.num_sims == 500

    def test_agent_config_has_device_field(self):
        """MCGSAgentConfig should have device field."""
        core = MCGSCoreConfig()
        config = MCGSAgentConfig(mcgs=core, device="cuda")
        assert config.device == "cuda"

    def test_agent_config_device_default(self):
        """MCGSAgentConfig device should default to cpu."""
        core = MCGSCoreConfig()
        config = MCGSAgentConfig(mcgs=core)
        assert config.device == "cpu"

    def test_nested_field_access(self):
        """Can access mcgs config fields via nested path."""
        core = MCGSCoreConfig(
            num_sims=100,
            c_exploration=2.0,
            max_rollout_depth=50,
            rollout_seed=42,
            illegal_action_penalty=1e6,
            batch_size=8,
        )
        config = MCGSAgentConfig(mcgs=core)

        assert config.mcgs.num_sims == 100
        assert config.mcgs.c_exploration == 2.0
        assert config.mcgs.max_rollout_depth == 50
        assert config.mcgs.rollout_seed == 42
        assert config.mcgs.illegal_action_penalty == 1e6
        assert config.mcgs.batch_size == 8

    def test_yaml_roundtrip(self):
        """Config should survive YAML serialization roundtrip."""
        core = MCGSCoreConfig(num_sims=888, c_exploration=2.1)
        original = MCGSAgentConfig(mcgs=core, device="cpu")

        yaml_str = yaml.dump(original.to_dict())
        loaded_dict = yaml.safe_load(yaml_str)
        restored = MCGSAgentConfig.from_dict(loaded_dict)

        assert restored.mcgs.num_sims == original.mcgs.num_sims
        assert restored.mcgs.c_exploration == original.mcgs.c_exploration
        assert restored.device == original.device
