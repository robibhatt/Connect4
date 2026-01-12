"""Tests for VanillaMCTSAgentConfig composition structure."""

import pytest
import yaml

from src.algorithms.vanilla_mcts.agent_config import VanillaMCTSAgentConfig
from src.algorithms.vanilla_mcts.mcts import MCTSConfig


class TestVanillaMCTSAgentConfigComposition:
    """Test that VanillaMCTSAgentConfig uses composition with embedded MCTSConfig."""

    def test_agent_config_embeds_mcts_config(self):
        """VanillaMCTSAgentConfig should embed MCTSConfig."""
        core = MCTSConfig(num_sims=500)
        config = VanillaMCTSAgentConfig(mcts=core)
        assert hasattr(config, "mcts")
        assert isinstance(config.mcts, MCTSConfig)
        assert config.mcts.num_sims == 500

    def test_agent_config_has_device_field(self):
        """VanillaMCTSAgentConfig should have device field."""
        core = MCTSConfig()
        config = VanillaMCTSAgentConfig(mcts=core, device="cuda")
        assert config.device == "cuda"

    def test_agent_config_device_default(self):
        """VanillaMCTSAgentConfig device should default to cpu."""
        core = MCTSConfig()
        config = VanillaMCTSAgentConfig(mcts=core)
        assert config.device == "cpu"

    def test_nested_field_access(self):
        """Can access mcts config fields via nested path."""
        core = MCTSConfig(
            num_sims=100,
            c_exploration=2.0,
            max_rollout_depth=50,
            rollout_seed=42,
            illegal_action_penalty=1e6,
        )
        config = VanillaMCTSAgentConfig(mcts=core)

        assert config.mcts.num_sims == 100
        assert config.mcts.c_exploration == 2.0
        assert config.mcts.max_rollout_depth == 50
        assert config.mcts.rollout_seed == 42
        assert config.mcts.illegal_action_penalty == 1e6


class TestVanillaMCTSAgentConfigSerialization:
    """Test VanillaMCTSAgentConfig serialization to/from dict."""

    def test_to_dict_nested_structure(self):
        """to_dict should produce nested structure."""
        core = MCTSConfig(num_sims=200, c_exploration=1.5)
        config = VanillaMCTSAgentConfig(mcts=core, device="mps")
        d = config.to_dict()

        assert "mcts" in d
        assert "device" in d
        assert d["mcts"]["num_sims"] == 200
        assert d["mcts"]["c_exploration"] == 1.5
        assert d["device"] == "mps"

    def test_from_dict_nested_format(self):
        """from_dict should load nested structure."""
        d = {
            "mcts": {
                "num_sims": 300,
                "c_exploration": 1.8,
                "max_rollout_depth": 100,
                "rollout_seed": 99,
                "illegal_action_penalty": 1e7,
            },
            "device": "cuda",
        }
        config = VanillaMCTSAgentConfig.from_dict(d)

        assert config.mcts.num_sims == 300
        assert config.mcts.c_exploration == 1.8
        assert config.mcts.max_rollout_depth == 100
        assert config.mcts.rollout_seed == 99
        assert config.mcts.illegal_action_penalty == 1e7
        assert config.device == "cuda"

    def test_roundtrip(self):
        """Config should survive serialization roundtrip."""
        core = MCTSConfig(num_sims=777, c_exploration=1.9)
        original = VanillaMCTSAgentConfig(mcts=core, device="mps")

        d = original.to_dict()
        restored = VanillaMCTSAgentConfig.from_dict(d)

        assert restored.mcts.num_sims == original.mcts.num_sims
        assert restored.mcts.c_exploration == original.mcts.c_exploration
        assert restored.device == original.device

    def test_yaml_roundtrip(self):
        """Config should survive YAML roundtrip."""
        core = MCTSConfig(num_sims=888, c_exploration=2.1)
        original = VanillaMCTSAgentConfig(mcts=core, device="cpu")

        yaml_str = yaml.dump(original.to_dict())
        loaded_dict = yaml.safe_load(yaml_str)
        restored = VanillaMCTSAgentConfig.from_dict(loaded_dict)

        assert restored.mcts.num_sims == original.mcts.num_sims
        assert restored.mcts.c_exploration == original.mcts.c_exploration
        assert restored.device == original.device
