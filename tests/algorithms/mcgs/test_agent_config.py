"""Tests for MCGSAgentConfig class."""

import pytest
import yaml

from src.algorithms.mcgs.agent_config import MCGSAgentConfig
from src.algorithms.mcgs.mcgs import MCGSCoreConfig
from src.agents.config import AgentConfig


class TestMCGSAgentConfig:
    """Tests for MCGSAgentConfig dataclass with composition."""

    def test_mcgs_agent_config_is_agent_config(self):
        """MCGSAgentConfig should inherit from AgentConfig."""
        config = MCGSAgentConfig(
            mcgs=MCGSCoreConfig(
                num_sims=100,
                c_exploration=1.414,
                max_rollout_depth=None,
                rollout_seed=None,
                illegal_action_penalty=1e9
            )
        )
        assert isinstance(config, AgentConfig)

    def test_mcgs_agent_config_has_embedded_core(self):
        """MCGSAgentConfig should have embedded MCGSCoreConfig."""
        core = MCGSCoreConfig(
            num_sims=100,
            c_exploration=1.414,
            max_rollout_depth=50,
            rollout_seed=42,
            illegal_action_penalty=1e9
        )
        config = MCGSAgentConfig(mcgs=core)

        assert config.mcgs.num_sims == 100
        assert config.mcgs.c_exploration == 1.414
        assert config.mcgs.max_rollout_depth == 50
        assert config.mcgs.rollout_seed == 42
        assert config.mcgs.illegal_action_penalty == 1e9

    def test_mcgs_agent_config_to_dict(self):
        """to_dict() should serialize to nested structure."""
        core = MCGSCoreConfig(
            num_sims=100,
            c_exploration=1.5,
            max_rollout_depth=None,
            rollout_seed=42,
            illegal_action_penalty=1e9
        )
        config = MCGSAgentConfig(mcgs=core)

        d = config.to_dict()

        assert 'mcgs' in d
        assert d['mcgs']['num_sims'] == 100
        assert d['mcgs']['c_exploration'] == 1.5
        assert d['mcgs']['max_rollout_depth'] is None
        assert d['mcgs']['rollout_seed'] == 42
        assert d['mcgs']['illegal_action_penalty'] == 1e9

    def test_mcgs_agent_config_from_dict(self):
        """from_dict() should deserialize nested structure."""
        d = {
            'mcgs': {
                'num_sims': 200,
                'c_exploration': 2.0,
                'max_rollout_depth': 100,
                'rollout_seed': None,
                'illegal_action_penalty': 1e9,
            },
            'device': 'cpu'
        }

        config = MCGSAgentConfig.from_dict(d)

        assert config.mcgs.num_sims == 200
        assert config.mcgs.c_exploration == 2.0
        assert config.mcgs.max_rollout_depth == 100

    def test_mcgs_agent_config_yaml_roundtrip(self):
        """Config should survive YAML serialization roundtrip."""
        core = MCGSCoreConfig(
            num_sims=150,
            c_exploration=1.414,
            max_rollout_depth=None,
            rollout_seed=42,
            illegal_action_penalty=1e9
        )
        config = MCGSAgentConfig(mcgs=core)

        # Serialize to YAML
        yaml_str = yaml.dump(config.to_dict())

        # Deserialize
        data = yaml.safe_load(yaml_str)
        restored = MCGSAgentConfig.from_dict(data)

        assert restored.mcgs.num_sims == config.mcgs.num_sims
        assert restored.mcgs.c_exploration == config.mcgs.c_exploration
        assert restored.mcgs.rollout_seed == config.mcgs.rollout_seed
