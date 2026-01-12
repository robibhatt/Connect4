"""Tests for MCGSAgent class."""

import pytest
from pathlib import Path

from src.algorithms.mcgs.agent import MCGSAgent
from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig
from src.agents.agent import Agent
from src.agents.checkpointable import CheckpointableAgent


class TestMCGSAgent:
    """Tests for MCGSAgent class."""

    def test_mcgs_agent_is_agent(self, tictactoe_game):
        """MCGSAgent should be an Agent subclass."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        agent = MCGSAgent(game=tictactoe_game, mcgs=mcgs)

        assert isinstance(agent, Agent)

    def test_mcgs_agent_is_checkpointable(self, tictactoe_game):
        """MCGSAgent should implement CheckpointableAgent."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        agent = MCGSAgent(game=tictactoe_game, mcgs=mcgs)

        assert isinstance(agent, CheckpointableAgent)

    def test_mcgs_agent_act(self, tictactoe_game):
        """MCGSAgent.act() should return valid action."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        agent = MCGSAgent(game=tictactoe_game, mcgs=mcgs)
        state = tictactoe_game.reset()

        action = agent.act(state)

        assert 0 <= action < tictactoe_game.action_size
        legal = tictactoe_game.legal_actions(state)
        assert legal[action]

    def test_mcgs_agent_start(self, tictactoe_game):
        """MCGSAgent.start() should clear MCGS cache."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        agent = MCGSAgent(game=tictactoe_game, mcgs=mcgs)
        state = tictactoe_game.reset()

        # Play to populate cache
        agent.act(state)

        # Start new game
        agent.start()

        assert len(mcgs.nodes) == 0

    def test_mcgs_agent_to_checkpoint(self, tictactoe_game, tmp_path):
        """to_checkpoint should not create model.pt (no model needed)."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        agent = MCGSAgent(game=tictactoe_game, mcgs=mcgs)

        # This should not raise and should not create model.pt
        agent.to_checkpoint(tmp_path)

        # No model.pt should exist
        assert not (tmp_path / 'model.pt').exists()

    def test_mcgs_agent_from_checkpoint(self, tictactoe_game, tmp_path):
        """from_checkpoint should reconstruct agent from agent.yaml."""
        import yaml

        # Create a mock checkpoint with nested structure
        agent_yaml = {
            'agent_class': 'MCGSAgent',
            'mcgs': {
                'mcgs': {
                    'num_sims': 100,
                    'c_exploration': 1.5,
                    'max_rollout_depth': None,
                    'rollout_seed': None,
                    'illegal_action_penalty': 1e9
                },
                'device': 'cpu'
            }
        }
        with (tmp_path / 'agent.yaml').open('w') as f:
            yaml.dump(agent_yaml, f)

        # Load agent
        agent = MCGSAgent.from_checkpoint(tmp_path, tictactoe_game)

        assert isinstance(agent, MCGSAgent)
        assert agent.mcgs.cfg.num_sims == 100
        assert agent.mcgs.cfg.c_exploration == 1.5

    def test_mcgs_agent_with_connect4(self, connect4_game):
        """MCGSAgent should work with Connect4."""
        cfg = MCGSCoreConfig(num_sims=10)
        mcgs = MCGS(game=connect4_game, cfg=cfg)
        agent = MCGSAgent(game=connect4_game, mcgs=mcgs)
        state = connect4_game.reset()

        action = agent.act(state)

        assert 0 <= action < connect4_game.action_size
