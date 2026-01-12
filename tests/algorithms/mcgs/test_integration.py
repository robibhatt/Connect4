"""Integration tests for MCGS algorithm."""

import pytest
import yaml
from pathlib import Path

from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.mcgs.mcgs import MCGSCoreConfig
from src.algorithms.shared.trainer_args import TrainerArgs


class TestMCGSIntegration:
    """Integration tests for MCGS end-to-end workflow."""

    def test_full_workflow_via_registry(self, tictactoe_game):
        """Test complete training workflow using registry."""
        # Get components from registry
        config_cls = AlgorithmRegistry.get_config_class('mcgs')
        trainer_factory = AlgorithmRegistry.get_trainer_factory('mcgs')
        agent_config_factory = AlgorithmRegistry.get_agent_config_factory('mcgs')

        # Create config with composition
        config = config_cls(
            core=MCGSCoreConfig(num_sims=10),
            trainer=TrainerArgs(num_test_games=1, verbose=False),
        )

        # Create and run trainer
        trainer = trainer_factory(tictactoe_game, None, config)
        trainer.run()

        # Create agent
        agent = trainer.create_agent()
        state = tictactoe_game.reset()
        action = agent.act(state)

        assert 0 <= action < tictactoe_game.action_size

        # Create agent config for checkpointing
        agent_config = agent_config_factory(config)
        assert agent_config.mcgs.num_sims == 10

    def test_checkpoint_roundtrip(self, tictactoe_game, tmp_path):
        """Test save and load checkpoint."""
        from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig
        from src.algorithms.mcgs.agent import MCGSAgent
        from src.algorithms.mcgs.agent_config import MCGSAgentConfig

        # Create original agent
        cfg = MCGSCoreConfig(num_sims=100, c_exploration=1.5)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)
        original_agent = MCGSAgent(game=tictactoe_game, mcgs=mcgs)

        # Save checkpoint
        original_agent.to_checkpoint(tmp_path)

        # Create agent.yaml (simulating what checkpoint_utils would do)
        agent_config = MCGSAgentConfig(mcgs=cfg)
        agent_yaml = {
            'agent_class': 'MCGSAgent',
            'mcgs': agent_config.to_dict()
        }
        with (tmp_path / 'agent.yaml').open('w') as f:
            yaml.dump(agent_yaml, f)

        # Load checkpoint
        loaded_agent = MCGSAgent.from_checkpoint(tmp_path, tictactoe_game)

        # Verify loaded agent works
        state = tictactoe_game.reset()
        action = loaded_agent.act(state)

        assert 0 <= action < tictactoe_game.action_size
        assert loaded_agent.mcgs.cfg.num_sims == 100
        assert loaded_agent.mcgs.cfg.c_exploration == 1.5

    def test_mcgs_plays_full_game(self, tictactoe_game):
        """MCGS should be able to play a complete game."""
        from src.algorithms.mcgs.mcgs import MCGS, MCGSCoreConfig

        cfg = MCGSCoreConfig(num_sims=20)
        mcgs = MCGS(game=tictactoe_game, cfg=cfg)

        state = tictactoe_game.reset()
        moves = 0

        while moves < 20:  # TicTacToe max moves is 9
            done, _ = tictactoe_game.terminal_value(state)
            if done:
                break

            action = mcgs.play_move(state, deterministic=True)
            state = tictactoe_game.next_state(state, action)
            moves += 1

        # Game should have ended
        done, _ = tictactoe_game.terminal_value(state)
        assert done

    def test_mcgs_metadata_correct(self):
        """MCGS metadata should indicate no model required."""
        metadata = AlgorithmRegistry.get_metadata('mcgs')

        # Key: MCGS doesn't need a neural network model
        assert metadata.requires_model is False

        # But it does need a checkpoint (for config)
        assert metadata.requires_checkpoint is True

        # Only agent.yaml, no model.pt
        assert 'agent.yaml' in metadata.checkpoint_files
        assert 'model.pt' not in metadata.checkpoint_files
