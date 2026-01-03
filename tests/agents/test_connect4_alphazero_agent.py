"""Tests for Connect4AlphaZeroAgent."""

import pytest
import torch
from pathlib import Path

from src.agents.connect4_alphazero_agent import Connect4AlphaZeroAgent
from src.agents.agent import Agent
from src.agents.checkpointable import CheckpointableAgent


# ===== Initialization Tests =====

def test_agent_initialization_stores_references(connect4_game, mock_mcts):
    """Agent should store game and MCTS references."""
    agent = Connect4AlphaZeroAgent(connect4_game, mock_mcts)

    assert agent.game is connect4_game
    assert agent.mcts is mock_mcts


def test_agent_is_instance_of_agent_abc(connect4_game, mock_mcts):
    """Agent should be instance of Agent ABC."""
    agent = Connect4AlphaZeroAgent(connect4_game, mock_mcts)

    assert isinstance(agent, Agent)


def test_agent_is_checkpointable(connect4_game, mock_mcts):
    """Agent should implement CheckpointableAgent interface."""
    agent = Connect4AlphaZeroAgent(connect4_game, mock_mcts)

    assert isinstance(agent, CheckpointableAgent)


# ===== Action Selection Tests =====

def test_act_delegates_to_mcts(connect4_game, mock_mcts):
    """act() should call MCTS.play_move()."""
    agent = Connect4AlphaZeroAgent(connect4_game, mock_mcts)
    state = connect4_game.reset()

    # Mock returns action 0
    mock_mcts.play_move.return_value = 0

    action = agent.act(state)

    # Should call play_move once with the state
    mock_mcts.play_move.assert_called_once()
    assert action == 0


def test_act_returns_valid_action(connect4_game, connect4_mcts_real):
    """act() should return legal action."""
    agent = Connect4AlphaZeroAgent(connect4_game, connect4_mcts_real)
    state = connect4_game.reset()

    action = agent.act(state)

    # Action should be legal (in range 0-6 for Connect4)
    legal_actions = connect4_game.legal_actions(state)
    assert legal_actions[action], f"Action {action} should be legal"
    assert 0 <= action < 7, f"Action {action} should be in range 0-6"


def test_act_with_mock_mcts(connect4_game, mock_mcts):
    """act() should work correctly with mocked MCTS."""
    agent = Connect4AlphaZeroAgent(connect4_game, mock_mcts)
    state = connect4_game.reset()

    # Set up mock to return different actions
    mock_mcts.play_move.side_effect = [0, 1, 2, 3]

    # Should return the mocked actions in sequence
    assert agent.act(state) == 0
    assert agent.act(state) == 1
    assert agent.act(state) == 2
    assert agent.act(state) == 3


# ===== Game Lifecycle Tests =====

def test_start_clears_mcts_tree(connect4_game, mock_mcts):
    """start() should clear MCTS tree."""
    agent = Connect4AlphaZeroAgent(connect4_game, mock_mcts)

    agent.start()

    # Should call clear() exactly once
    mock_mcts.clear.assert_called_once()


def test_multiple_acts_without_start(connect4_game, mock_mcts):
    """Can call act() multiple times without calling start()."""
    agent = Connect4AlphaZeroAgent(connect4_game, mock_mcts)
    state = connect4_game.reset()

    mock_mcts.play_move.return_value = 0

    # Should be able to call act() multiple times
    for _ in range(5):
        agent.act(state)

    # play_move should have been called 5 times
    assert mock_mcts.play_move.call_count == 5


# ===== Checkpoint Save Tests =====

def test_to_checkpoint_saves_model_pt(tmp_path, sample_connect4_agent):
    """to_checkpoint() should save model.pt."""
    checkpoint_dir = tmp_path / "test_checkpoint"
    checkpoint_dir.mkdir()

    sample_connect4_agent.to_checkpoint(checkpoint_dir)

    model_path = checkpoint_dir / "model.pt"
    assert model_path.exists()
    assert model_path.is_file()


def test_to_checkpoint_uses_torch_save(tmp_path, sample_connect4_agent):
    """to_checkpoint() should use torch.save correctly."""
    checkpoint_dir = tmp_path / "test_checkpoint"
    checkpoint_dir.mkdir()

    sample_connect4_agent.to_checkpoint(checkpoint_dir)

    model_path = checkpoint_dir / "model.pt"

    # Should be loadable with torch.load
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0


# ===== Checkpoint Load Tests =====

def test_from_checkpoint_reconstructs_agent(checkpoint_dir_with_connect4_agent, connect4_game):
    """from_checkpoint() should reconstruct agent."""
    agent = Connect4AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir_with_connect4_agent,
        game=connect4_game
    )

    assert isinstance(agent, Connect4AlphaZeroAgent)
    assert agent.game is connect4_game
    assert hasattr(agent, 'mcts')


def test_from_checkpoint_loads_model_weights(checkpoint_dir_with_connect4_agent,
                                              connect4_game,
                                              sample_connect4_agent):
    """Loaded agent should have correct model weights."""
    # Get original weights
    original_state_dict = sample_connect4_agent.mcts.model.state_dict()

    # Load agent
    loaded_agent = Connect4AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir_with_connect4_agent,
        game=connect4_game
    )

    loaded_state_dict = loaded_agent.mcts.model.state_dict()

    # Compare weights
    for key in original_state_dict.keys():
        assert key in loaded_state_dict
        assert torch.allclose(original_state_dict[key], loaded_state_dict[key])


def test_from_checkpoint_creates_mcts(checkpoint_dir_with_connect4_agent, connect4_game):
    """Loaded agent should have MCTS with correct config."""
    agent = Connect4AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir_with_connect4_agent,
        game=connect4_game
    )

    assert agent.mcts is not None
    assert hasattr(agent.mcts, 'cfg')
    assert agent.mcts.cfg.num_sims == 10  # From fixture config
    assert agent.mcts.cfg.c_puct == 1.0


def test_from_checkpoint_device_override(checkpoint_dir_with_connect4_agent, connect4_game):
    """Device can be overridden during load."""
    agent = Connect4AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir_with_connect4_agent,
        game=connect4_game,
        device='cpu'
    )

    # Model should be on CPU
    for param in agent.mcts.model.parameters():
        assert param.device.type == 'cpu'


def test_from_checkpoint_eval_mode(checkpoint_dir_with_connect4_agent, connect4_game):
    """Loaded model should be in eval mode."""
    agent = Connect4AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir_with_connect4_agent,
        game=connect4_game
    )

    # Model should be in eval mode
    assert not agent.mcts.model.training


# ===== Integration Tests =====

def test_loaded_agent_can_play(checkpoint_dir_with_connect4_agent, connect4_game):
    """Loaded agent should be able to play a game."""
    agent = Connect4AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir_with_connect4_agent,
        game=connect4_game
    )

    agent.start()
    state = connect4_game.reset()

    # Should be able to select an action
    action = agent.act(state)
    assert connect4_game.legal_actions(state)[action]
    assert 0 <= action < 7

    # Should be able to continue playing
    state = connect4_game.next_state(state, action)
    action2 = agent.act(state)
    assert connect4_game.legal_actions(state)[action2]
    assert 0 <= action2 < 7
