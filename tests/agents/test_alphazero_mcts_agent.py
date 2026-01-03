"""Tests for AlphaZeroMCTSAgent (base class)."""

import pytest
import torch
import yaml
from pathlib import Path

from src.agents.alphazero_mcts_agent import AlphaZeroMCTSAgent
from src.games.tictactoe.models.mlp import TicTacToeMLPNet


# ===== Direct Construction Tests =====

def test_init_stores_game_and_mcts(tictactoe_game, mock_mcts):
    """__init__ should store game and MCTS."""
    agent = AlphaZeroMCTSAgent(tictactoe_game, mock_mcts)

    assert agent.game is tictactoe_game
    assert agent.mcts is mock_mcts


def test_init_works_with_any_game(tictactoe_game, connect4_game, mock_mcts):
    """Agent should work with both TicTacToe and Connect4."""
    agent_ttt = AlphaZeroMCTSAgent(tictactoe_game, mock_mcts)
    agent_c4 = AlphaZeroMCTSAgent(connect4_game, mock_mcts)

    assert agent_ttt.game is tictactoe_game
    assert agent_c4.game is connect4_game


# ===== Agent Interface Tests =====

def test_act_delegates_to_mcts(tictactoe_game, mock_mcts):
    """act() should delegate to MCTS.play_move()."""
    agent = AlphaZeroMCTSAgent(tictactoe_game, mock_mcts)
    state = tictactoe_game.reset()

    mock_mcts.play_move.return_value = 3
    action = agent.act(state)

    mock_mcts.play_move.assert_called_once()
    assert action == 3


def test_start_clears_mcts(tictactoe_game, mock_mcts):
    """start() should clear MCTS tree."""
    agent = AlphaZeroMCTSAgent(tictactoe_game, mock_mcts)

    agent.start()

    mock_mcts.clear.assert_called_once()


def test_agent_lifecycle(tictactoe_game, tictactoe_mcts_real):
    """Full game playthrough with agent."""
    agent = AlphaZeroMCTSAgent(tictactoe_game, tictactoe_mcts_real)

    # Start game
    agent.start()
    state = tictactoe_game.reset()

    # Play a few moves
    for _ in range(3):
        action = agent.act(state)
        assert tictactoe_game.legal_actions(state)[action]

        state = tictactoe_game.next_state(state, action)

        # Check if game ended
        done, _ = tictactoe_game.terminal_value(state)
        if done:
            break


# ===== from_checkpoint Tests =====

@pytest.fixture
def old_style_checkpoint_dir(tmp_path):
    """Create old-style checkpoint with model.pt and train.yaml."""
    checkpoint_dir = tmp_path / "old_checkpoint"
    checkpoint_dir.mkdir()

    # Save model.pt
    model = TicTacToeMLPNet(hidden=8)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    # Save train.yaml with MCTS config
    train_config = {
        'mcts': {
            'num_sims': 50,
            'c_puct': 1.25,
            'dirichlet_alpha': 0.3,
            'dirichlet_eps': 0.25,
            'illegal_action_penalty': 1e9
        }
    }
    with (checkpoint_dir / "train.yaml").open('w') as f:
        yaml.dump(train_config, f)

    return checkpoint_dir


def test_from_checkpoint_loads_model(old_style_checkpoint_dir, tictactoe_game):
    """from_checkpoint() should load model and create MCTS."""
    agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=old_style_checkpoint_dir,
        game=tictactoe_game,
        model_cls=TicTacToeMLPNet,
        model_kwargs={'hidden': 8}
    )

    assert isinstance(agent, AlphaZeroMCTSAgent)
    assert agent.game is tictactoe_game
    assert agent.mcts is not None


def test_from_checkpoint_creates_mcts_from_yaml(old_style_checkpoint_dir, tictactoe_game):
    """MCTS config should be loaded from train.yaml."""
    agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=old_style_checkpoint_dir,
        game=tictactoe_game,
        model_cls=TicTacToeMLPNet,
        model_kwargs={'hidden': 8}
    )

    # Check MCTS config matches train.yaml
    assert agent.mcts.cfg.num_sims == 50
    assert agent.mcts.cfg.c_puct == 1.25
    assert agent.mcts.cfg.dirichlet_alpha == 0.3
    assert agent.mcts.cfg.dirichlet_eps == 0.25
    assert agent.mcts.cfg.illegal_action_penalty == 1e9


def test_from_checkpoint_device_override(old_style_checkpoint_dir, tictactoe_game):
    """Device parameter should work correctly."""
    agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=old_style_checkpoint_dir,
        game=tictactoe_game,
        model_cls=TicTacToeMLPNet,
        model_kwargs={'hidden': 8},
        device='cpu'
    )

    # Model should be on CPU
    for param in agent.mcts.model.parameters():
        assert param.device.type == 'cpu'


def test_from_checkpoint_mcts_params_override(old_style_checkpoint_dir, tictactoe_game):
    """MCTS params can be overridden at load time."""
    agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=old_style_checkpoint_dir,
        game=tictactoe_game,
        model_cls=TicTacToeMLPNet,
        model_kwargs={'hidden': 8},
        mcts_params={'num_sims': 100, 'c_puct': 2.0}  # Override
    )

    # Overridden params
    assert agent.mcts.cfg.num_sims == 100
    assert agent.mcts.cfg.c_puct == 2.0

    # Non-overridden params (from train.yaml)
    assert agent.mcts.cfg.dirichlet_alpha == 0.3
    assert agent.mcts.cfg.dirichlet_eps == 0.25


def test_from_checkpoint_model_kwargs(old_style_checkpoint_dir, tictactoe_game):
    """model_kwargs should be passed to model constructor."""
    agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=old_style_checkpoint_dir,
        game=tictactoe_game,
        model_cls=TicTacToeMLPNet,
        model_kwargs={'hidden': 8}  # Must match saved model
    )

    assert isinstance(agent.mcts.model, TicTacToeMLPNet)


def test_from_checkpoint_missing_model_pt(tmp_path, tictactoe_game):
    """Should raise FileNotFoundError for missing model.pt."""
    # Create directory with only train.yaml
    checkpoint_dir = tmp_path / "incomplete_checkpoint"
    checkpoint_dir.mkdir()

    train_config = {'mcts': {'num_sims': 50, 'c_puct': 1.0}}
    with (checkpoint_dir / "train.yaml").open('w') as f:
        yaml.dump(train_config, f)

    with pytest.raises(FileNotFoundError, match="No model.pt found"):
        AlphaZeroMCTSAgent.from_checkpoint(
            model_dir=checkpoint_dir,
            game=tictactoe_game,
            model_cls=TicTacToeMLPNet
        )


def test_from_checkpoint_missing_train_yaml(tmp_path, tictactoe_game):
    """Should raise FileNotFoundError for missing train.yaml."""
    # Create directory with only model.pt
    checkpoint_dir = tmp_path / "incomplete_checkpoint"
    checkpoint_dir.mkdir()

    model = TicTacToeMLPNet(hidden=8)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    with pytest.raises(FileNotFoundError, match="No train.yaml found"):
        AlphaZeroMCTSAgent.from_checkpoint(
            model_dir=checkpoint_dir,
            game=tictactoe_game,
            model_cls=TicTacToeMLPNet,
            model_kwargs={'hidden': 8}
        )


def test_from_checkpoint_model_in_eval_mode(old_style_checkpoint_dir, tictactoe_game):
    """Loaded model should be in eval mode."""
    agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=old_style_checkpoint_dir,
        game=tictactoe_game,
        model_cls=TicTacToeMLPNet,
        model_kwargs={'hidden': 8}
    )

    assert not agent.mcts.model.training


def test_from_checkpoint_loaded_agent_can_play(old_style_checkpoint_dir, tictactoe_game):
    """Loaded agent should be able to play."""
    agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=old_style_checkpoint_dir,
        game=tictactoe_game,
        model_cls=TicTacToeMLPNet,
        model_kwargs={'hidden': 8}
    )

    agent.start()
    state = tictactoe_game.reset()

    action = agent.act(state)
    assert tictactoe_game.legal_actions(state)[action]
