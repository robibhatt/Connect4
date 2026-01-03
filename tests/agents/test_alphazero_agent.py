"""Tests for generic AlphaZeroAgent."""

import pytest
import torch
from pathlib import Path

from src.agents.alphazero_agent import AlphaZeroAgent
from src.agents.agent import Agent
from src.agents.checkpointable import CheckpointableAgent


# ===== Initialization Tests =====

@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "mock_mcts"),
    ("connect4_game", "mock_mcts"),
])
def test_agent_initialization_stores_references(game_fixture, mcts_fixture, request):
    """Agent should store game and MCTS references."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)

    assert agent.game is game
    assert agent.mcts is mcts


@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "mock_mcts"),
    ("connect4_game", "mock_mcts"),
])
def test_agent_is_instance_of_agent_abc(game_fixture, mcts_fixture, request):
    """Agent should be instance of Agent ABC."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)

    assert isinstance(agent, Agent)


@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "mock_mcts"),
    ("connect4_game", "mock_mcts"),
])
def test_agent_is_checkpointable(game_fixture, mcts_fixture, request):
    """Agent should implement CheckpointableAgent interface."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)

    assert isinstance(agent, CheckpointableAgent)


# ===== Action Selection Tests =====

@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "mock_mcts"),
    ("connect4_game", "mock_mcts"),
])
def test_act_delegates_to_mcts(game_fixture, mcts_fixture, request):
    """act() should call MCTS.play_move()."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)
    state = game.reset()

    # Mock returns action 0
    mcts.play_move.return_value = 0

    action = agent.act(state)

    # Should call play_move once with the state
    mcts.play_move.assert_called_once()
    assert action == 0


@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "tictactoe_mcts_real"),
    ("connect4_game", "connect4_mcts_real"),
])
def test_act_returns_valid_action(game_fixture, mcts_fixture, request):
    """act() should return legal action."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)
    state = game.reset()

    action = agent.act(state)

    # Action should be legal
    legal_actions = game.legal_actions(state)
    assert legal_actions[action], f"Action {action} should be legal"


@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "mock_mcts"),
    ("connect4_game", "mock_mcts"),
])
def test_act_with_mock_mcts(game_fixture, mcts_fixture, request):
    """act() should work correctly with mocked MCTS."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)
    state = game.reset()

    # Set up mock to return different actions
    mcts.play_move.side_effect = [0, 1, 2]

    # Should return the mocked actions in sequence
    assert agent.act(state) == 0
    assert agent.act(state) == 1
    assert agent.act(state) == 2


# ===== Game Lifecycle Tests =====

@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "mock_mcts"),
    ("connect4_game", "mock_mcts"),
])
def test_start_clears_mcts_tree(game_fixture, mcts_fixture, request):
    """start() should clear MCTS tree."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)

    agent.start()

    # Should call clear() exactly once
    mcts.clear.assert_called_once()


@pytest.mark.parametrize("game_fixture,mcts_fixture", [
    ("tictactoe_game", "mock_mcts"),
    ("connect4_game", "mock_mcts"),
])
def test_multiple_acts_without_start(game_fixture, mcts_fixture, request):
    """Can call act() multiple times without calling start()."""
    game = request.getfixturevalue(game_fixture)
    mcts = request.getfixturevalue(mcts_fixture)

    agent = AlphaZeroAgent(game, mcts)
    state = game.reset()

    mcts.play_move.return_value = 0

    # Should be able to call act() multiple times
    for _ in range(5):
        agent.act(state)

    # play_move should have been called 5 times
    assert mcts.play_move.call_count == 5


# ===== Checkpoint Save Tests =====

@pytest.mark.parametrize("agent_fixture", [
    "sample_alphazero_agent_tictactoe",
    "sample_alphazero_agent_connect4",
])
def test_to_checkpoint_saves_model_pt(tmp_path, agent_fixture, request):
    """to_checkpoint() should save model.pt."""
    agent = request.getfixturevalue(agent_fixture)

    checkpoint_dir = tmp_path / "test_checkpoint"
    checkpoint_dir.mkdir()

    agent.to_checkpoint(checkpoint_dir)

    model_path = checkpoint_dir / "model.pt"
    assert model_path.exists()
    assert model_path.is_file()


@pytest.mark.parametrize("agent_fixture", [
    "sample_alphazero_agent_tictactoe",
    "sample_alphazero_agent_connect4",
])
def test_to_checkpoint_uses_torch_save(tmp_path, agent_fixture, request):
    """to_checkpoint() should use torch.save correctly."""
    agent = request.getfixturevalue(agent_fixture)

    checkpoint_dir = tmp_path / "test_checkpoint"
    checkpoint_dir.mkdir()

    agent.to_checkpoint(checkpoint_dir)

    model_path = checkpoint_dir / "model.pt"

    # Should be loadable with torch.load
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0


# ===== Checkpoint Load Tests =====

@pytest.mark.parametrize("checkpoint_fixture,game_fixture", [
    ("checkpoint_dir_with_tictactoe_agent", "tictactoe_game"),
    ("checkpoint_dir_with_connect4_agent", "connect4_game"),
])
def test_from_checkpoint_reconstructs_agent(checkpoint_fixture, game_fixture, request):
    """from_checkpoint() should reconstruct agent."""
    checkpoint_dir = request.getfixturevalue(checkpoint_fixture)
    game = request.getfixturevalue(game_fixture)

    agent = AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        game=game
    )

    assert isinstance(agent, AlphaZeroAgent)
    assert agent.game is game
    assert hasattr(agent, 'mcts')


@pytest.mark.parametrize("checkpoint_fixture,game_fixture,agent_fixture", [
    ("checkpoint_dir_with_tictactoe_agent", "tictactoe_game", "sample_alphazero_agent_tictactoe"),
    ("checkpoint_dir_with_connect4_agent", "connect4_game", "sample_alphazero_agent_connect4"),
])
def test_from_checkpoint_loads_model_weights(checkpoint_fixture, game_fixture, agent_fixture, request):
    """Loaded agent should have correct model weights."""
    checkpoint_dir = request.getfixturevalue(checkpoint_fixture)
    game = request.getfixturevalue(game_fixture)
    sample_agent = request.getfixturevalue(agent_fixture)

    # Get original weights
    original_state_dict = sample_agent.mcts.model.state_dict()

    # Load agent
    loaded_agent = AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        game=game
    )

    loaded_state_dict = loaded_agent.mcts.model.state_dict()

    # Compare weights
    for key in original_state_dict.keys():
        assert key in loaded_state_dict
        assert torch.allclose(original_state_dict[key], loaded_state_dict[key])


@pytest.mark.parametrize("checkpoint_fixture,game_fixture", [
    ("checkpoint_dir_with_tictactoe_agent", "tictactoe_game"),
    ("checkpoint_dir_with_connect4_agent", "connect4_game"),
])
def test_from_checkpoint_creates_mcts(checkpoint_fixture, game_fixture, request):
    """Loaded agent should have MCTS with correct config."""
    checkpoint_dir = request.getfixturevalue(checkpoint_fixture)
    game = request.getfixturevalue(game_fixture)

    agent = AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        game=game
    )

    assert agent.mcts is not None
    assert hasattr(agent.mcts, 'cfg')
    assert agent.mcts.cfg.num_sims == 10  # From fixture config
    assert agent.mcts.cfg.c_puct == 1.0


@pytest.mark.parametrize("checkpoint_fixture,game_fixture", [
    ("checkpoint_dir_with_tictactoe_agent", "tictactoe_game"),
    ("checkpoint_dir_with_connect4_agent", "connect4_game"),
])
def test_from_checkpoint_device_override(checkpoint_fixture, game_fixture, request):
    """Device can be overridden during load."""
    checkpoint_dir = request.getfixturevalue(checkpoint_fixture)
    game = request.getfixturevalue(game_fixture)

    agent = AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        game=game,
        device='cpu'
    )

    # Model should be on CPU
    for param in agent.mcts.model.parameters():
        assert param.device.type == 'cpu'


@pytest.mark.parametrize("checkpoint_fixture,game_fixture", [
    ("checkpoint_dir_with_tictactoe_agent", "tictactoe_game"),
    ("checkpoint_dir_with_connect4_agent", "connect4_game"),
])
def test_from_checkpoint_eval_mode(checkpoint_fixture, game_fixture, request):
    """Loaded model should be in eval mode."""
    checkpoint_dir = request.getfixturevalue(checkpoint_fixture)
    game = request.getfixturevalue(game_fixture)

    agent = AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        game=game
    )

    # Model should be in eval mode
    assert not agent.mcts.model.training


# ===== Integration Tests =====

@pytest.mark.parametrize("checkpoint_fixture,game_fixture", [
    ("checkpoint_dir_with_tictactoe_agent", "tictactoe_game"),
    ("checkpoint_dir_with_connect4_agent", "connect4_game"),
])
def test_loaded_agent_can_play(checkpoint_fixture, game_fixture, request):
    """Loaded agent should be able to play a game."""
    checkpoint_dir = request.getfixturevalue(checkpoint_fixture)
    game = request.getfixturevalue(game_fixture)

    agent = AlphaZeroAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        game=game
    )

    agent.start()
    state = game.reset()

    # Should be able to select an action
    action = agent.act(state)
    assert game.legal_actions(state)[action]

    # Should be able to continue playing
    state = game.next_state(state, action)
    action2 = agent.act(state)
    assert game.legal_actions(state)[action2]


# ===== Deprecation Tests =====

def test_deprecated_tictactoe_class_shows_warning(tictactoe_game, tictactoe_mcts_real):
    """Using TicTacToeAlphaZeroAgent should show deprecation warning."""
    from src.agents.tictactoe_alphazero_agent import TicTacToeAlphaZeroAgent

    with pytest.warns(DeprecationWarning, match="Use AlphaZeroAgent instead"):
        TicTacToeAlphaZeroAgent(tictactoe_game, tictactoe_mcts_real)


def test_deprecated_connect4_class_shows_warning(connect4_game, connect4_mcts_real):
    """Using Connect4AlphaZeroAgent should show deprecation warning."""
    from src.agents.connect4_alphazero_agent import Connect4AlphaZeroAgent

    with pytest.warns(DeprecationWarning, match="Use AlphaZeroAgent instead"):
        Connect4AlphaZeroAgent(connect4_game, connect4_mcts_real)


def test_deprecated_classes_are_subclasses():
    """Deprecated classes should be subclasses of AlphaZeroAgent."""
    from src.agents.tictactoe_alphazero_agent import TicTacToeAlphaZeroAgent
    from src.agents.connect4_alphazero_agent import Connect4AlphaZeroAgent

    assert issubclass(TicTacToeAlphaZeroAgent, AlphaZeroAgent)
    assert issubclass(Connect4AlphaZeroAgent, AlphaZeroAgent)
