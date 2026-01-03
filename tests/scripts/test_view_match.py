"""
Tests for view_match script.

These tests verify:
- Config loading and validation
- Agent creation from config
- UI selection based on game type
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from scripts.view_match import (
    load_view_match_config,
    create_agent,
    create_match_ui,
    ViewMatchConfig,
    main
)
from src.games.tictactoe import TicTacToe
from src.games.connect4 import Connect4
from src.agents import RandomAgent


# ===== A. Config Loading Tests (6 tests) =====

def test_load_valid_config():
    """Test 55: Load valid config returns ViewMatchConfig"""
    config_data = {
        'game': 'tictactoe',
        'agent1': {
            'type': 'random',
            'name': 'Random1'
        },
        'agent2': {
            'type': 'random',
            'name': 'Random2'
        },
        'ui': {
            'pause_after_move': 0.3,
            'window_size': 600
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        config = load_view_match_config(config_path)

        assert config.game_name == 'tictactoe'
        assert config.agent1_type == 'random'
        assert config.agent1_name == 'Random1'
        assert config.agent2_type == 'random'
        assert config.agent2_name == 'Random2'
        assert config.pause_after_move == 0.3
        assert config.window_size == 600
    finally:
        config_path.unlink()


def test_missing_game_field_raises_error():
    """Test 56: Missing 'game' field raises ValueError"""
    config_data = {
        'agent1': {'type': 'random'},
        'agent2': {'type': 'random'}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="game"):
            load_view_match_config(config_path)
    finally:
        config_path.unlink()


def test_missing_agent1_raises_error():
    """Test 57: Missing 'agent1' section raises ValueError"""
    config_data = {
        'game': 'tictactoe',
        'agent2': {'type': 'random'}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="agent1"):
            load_view_match_config(config_path)
    finally:
        config_path.unlink()


def test_missing_agent2_raises_error():
    """Test 58: Missing 'agent2' section raises ValueError"""
    config_data = {
        'game': 'tictactoe',
        'agent1': {'type': 'random'}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="agent2"):
            load_view_match_config(config_path)
    finally:
        config_path.unlink()


def test_invalid_game_name_raises_error():
    """Test 59: Invalid game name raises ValueError"""
    config_data = {
        'game': 'invalid_game',
        'agent1': {'type': 'random'},
        'agent2': {'type': 'random'}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        with pytest.raises(ValueError):
            load_view_match_config(config_path)
    finally:
        config_path.unlink()


def test_agent_name_defaults_to_type():
    """Test 60: Agent name defaults to type when not provided"""
    config_data = {
        'game': 'tictactoe',
        'agent1': {'type': 'random'},
        'agent2': {'type': 'random'}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        config = load_view_match_config(config_path)

        # Should default to capitalized type name
        assert config.agent1_name in ['random', 'Random', 'RandomAgent']
        assert config.agent2_name in ['random', 'Random', 'RandomAgent']
    finally:
        config_path.unlink()


# ===== B. Agent Creation Tests (4 tests) =====

def test_create_random_agent():
    """Test 61: create_agent returns RandomAgent for type='random'"""
    game = TicTacToe()
    agent = create_agent('random', None, game)

    assert isinstance(agent, RandomAgent)


def test_create_alphazero_agent():
    """Test 62: create_agent returns AlphaZeroAgent for valid checkpoint"""
    # This test requires a valid checkpoint directory
    # We'll mock the load_agent_checkpoint function
    game = TicTacToe()

    mock_checkpoint_dir = Path('/fake/checkpoint')

    with patch('scripts.view_match.load_agent_checkpoint') as mock_load:
        mock_agent = Mock()
        mock_load.return_value = mock_agent

        agent = create_agent('alphazero', mock_checkpoint_dir, game)

        assert agent is mock_agent
        # load_agent_checkpoint only takes checkpoint_dir (game is loaded from checkpoint)
        mock_load.assert_called_once_with(mock_checkpoint_dir)


def test_alphazero_without_checkpoint_raises_error():
    """Test 63: AlphaZero without checkpoint raises ValueError"""
    game = TicTacToe()

    with pytest.raises(ValueError, match="checkpoint"):
        create_agent('alphazero', None, game)


def test_invalid_checkpoint_dir_raises_error():
    """Test 64: Invalid checkpoint directory raises ValueError"""
    game = TicTacToe()
    invalid_dir = Path('/nonexistent/checkpoint')

    with pytest.raises(ValueError):
        create_agent('alphazero', invalid_dir, game)


# ===== C. UI Selection Tests (4 tests) =====

def test_create_tictactoe_match_ui():
    """Test 65: create_match_ui returns TicTacToeMatchUI for TicTacToe"""
    from src.games.tictactoe.ui.tictactoe_match_ui import TicTacToeMatchUI

    game = TicTacToe()
    agent1 = RandomAgent(game)
    agent2 = RandomAgent(game)

    config = ViewMatchConfig(
        game_name='tictactoe',
        agent1_type='random',
        agent1_name='Agent1',
        agent1_checkpoint_dir=None,
        agent2_type='random',
        agent2_name='Agent2',
        agent2_checkpoint_dir=None,
        pause_after_move=0.3,
        window_size=600
    )

    ui = create_match_ui(game, agent1, agent2, config)

    assert isinstance(ui, TicTacToeMatchUI)


def test_create_connect4_match_ui():
    """Test 66: create_match_ui returns Connect4MatchUI for Connect4"""
    from src.games.connect4.ui.connect4_match_ui import Connect4MatchUI

    game = Connect4()
    agent1 = RandomAgent(game)
    agent2 = RandomAgent(game)

    config = ViewMatchConfig(
        game_name='connect4',
        agent1_type='random',
        agent1_name='Agent1',
        agent1_checkpoint_dir=None,
        agent2_type='random',
        agent2_name='Agent2',
        agent2_checkpoint_dir=None,
        pause_after_move=0.3,
        window_size=600
    )

    ui = create_match_ui(game, agent1, agent2, config)

    assert isinstance(ui, Connect4MatchUI)


def test_ui_receives_agent_names():
    """Test 67: UI receives correct agent names from config"""
    from src.games.tictactoe.ui.tictactoe_match_ui import TicTacToeMatchUI

    game = TicTacToe()
    agent1 = RandomAgent(game)
    agent2 = RandomAgent(game)

    config = ViewMatchConfig(
        game_name='tictactoe',
        agent1_type='random',
        agent1_name='AlphaBot',
        agent1_checkpoint_dir=None,
        agent2_type='random',
        agent2_name='BetaBot',
        agent2_checkpoint_dir=None,
        pause_after_move=0.3,
        window_size=600
    )

    ui = create_match_ui(game, agent1, agent2, config)

    assert ui.agent1_name == 'AlphaBot'
    assert ui.agent2_name == 'BetaBot'


def test_ui_receives_pause_config():
    """Test 68: UI receives pause_after_move from config"""
    game = TicTacToe()
    agent1 = RandomAgent(game)
    agent2 = RandomAgent(game)

    config = ViewMatchConfig(
        game_name='tictactoe',
        agent1_type='random',
        agent1_name='Agent1',
        agent1_checkpoint_dir=None,
        agent2_type='random',
        agent2_name='Agent2',
        agent2_checkpoint_dir=None,
        pause_after_move=0.5,
        window_size=600
    )

    ui = create_match_ui(game, agent1, agent2, config)

    assert ui.pause_after_move == 0.5


# ===== D. Integration Tests (2 tests) =====

def test_main_loads_config_and_creates_ui():
    """Test 69: main() loads config, creates agents and UI"""
    config_data = {
        'game': 'tictactoe',
        'agent1': {'type': 'random', 'name': 'Agent1'},
        'agent2': {'type': 'random', 'name': 'Agent2'},
        'ui': {'pause_after_move': 0.3, 'window_size': 600}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        # Mock UI.run() to prevent actual pygame loop
        with patch('scripts.view_match.TicTacToeMatchUI') as MockUI:
            mock_ui_instance = Mock()
            MockUI.return_value = mock_ui_instance

            # Mock load_view_match_config to return a config using our test file
            with patch('scripts.view_match.load_view_match_config') as mock_load:
                from scripts.view_match import ViewMatchConfig
                mock_load.return_value = ViewMatchConfig(
                    game_name='tictactoe',
                    agent1_type='random',
                    agent1_name='Agent1',
                    agent1_checkpoint_dir=None,
                    agent2_type='random',
                    agent2_name='Agent2',
                    agent2_checkpoint_dir=None,
                    pause_after_move=0.3,
                    window_size=600
                )

                main()

                # Verify UI was created and run() was called
                MockUI.assert_called_once()
                mock_ui_instance.run.assert_called_once()
    finally:
        config_path.unlink()


def test_main_handles_config_errors_gracefully():
    """Test 70: main() handles config errors and exits cleanly"""
    # Create invalid config
    config_data = {'invalid': 'config'}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        with patch('scripts.view_match.Path') as MockPath:
            MockPath.return_value = config_path

            with patch('scripts.view_match.print_error') as mock_print_error:
                with patch('sys.exit') as mock_exit:
                    main()

                    # Should have printed error and exited
                    mock_print_error.assert_called()
                    mock_exit.assert_called_with(1)
    finally:
        config_path.unlink()
