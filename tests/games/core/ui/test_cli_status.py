"""
Tests for CLIStatusDisplay class.

Tests both TTY and non-TTY modes with mocked stdout.
"""

from io import StringIO
from unittest.mock import patch, MagicMock
import pytest

from src.games.core.ui.cli_status import CLIStatusDisplay


class TestCLIStatusDisplayInit:
    """Test CLIStatusDisplay initialization"""

    def test_init_human_vs_agent_mode(self):
        """Test initialization with human_vs_agent mode"""
        display = CLIStatusDisplay(mode="human_vs_agent")
        assert display.mode == "human_vs_agent"
        assert display._last_block_height == 0
        assert display._first_render is True

    def test_init_agent_vs_agent_mode(self):
        """Test initialization with agent_vs_agent mode"""
        display = CLIStatusDisplay(mode="agent_vs_agent")
        assert display.mode == "agent_vs_agent"

    def test_init_invalid_mode_raises_error(self):
        """Test initialization with invalid mode raises ValueError"""
        with pytest.raises(ValueError, match="Invalid mode"):
            CLIStatusDisplay(mode="invalid_mode")

    def test_init_tty_override(self):
        """Test TTY detection can be overridden"""
        display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=False)
        assert display._tty_enabled is False

        display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=True)
        assert display._tty_enabled is True


class TestCLIStatusDisplayHumanVsAgent:
    """Test CLIStatusDisplay for human vs agent mode"""

    def test_update_status_human_vs_agent_non_tty(self):
        """Test status update for human vs agent in non-TTY mode"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=False)
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Your move",
                control_hints="Click=move, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        assert "TicTacToe" in output
        assert "You (X) vs Agent (O)" in output
        assert "Your move" in output
        assert "Click=move" in output
        assert "═" in output  # Separator

    def test_update_status_human_vs_agent_tty(self):
        """Test status update for human vs agent in TTY mode"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=True)
            display.update_status(
                game_name="Connect4",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Agent move…",
                control_hints="Click=move, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        # In TTY mode, should contain ANSI escape codes
        assert "\033[K" in output  # Clear line escape code
        assert "Connect4" in output
        assert "Agent move…" in output


class TestCLIStatusDisplayAgentVsAgent:
    """Test CLIStatusDisplay for agent vs agent mode"""

    def test_update_status_agent_vs_agent_non_tty(self):
        """Test status update for agent vs agent in non-TTY mode"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="agent_vs_agent", tty_enabled=False)
            display.update_status(
                game_name="TicTacToe",
                agent1_name="AlphaZero",
                agent2_name="Random",
                agent1_symbol="X",
                agent2_symbol="O",
                status_msg="AlphaZero to move",
                games_played=42,
                wins1=18,
                wins2=15,
                draws=9,
                control_hints="SPACE=step, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        assert "TicTacToe" in output
        assert "AlphaZero (X) vs Random (O)" in output
        assert "AlphaZero to move" in output
        assert "Games:  42" in output
        assert "AlphaZero: 18 wins" in output
        assert "Random: 15 wins" in output
        assert "Draws: 9" in output
        assert "SPACE=step" in output

    def test_update_status_agent_vs_agent_tty(self):
        """Test status update for agent vs agent in TTY mode"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="agent_vs_agent", tty_enabled=True)
            display.update_status(
                game_name="Connect4",
                agent1_name="Agent1",
                agent2_name="Agent2",
                agent1_symbol="X",
                agent2_symbol="O",
                status_msg="Agent1 to move",
                games_played=10,
                wins1=5,
                wins2=3,
                draws=2,
                control_hints="SPACE=step, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        # In TTY mode, should contain ANSI escape codes
        assert "\033[K" in output  # Clear line escape code
        assert "Agent1" in output
        assert "Agent2" in output
        assert "Games:  10" in output


class TestCLIStatusDisplayUpdates:
    """Test CLIStatusDisplay update mechanism"""

    def test_multiple_updates_non_tty(self):
        """Test multiple status updates in non-TTY mode"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=False)

            # First update
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Your move",
                control_hints="Click=move, N=new, Q=quit"
            )

            # Second update
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Agent move…",
                control_hints="Click=move, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        # Both messages should appear in output (non-TTY mode doesn't overwrite)
        assert "Your move" in output
        assert "Agent move…" in output

    def test_multiple_updates_tty(self):
        """Test multiple status updates in TTY mode"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=True)

            # First update
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Your move",
                control_hints="Click=move, N=new, Q=quit"
            )

            # Second update (should move cursor up)
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Agent move…",
                control_hints="Click=move, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        # In TTY mode, should have cursor movement codes
        assert "\033[" in output  # ANSI escape sequence
        # Both status messages should appear (before and after update)
        assert "Your move" in output
        assert "Agent move…" in output


class TestCLIStatusDisplayClear:
    """Test CLIStatusDisplay clear functionality"""

    def test_clear_non_tty(self):
        """Test clear in non-TTY mode (no-op)"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=False)
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Your move",
                control_hints="Click=move, N=new, Q=quit"
            )
            display.clear()

        # Clear in non-TTY mode should not produce output
        output = captured_output.getvalue()
        # Just verify the update happened
        assert "TicTacToe" in output

    def test_clear_tty(self):
        """Test clear in TTY mode"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=True)
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="Your move",
                control_hints="Click=move, N=new, Q=quit"
            )
            display.clear()

        output = captured_output.getvalue()
        # Should contain ANSI escape codes for clearing
        assert "\033[" in output  # ANSI escape sequences

    def test_clear_resets_state(self):
        """Test clear resets internal state"""
        display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=True)
        display._last_block_height = 10
        display._first_render = False

        with patch('sys.stdout', StringIO()):
            display.clear()

        assert display._last_block_height == 0
        assert display._first_render is True


class TestCLIStatusDisplayEdgeCases:
    """Test CLIStatusDisplay edge cases"""

    def test_long_agent_names(self):
        """Test status display with very long agent names"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="agent_vs_agent", tty_enabled=False)
            display.update_status(
                game_name="Connect4",
                agent1_name="VeryLongAgentNameThatExceedsNormalLength",
                agent2_name="AnotherVeryLongAgentNameForTesting",
                agent1_symbol="X",
                agent2_symbol="O",
                status_msg="Game in progress",
                games_played=100,
                wins1=50,
                wins2=40,
                draws=10,
                control_hints="SPACE=step, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        # Should handle long names without crashing
        assert "VeryLongAgentNameThatExceedsNormalLength" in output
        assert "AnotherVeryLongAgentNameForTesting" in output

    def test_special_characters_in_status(self):
        """Test status display with special characters"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="human_vs_agent", tty_enabled=False)
            display.update_status(
                game_name="TicTacToe",
                human_symbol="X",
                agent_symbol="O",
                status_msg="YOU WIN 🎉",
                control_hints="Click=move, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        assert "YOU WIN 🎉" in output

    def test_zero_scores(self):
        """Test status display with zero scores"""
        captured_output = StringIO()

        with patch('sys.stdout', captured_output):
            display = CLIStatusDisplay(mode="agent_vs_agent", tty_enabled=False)
            display.update_status(
                game_name="TicTacToe",
                agent1_name="Agent1",
                agent2_name="Agent2",
                agent1_symbol="X",
                agent2_symbol="O",
                status_msg="First game",
                games_played=0,
                wins1=0,
                wins2=0,
                draws=0,
                control_hints="SPACE=step, N=new, Q=quit"
            )

        output = captured_output.getvalue()
        assert "Games:  0" in output
        assert "Agent1: 0 wins" in output
        assert "Agent2: 0 wins" in output
        assert "Draws: 0" in output
