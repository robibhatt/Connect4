"""Tests for simulate_match function."""
import pytest
import numpy as np
from unittest.mock import Mock

from src.games.core.game_play import simulate_match
from src.agents.agent import Agent


# ===== Mock Agent Classes =====

class AlwaysWinAgent(Agent):
    """Agent that always makes perfect moves to win (for TicTacToe)."""

    def __init__(self, game):
        super().__init__(game)
        self.start_called = 0

    def start(self):
        self.start_called += 1

    def act(self, s):
        """Always pick first legal action."""
        legal = self.game.legal_actions(s)
        return int(np.argmax(legal))


class RandomTestAgent(Agent):
    """Agent that picks random legal actions with deterministic RNG."""

    def __init__(self, game, seed=42):
        super().__init__(game)
        self.rng = np.random.default_rng(seed)
        self.start_called = 0

    def start(self):
        self.start_called += 1

    def act(self, s):
        legal = self.game.legal_actions(s)
        legal_indices = np.where(legal)[0]
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available")
        return int(self.rng.choice(legal_indices))


class IllegalMoveAgent(Agent):
    """Agent that makes an illegal move (occupied square) after first move."""

    def __init__(self, game, other_agent):
        super().__init__(game)
        self.other_agent = other_agent
        self.move_count = 0

    def act(self, s):
        # On second move, try to play the same square as first move
        if self.move_count == 1:
            # Return action 0 again (should be occupied)
            return 0

        self.move_count += 1
        # First move: play action 0
        return 0


# ===== Basic Match Execution Tests =====

def test_simulate_match_completes(tictactoe_game, capsys):
    """simulate_match should complete specified number of games."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=2)

    # Capture output to verify completion
    captured = capsys.readouterr()
    assert "Games played: 2" in captured.out
    assert "Match Results" in captured.out


def test_simulate_match_single_game(tictactoe_game, capsys):
    """simulate_match should work with single game."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=1)

    captured = capsys.readouterr()
    assert "Games played: 1" in captured.out


def test_simulate_match_odd_number_of_games(tictactoe_game, capsys):
    """simulate_match should handle odd number of games."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=3)

    captured = capsys.readouterr()
    assert "Games played: 3" in captured.out


# ===== Alternating First Player Tests =====

def test_simulate_match_alternates_first_player(tictactoe_game, capsys):
    """simulate_match should alternate which agent moves first."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=4)

    captured = capsys.readouterr()
    lines = captured.out.split('\n')
    game_lines = [l for l in lines if l.startswith('[game')]

    # Games 0 and 2 should have Agent1 first
    assert 'first=Agent1' in game_lines[0]
    assert 'first=Agent1' in game_lines[2]

    # Games 1 and 3 should have Agent2 first
    assert 'first=Agent2' in game_lines[1]
    assert 'first=Agent2' in game_lines[3]


# ===== Agent Lifecycle Tests =====

def test_simulate_match_calls_agent_start(tictactoe_game, capsys):
    """simulate_match should call start() on both agents before each game."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=3)

    # Each agent should have start() called once per game
    assert agent1.start_called == 3
    assert agent2.start_called == 3


# ===== Win/Loss/Draw Tracking Tests =====

def test_simulate_match_tracks_wins_and_draws(tictactoe_game, capsys):
    """simulate_match should accurately track wins, losses, and draws."""
    # Play enough games to likely get some wins for each agent
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=10)

    captured = capsys.readouterr()

    # Verify summary contains win counts
    assert "Agent 1 total wins:" in captured.out
    assert "Agent 2 total wins:" in captured.out
    assert "Draws:" in captured.out
    assert "as first player:" in captured.out
    assert "as second player:" in captured.out


def test_simulate_match_win_count_accuracy(connect4_game, capsys):
    """Verify that win counts add up correctly."""
    agent1 = RandomTestAgent(connect4_game, seed=100)
    agent2 = RandomTestAgent(connect4_game, seed=101)

    num_games = 5
    simulate_match(connect4_game, agent1, agent2, num_games=num_games)

    captured = capsys.readouterr()

    # Count actual outcomes from game logs
    game_lines = [l for l in captured.out.split('\n') if l.startswith('[game')]
    assert len(game_lines) == num_games

    agent1_wins = sum(1 for l in game_lines if 'winner=Agent1' in l)
    agent2_wins = sum(1 for l in game_lines if 'winner=Agent2' in l)
    draws = sum(1 for l in game_lines if 'winner=Draw' in l)

    # Total should equal num_games
    assert agent1_wins + agent2_wins + draws == num_games


# ===== Position Tracking Tests =====

def test_simulate_match_tracks_wins_by_position(tictactoe_game, capsys):
    """simulate_match should track agent1 wins as first vs second player."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=10)

    captured = capsys.readouterr()

    # Extract win counts from output
    lines = captured.out.split('\n')

    # Find the lines with position breakdowns
    has_first_player_line = any('as first player:' in l for l in lines)
    has_second_player_line = any('as second player:' in l for l in lines)

    assert has_first_player_line
    assert has_second_player_line


# ===== Statistics Tests =====

def test_simulate_match_calculates_mean_value(tictactoe_game, capsys):
    """simulate_match should calculate mean value from agent1 POV."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=5)

    captured = capsys.readouterr()
    assert "Mean value (Agent 1 POV):" in captured.out


def test_simulate_match_win_rates_sum_to_one(connect4_game, capsys):
    """Win rates and draw rate should sum to approximately 1.0."""
    agent1 = RandomTestAgent(connect4_game, seed=200)
    agent2 = RandomTestAgent(connect4_game, seed=201)

    simulate_match(connect4_game, agent1, agent2, num_games=10)

    captured = capsys.readouterr()
    lines = captured.out.split('\n')

    # Extract win rates from output (they're in parentheses)
    import re
    agent1_rate = None
    agent2_rate = None
    draw_rate = None

    for line in lines:
        if 'Agent 1 total wins:' in line:
            match = re.search(r'\(([0-9.]+)\)', line)
            if match:
                agent1_rate = float(match.group(1))
        elif 'Agent 2 total wins:' in line:
            match = re.search(r'\(([0-9.]+)\)', line)
            if match:
                agent2_rate = float(match.group(1))
        elif 'Draws:' in line:
            match = re.search(r'\(([0-9.]+)\)', line)
            if match:
                draw_rate = float(match.group(1))

    # All rates should be found
    assert agent1_rate is not None
    assert agent2_rate is not None
    assert draw_rate is not None

    # Rates should sum to 1.0 (within floating point tolerance)
    total_rate = agent1_rate + agent2_rate + draw_rate
    assert abs(total_rate - 1.0) < 0.01


# ===== Error Handling Tests =====

def test_simulate_match_raises_on_illegal_move(tictactoe_game):
    """simulate_match should raise ValueError for illegal moves."""
    agent2 = RandomTestAgent(tictactoe_game, seed=42)
    agent1 = IllegalMoveAgent(tictactoe_game, agent2)

    with pytest.raises(ValueError, match="Illegal action"):
        simulate_match(tictactoe_game, agent1, agent2, num_games=1)


# ===== Output Format Tests =====

def test_simulate_match_output_format(tictactoe_game, capsys):
    """simulate_match should produce correctly formatted output."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=2)

    captured = capsys.readouterr()

    # Check for expected sections
    assert "=== Match Results ===" in captured.out
    assert "Games played:" in captured.out
    assert "Agent 1 total wins:" in captured.out
    assert "Agent 2 total wins:" in captured.out
    assert "Draws:" in captured.out
    assert "Mean value" in captured.out

    # Check game-by-game output format
    assert "[game 0000]" in captured.out or "[game 0]" in captured.out
    assert "first=" in captured.out
    assert "second=" in captured.out
    assert "winner=" in captured.out


def test_simulate_match_game_numbers_are_sequential(connect4_game, capsys):
    """Game numbers in output should be sequential."""
    agent1 = RandomTestAgent(connect4_game, seed=42)
    agent2 = RandomTestAgent(connect4_game, seed=43)

    simulate_match(connect4_game, agent1, agent2, num_games=3)

    captured = capsys.readouterr()
    lines = captured.out.split('\n')
    game_lines = [l for l in lines if l.startswith('[game')]

    # Should have exactly 3 game lines
    assert len(game_lines) == 3

    # Check sequential numbering
    assert '[game 0000]' in game_lines[0] or '[game 0]' in game_lines[0]
    assert '[game 0001]' in game_lines[1] or '[game 1]' in game_lines[1]
    assert '[game 0002]' in game_lines[2] or '[game 2]' in game_lines[2]


# ===== Integration Tests =====

def test_simulate_match_with_connect4(connect4_game, capsys):
    """simulate_match should work with Connect4 game."""
    agent1 = RandomTestAgent(connect4_game, seed=42)
    agent2 = RandomTestAgent(connect4_game, seed=43)

    simulate_match(connect4_game, agent1, agent2, num_games=2)

    captured = capsys.readouterr()
    assert "Games played: 2" in captured.out
    assert "Match Results" in captured.out


def test_simulate_match_with_tictactoe(tictactoe_game, capsys):
    """simulate_match should work with TicTacToe game."""
    agent1 = RandomTestAgent(tictactoe_game, seed=42)
    agent2 = RandomTestAgent(tictactoe_game, seed=43)

    simulate_match(tictactoe_game, agent1, agent2, num_games=2)

    captured = capsys.readouterr()
    assert "Games played: 2" in captured.out
    assert "Match Results" in captured.out
