"""Tests for scripts/speed_test.py"""

import pytest
from pathlib import Path


class TestRunSpeedTest:
    """Tests for run_speed_test function"""

    def test_returns_move_times_list(self):
        """Speed test returns list of move times."""
        from scripts.speed_test import run_speed_test

        times = run_speed_test(game_name='tictactoe', agent_checkpoint=None, num_games=2)

        assert isinstance(times, list)
        assert len(times) > 0
        assert all(isinstance(t, float) for t in times)
        assert all(t >= 0 for t in times)


class TestWriteResults:
    """Tests for write_results function"""

    def test_creates_file_with_stats(self, tmp_path):
        """Results file is created with mean and std."""
        from scripts.speed_test import write_results

        times = [0.1, 0.2, 0.15]
        output_path = tmp_path / "speed_test_results.txt"

        write_results(output_path, times, agent_name='TestAgent', game_name='tictactoe', num_games=1)

        assert output_path.exists()
        content = output_path.read_text()
        assert 'Mean:' in content
        assert 'Std:' in content
        assert 'TestAgent' in content
