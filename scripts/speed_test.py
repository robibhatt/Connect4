"""Speed test: measure agent move timing via self-play."""
import time
from pathlib import Path

import numpy as np
import yaml

from src.agents import RandomAgent, load_agent_checkpoint
from src.games.core.registry import GameRegistry


def run_speed_test(game_name: str, agent_checkpoint: Path | None, num_games: int = 20) -> list[float]:
    """Run self-play games and return list of move times."""
    GameClass = GameRegistry.get_game(game_name)
    game = GameClass()

    if agent_checkpoint:
        agent = load_agent_checkpoint(agent_checkpoint)
    else:
        agent = RandomAgent(game=game)

    move_times = []
    for _ in range(num_games):
        agent.start()
        state = game.reset()
        while True:
            done, _ = game.terminal_value(state)
            if done:
                break
            start = time.perf_counter()
            action = agent.act(state)
            move_times.append(time.perf_counter() - start)
            state = game.next_state(state, action)

    return move_times


def write_results(output_path: Path, times: list[float], agent_name: str, game_name: str, num_games: int):
    """Write timing statistics to file."""
    mean_t, std_t = np.mean(times), np.std(times)
    output_path.write_text(
        f"Speed Test Results\n"
        f"Agent: {agent_name}\n"
        f"Game: {game_name}\n"
        f"Games: {num_games}\n"
        f"Total moves: {len(times)}\n"
        f"Mean: {mean_t:.6f}s\n"
        f"Std:  {std_t:.6f}s\n"
    )


def main():
    config_path = Path(__file__).parent / "speed_test.yaml"
    config = yaml.safe_load(config_path.read_text())

    checkpoint = Path(config['agent']['checkpoint_dir'])
    num_games = config.get('num_games', 20)

    times = run_speed_test(config['game'], checkpoint, num_games)

    output_path = checkpoint / "speed_test_results.txt"
    write_results(output_path, times, checkpoint.name, config['game'], num_games)
    print(f"Results written to {output_path}")


if __name__ == '__main__':
    main()
