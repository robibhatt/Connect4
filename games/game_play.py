from games.game import Game
from agent import Agent
import numpy as np


def simulate_match(game: Game, agent1: Agent, agent2: Agent, num_games: int) -> None:
    """
    Play `num_games` games of agent1 vs agent2, alternating who moves first.
    Print aggregate stats, including agent1 wins as first vs second player.
    """
    wins1 = wins2 = draws = 0
    wins1_as_first = wins1_as_second = 0
    values_agent1: list[float] = []

    for g in range(int(num_games)):
        agent1.start()
        agent2.start()

        s = game.reset()

        # Alternate who agent1 is (+1 or -1)
        agent1_player = +1 if (g % 2 == 0) else -1
        agent2_player = -agent1_player

        while True:
            done, v = game.terminal_value(s)
            if done:
                # v is from player-to-move; convert to agent1 POV
                v1 = float(v) if game.to_play(s) == agent1_player else -float(v)
                values_agent1.append(v1)

                # Update aggregate stats
                if v1 > 0:
                    wins1 += 1
                    if agent1_player == +1:
                        wins1_as_first += 1
                    else:
                        wins1_as_second += 1
                    outcome_str = "Agent1 wins"
                elif v1 < 0:
                    wins2 += 1
                    outcome_str = "Agent2 wins"
                else:
                    draws += 1
                    outcome_str = "Draw"

                # Per-game print
                first = "Agent1" if agent1_player == +1 else "Agent2"
                second = "Agent2" if agent1_player == +1 else "Agent1"
                winner = (
                    "Agent1" if v1 > 0 else
                    "Agent2" if v1 < 0 else
                    "Draw"
                )
                print(f"[game {g:04d}] first={first} second={second} winner={winner} ({outcome_str})")

                break

            mover = agent1 if game.to_play(s) == agent1_player else agent2
            a = int(mover.act(s))

            legal = game.legal_actions(s)
            if not legal[a]:
                raise ValueError(f"Illegal action {a} for player {game.to_play(s)}.")

            s = game.next_state(s, a)

    n = max(1, int(num_games))
    mean_v = float(np.mean(values_agent1)) if values_agent1 else 0.0

    print("=== Match Results ===")
    print(f"Games played: {num_games}")
    print()
    print(f"Agent 1 total wins: {wins1} ({wins1 / n:.3f})")
    print(f"  ├─ as first player:  {wins1_as_first}")
    print(f"  └─ as second player: {wins1_as_second}")
    print(f"Agent 2 total wins: {wins2} ({wins2 / n:.3f})")
    print(f"Draws:              {draws} ({draws / n:.3f})")
    print()
    print(f"Mean value (Agent 1 POV): {mean_v:.3f}")