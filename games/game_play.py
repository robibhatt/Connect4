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
        agent1_moves_first = (g % 2 == 0)

        # Track whose perspective the canonical state represents: +1 for agent1, -1 for agent2.
        player_to_move = +1 if agent1_moves_first else -1

        while True:
            done, v = game.terminal_value(s)
            if done:
                # v is from player-to-move; convert to agent1 POV
                v1 = float(v) if player_to_move == +1 else -float(v)
                values_agent1.append(v1)

                if v1 > 0:
                    wins1 += 1
                    if agent1_moves_first:
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

                first = "Agent1" if agent1_moves_first else "Agent2"
                second = "Agent2" if agent1_moves_first else "Agent1"
                winner = "Agent1" if v1 > 0 else "Agent2" if v1 < 0 else "Draw"
                print(f"[game {g:04d}] first={first} second={second} winner={winner} ({outcome_str})")

                break

            mover = agent1 if player_to_move == +1 else agent2
            a = int(mover.act(s))

            legal = game.legal_actions(s)
            if not legal[a]:
                raise ValueError(f"Illegal action {a} for player {player_to_move}.")

            s = game.next_state(s, a)
            player_to_move *= -1

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
