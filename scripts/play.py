from pathlib import Path

from src.agents import RandomAgent, load_agent_checkpoint
from src.games.core.game_play import simulate_match
from src.games.tictactoe.ui import TicTacToeUI
from src.games.connect4.ui import Connect4UI


# UI registry (keep for now - UI registry is out of scope)
UIS = {
    'tictactoe': TicTacToeUI,
    'connect4': Connect4UI,
}


def main():
    # Configuration - point to saved agent directory
    checkpoint_dir = Path('saved_agents/20260102_143045_tictactoe_TicTacToeAlphaZeroAgent')

    # Load agent (fully self-contained!)
    agent = load_agent_checkpoint(checkpoint_dir)
    game = agent.game
    game_name = game.__class__.__name__.lower()

    print(f"\n{'='*60}")
    print(f"Playing {game_name} with {agent.__class__.__name__}")
    print(f"{'='*60}\n")

    # Create opponent
    random_agent = RandomAgent(game=game)

    # Simulate matches
    simulate_match(
        game=game,
        agent1=random_agent,
        agent2=agent,
        num_games=100
    )

    # Play interactively with UI
    ui_cls = UIS.get(game_name)
    if ui_cls:
        ui = ui_cls(game=game, agent=agent, pause_seconds=0.4)
        ui.run()
    else:
        print(f"No UI available for {game_name}")


if __name__ == '__main__':
    main()