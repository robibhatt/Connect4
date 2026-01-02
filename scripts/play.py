from src.agents import RandomAgent, AlphaZeroMCTSAgent
from src.games.tiktaktoe import TicTacToe
from src.models.tiktaktoenet import TicTacToeNet
from src.games.game_play import simulate_match
from src.games.ui import TicTacToeUI, Connect4UI
from pathlib import Path


def main():

    game = TicTacToe()
    model_dir = Path('trained_models/20251215_144209_TicTacToeNet')

    # Create agents
    random_agent = RandomAgent(game=game)

    mcts_agent = AlphaZeroMCTSAgent.from_checkpoint(
        model_dir=model_dir,
        game=game,
        model_cls=TicTacToeNet,
        device='mps'
    )

    # Simulate matches: Random vs MCTS
    simulate_match(
        game=game,
        agent1=random_agent,
        agent2=mcts_agent,
        num_games=100
    )

    # Play interactively with new UI
    ui = TicTacToeUI(game=game, agent=mcts_agent, pause_seconds=0.4)
    ui.run()





if __name__ == '__main__':
    main()