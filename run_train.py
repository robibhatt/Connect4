from trainer import TrainerArgs, Trainer
from games.tiktaktoe import TicTacToe
from models.tiktaktoenet import TicTacToeNet
from mcts import MCTSConfig, MCTS
from evaluator import TorchEvaluator
from agent import RandomAgent, MCTSAgent
from games.game_play import simulate_match
import torch
from datetime import datetime
from pathlib import Path

def save_model_with_timestamp(model, root_dir="trained_models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.__class__.__name__
    folder_name = f"{timestamp}_{model_name}"

    save_dir = Path(root_dir) / folder_name
    save_dir.mkdir(parents=True, exist_ok=False)

    save_path = save_dir / "model.pt"
    torch.save(model.state_dict(), save_path)

    return save_dir


def main():

    args = TrainerArgs(
        iterations=400,
        games_per_iteration=8,
        temp_moves=4,
        tau=1.0,
        deterministic_after_temp=True,
        add_dirichlet_noise=True,
        batch_size=64,
        train_steps_per_iteration=50,
        lr=3e-3,
        weight_decay=1e-4,
        value_loss_coef=1.0,
        buffer_capacity=40_000,
        device="mps",
        clear_mcts_each_game=True,
    )

    game = TicTacToe()
    model = TicTacToeNet()
    evaluator = TorchEvaluator(model=model,
                               device=torch.device(args.device))
    

    mcts_cfg = MCTSConfig(
        num_sims=50,
        c_puct=1.25,
        dirichlet_alpha=0.6,   # higher alpha for small action space
        dirichlet_eps=0.20,    # slightly gentler than 0.25
        illegal_action_penalty=1e9,
    )

    mcts = MCTS(game=game,
                evaluator=evaluator,
                cfg=mcts_cfg)
    
    trainer = Trainer(
        game=game,
        model=model,
        mcts=mcts,
        args=args
    )

    trainer.run()

    save_model_with_timestamp(model)

    # testing code
    # random_agent = RandomAgent(game=game)
    # mcts_agent = MCTSAgent(game= game, mcts=mcts)
    # simulate_match(
    #     game=game,
    #     agent1=random_agent,
    #     agent2=mcts_agent,
    #     num_games=100
    # )


if __name__ == '__main__':
    main()






