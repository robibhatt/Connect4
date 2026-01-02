from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.games.core.game import Game, State
from src.algorithms.alphazero.mcts import MCTS
from src.algorithms.alphazero.replay_buffer import ReplayBuffer
from src.agents.checkpointable import CheckpointableAgent
from src.agents.registry import AgentRegistry


@dataclass
class TrainerArgs:
    # Self-play
    iterations: int = 100
    games_per_iteration: int = 25
    temp_moves: int = 10          # number of moves with tau=1, then tau≈0
    tau: float = 1.0              # exploration temperature during temp_moves
    deterministic_after_temp: bool = True
    add_dirichlet_noise: bool = True

    # Training
    batch_size: int = 128
    train_steps_per_iteration: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_coef: float = 1.0

    # Buffer
    buffer_capacity: int = 100_000

    # Misc
    device: str = "mps"
    clear_mcts_each_game: bool = True


class Trainer:
    def __init__(self, game: Game, model: nn.Module, mcts: MCTS, args: TrainerArgs):
        self.game = game
        self.model = model
        self.mcts = mcts
        self.args = args

        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.buffer = ReplayBuffer(capacity=args.buffer_capacity)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -------------------------
    # Top-level loop
    # -------------------------

    def run(self):
        for it in range(self.args.iterations):
            self.self_play_phase()
            metrics = self.train_phase()
            print(f"[iter {it:04d}] buffer={len(self.buffer)} "
                  f"loss={metrics['loss']:.4f} "
                  f"policy={metrics['policy_loss']:.4f} "
                  f"value={metrics['value_loss']:.4f}")

    # -------------------------
    # Self-play
    # -------------------------

    def self_play_phase(self):
        game_counter = 0
        for _ in range(self.args.games_per_iteration):
            samples = self.play_one_game()
            game_counter += 1
            print('games played:', game_counter)
            # symmetry augmentation is handled by the buffer helper
            self.buffer.add_game(samples, augment_fn=self.game.symmetries)

    def play_one_game(self):
        """
        Returns a list of (x, pi, z) samples for one self-play game.
        IMPORTANT: x must already be game.encode(state) from player-to-move perspective.
        """
        if self.args.clear_mcts_each_game:
            self.mcts.clear()

        s = self.game.reset()
        history: list[tuple[np.ndarray, np.ndarray]] = []  # (x, pi)

        move_idx = 0
        while True:
            done, _ = self.game.terminal_value(s)
            if done:
                break

            # temperature schedule
            if move_idx < self.args.temp_moves:
                tau = self.args.tau
                deterministic = False
            else:
                tau = 1e-8  # effectively argmax in visits_to_policy
                deterministic = self.args.deterministic_after_temp

            pi = self.mcts.run(s, add_dirichlet_noise=self.args.add_dirichlet_noise, tau=tau)
            x = self.game.encode(s).astype(np.float32, copy=False)

            history.append((x, pi))

            a = self.mcts.select_action(pi, deterministic=deterministic)
            s = self.game.next_state(s, a)
            move_idx += 1

        # Label outcomes:
        # We avoid needing "winner" by propagating from terminal_value with sign flips.
        # terminal_value(s_terminal) returns value from viewpoint of player-to-move at terminal.
        done, v_terminal = self.game.terminal_value(s)
        assert done
        v = float(v_terminal)

        # For the last decision state, value is -v_terminal (because player-to-move flipped after last move).
        # Then keep flipping as we go backwards.
        samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for x, pi in reversed(history):
            v = -v
            z = v
            samples.append((x, pi, z))

        samples.reverse()
        return samples

    # -------------------------
    # Training
    # -------------------------

    def train_phase(self):
        # If buffer is tiny early on, just do what you can
        steps = self.args.train_steps_per_iteration
        if len(self.buffer) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        self.model.train()

        loss_acc = 0.0
        pol_acc = 0.0
        val_acc = 0.0

        for _ in range(steps):
            X, PI, Z = self.buffer.sample(self.args.batch_size)

            x = torch.from_numpy(X).to(self.device)                # (B, ...)
            pi_tgt = torch.from_numpy(PI).to(self.device)           # (B, A)
            z_tgt = torch.from_numpy(Z).to(self.device)             # (B,)

            logits, v_pred = self.model(x)                          # logits: (B, A), v_pred: (B,) or (B,1)
            v_pred = v_pred.squeeze(-1)

            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pi_tgt * logp).sum(dim=-1).mean()

            value_loss = F.mse_loss(v_pred, z_tgt)

            loss = policy_loss + self.args.value_loss_coef * value_loss

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            loss_acc += float(loss.item())
            pol_acc += float(policy_loss.item())
            val_acc += float(value_loss.item())

        inv = 1.0 / steps
        return {
            "loss": loss_acc * inv,
            "policy_loss": pol_acc * inv,
            "value_loss": val_acc * inv,
        }

    # -------------------------
    # Agent creation
    # -------------------------

    def create_agent(self) -> CheckpointableAgent:
        """
        Create agent from trained model using registry.

        This is called after training completes to package the
        trained model into a playable agent.

        Returns:
            Agent instance ready to be saved or used for play

        Raises:
            KeyError: If agent class not found in registry
        """
        game_class_name = self.game.__class__.__name__
        agent_class_name = f"{game_class_name}AlphaZeroAgent"

        # Get from registry
        AgentClass = AgentRegistry.get_agent(agent_class_name)

        # Construct agent
        return AgentClass(game=self.game, mcts=self.mcts)