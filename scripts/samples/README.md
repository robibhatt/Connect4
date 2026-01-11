# Sample YAML Configurations

Copy any of these to the corresponding script location and run.

## Training (`train_*.yaml`)

Copy to `scripts/train.yaml`, then run `python scripts/train.py`

| File | Description |
|------|-------------|
| `train_tictactoe_alphazero_quick.yaml` | Fast AlphaZero training (~5 min) |
| `train_tictactoe_alphazero_full.yaml` | Full AlphaZero training (~30-60 min) |
| `train_connect4_alphazero_quick.yaml` | Fast Connect4 training (~10-15 min) |
| `train_connect4_alphazero_full.yaml` | Full Connect4 training (~2-4 hours) |
| `train_tictactoe_vanilla_mcts.yaml` | Vanilla MCTS validation |
| `train_connect4_vanilla_mcts.yaml` | Vanilla MCTS validation |

## Match Simulation (`match_*.yaml`)

Copy to `scripts/simulate_match.yaml`, then run `python scripts/simulate_match.py`

| File | Description |
|------|-------------|
| `match_tictactoe_alphazero_vs_random.yaml` | Test trained agent vs random |
| `match_tictactoe_alphazero_vs_alphazero.yaml` | Compare two trained agents |
| `match_connect4_alphazero_vs_random.yaml` | Connect4 agent vs random |
| `match_random_vs_random.yaml` | Sanity check (~50/50 results) |

## Human Play (`play_*.yaml`)

Copy to `scripts/human_play.yaml`, then run `python scripts/human_play.py`

| File | Description |
|------|-------------|
| `play_tictactoe_vs_alphazero.yaml` | Play against your trained agent |
| `play_tictactoe_vs_random.yaml` | Easy mode |
| `play_connect4_vs_alphazero.yaml` | Play Connect4 vs trained agent |
| `play_connect4_vs_random.yaml` | Easy mode |

## Quick Start

```bash
# Train a quick TicTacToe agent
cp scripts/samples/train_tictactoe_alphazero_quick.yaml scripts/train.yaml
python scripts/train.py

# Test it against random
# (update checkpoint_dir in the yaml first!)
cp scripts/samples/match_tictactoe_alphazero_vs_random.yaml scripts/simulate_match.yaml
python scripts/simulate_match.py

# Play against it yourself
cp scripts/samples/play_tictactoe_vs_alphazero.yaml scripts/human_play.yaml
python scripts/human_play.py
```
