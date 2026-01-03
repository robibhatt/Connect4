# Match Simulation Sample Configurations

Sample configurations for `scripts/simulate_match.py`.

## Usage

1. Copy desired config to scripts directory:
   ```bash
   cp configs/match/tictactoe_random_vs_random.yaml scripts/simulate_match.yaml
   ```

2. For AlphaZero agents, edit checkpoint paths:
   ```bash
   vim scripts/simulate_match.yaml
   # Update checkpoint_dir for agent1 and/or agent2
   ```

3. Run match simulation:
   ```bash
   python -m scripts.simulate_match
   ```

## Available Configs

### TicTacToe
- **tictactoe_random_vs_random.yaml** - Two random agents (works immediately)
- **tictactoe_alphazero_vs_random.yaml** - Trained vs random (benchmark your agent)
- **tictactoe_alphazero_vs_alphazero.yaml** - Two trained agents (compare models)

### Connect4
- **connect4_random_vs_random.yaml** - Two random agents (works immediately)
- **connect4_alphazero_vs_random.yaml** - Trained vs random (benchmark your agent)
- **connect4_alphazero_vs_alphazero.yaml** - Two trained agents (compare models)

## Configuration Options

- **num_games**: Number of games to simulate (higher = more accurate statistics)
- **agent1/agent2 type**: Either `random` or `alphazero`
- **checkpoint_dir**: Required for AlphaZero agents, points to trained model directory

## Output

The script shows:
- Game-by-game results with who went first and winner
- Aggregate statistics:
  - Total wins per agent (with percentages)
  - Wins as first vs second player
  - Draw count
  - Mean value from Agent 1's perspective
