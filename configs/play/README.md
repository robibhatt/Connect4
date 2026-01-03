# Human Play Sample Configurations

Sample configurations for `scripts/human_play.py`.

## Usage

1. Copy desired config to scripts directory:
   ```bash
   cp configs/play/tictactoe_random.yaml scripts/human_play.yaml
   ```

2. For AlphaZero configs, edit checkpoint path:
   ```bash
   vim scripts/human_play.yaml
   # Update checkpoint_dir to your trained agent
   ```

3. Run game:
   ```bash
   python -m scripts.human_play
   ```

## Available Configs

- **tictactoe_random.yaml** - Play against random TicTacToe agent (no training needed)
- **tictactoe_alphazero.yaml** - Play against trained TicTacToe agent
- **connect4_random.yaml** - Play against random Connect4 agent (no training needed)
- **connect4_alphazero.yaml** - Play against trained Connect4 agent

## Notes

- Random agents work immediately - no training required
- AlphaZero agents require updating `checkpoint_dir` to point to your trained model
- Find checkpoints in: `saved_agents/`
- Adjust `pause_seconds` to control AI thinking display time
