# Training Sample Configurations

Sample configurations for `scripts/train.py`.

## Usage

1. Copy desired config to scripts directory:
   ```bash
   cp configs/train/tictactoe_quick.yaml scripts/train.yaml
   ```

2. Edit if needed:
   ```bash
   vim scripts/train.yaml
   ```

3. Run training:
   ```bash
   python -m scripts.train
   ```

## Available Configs

- **tictactoe_quick.yaml** - Fast training for testing (100 iterations)
- **tictactoe_full.yaml** - Full training for good performance (400 iterations)
- **connect4_quick.yaml** - Fast Connect4 training (50 iterations)
- **connect4_full.yaml** - Full Connect4 training (300 iterations)

## Device Options

Update the `device` field based on your hardware:
- `cpu` - CPU only (slow but works everywhere)
- `cuda` - NVIDIA GPU (fast, requires CUDA)
- `mps` - Apple Silicon GPU (fast on M1/M2/M3 Macs)
