"""
Integration tests for new training configuration system.

Tests the full flow: YAML -> AlphaZeroConfig -> Trainer creation
"""

import pytest
import yaml
import tempfile
import torch
from pathlib import Path

from src.algorithms.registry import AlgorithmRegistry
from src.algorithms.alphazero.config import AlphaZeroConfig
from src.algorithms.alphazero import Trainer, MCTS
from src.games.core.registry import GameRegistry
from src.models.registry import ModelRegistry


# ===== YAML Parsing Tests =====

def test_load_new_yaml_format():
    """Test loading new algorithm-based YAML format"""
    yaml_content = """
game: tictactoe

algorithm:
  name: alphazero
  device: cpu
  model:
    class: TicTacToeMLPNet
    hidden: 64
  iterations: 400
  num_sims: 50
  games_per_iteration: 8
  c_puct: 1.25
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_path = Path(f.name)

    try:
        # Parse YAML
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)

        # Extract game name
        game_name = config_dict.get("game")
        assert game_name == 'tictactoe'

        # Extract algorithm config
        algo_config = config_dict.get("algorithm", {})
        algo_name = algo_config.get("name")
        assert algo_name == 'alphazero'

        # Get config class from registry
        ConfigClass = AlgorithmRegistry.get_config_class(algo_name)
        assert ConfigClass == AlphaZeroConfig

        # Extract model config
        model_config = algo_config.get("model", {})
        model_class_name = model_config.get('class')
        model_kwargs = {k: v for k, v in model_config.items() if k != 'class'}

        # Build config params
        config_params = {
            'model_class': model_class_name,
            'model_kwargs': model_kwargs,
            **{k: v for k, v in algo_config.items() if k not in ['name', 'model']}
        }

        # Instantiate config
        config = ConfigClass(**config_params)

        assert isinstance(config, AlphaZeroConfig)
        assert config.model_class == 'TicTacToeMLPNet'
        assert config.iterations == 400
        assert config.num_sims == 50

    finally:
        config_path.unlink()


def test_algorithm_not_found_raises_error():
    """Test that unknown algorithm raises helpful error"""
    yaml_content = """
game: tictactoe
algorithm:
  name: unknown_algorithm
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        config_path = Path(f.name)

    try:
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)

        algo_name = config_dict.get("algorithm", {}).get("name")

        with pytest.raises(KeyError, match="unknown_algorithm"):
            AlgorithmRegistry.get_config_class(algo_name)
    finally:
        config_path.unlink()


# ===== Trainer Factory Tests =====

def test_trainer_factory_creates_trainer():
    """Test that factory creates working Trainer instance"""
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={'hidden': 64},
        device='cpu',
        iterations=10,
        games_per_iteration=2,
        num_sims=5,
    )

    # Get game and model
    GameClass = GameRegistry.get_game('tictactoe')
    game = GameClass()

    ModelClass = ModelRegistry.get_model('TicTacToeMLPNet')
    model = ModelClass(hidden=64)

    # Get factory from registry
    factory = AlgorithmRegistry.get_trainer_factory('alphazero')

    # Create trainer using factory
    trainer = factory(game, model, config)

    # Validate trainer
    assert isinstance(trainer, Trainer)
    assert hasattr(trainer, 'run')
    assert trainer.args.iterations == 10
    assert trainer.args.games_per_iteration == 2


# ===== Backward Compatibility Tests =====

def test_config_works_with_existing_trainer():
    """Test that AlphaZeroConfig works with existing Trainer class"""
    config = AlphaZeroConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={'hidden': 64},
        device='cpu',
        iterations=10,
        games_per_iteration=2,
        num_sims=5,
    )

    # Extract legacy configs
    trainer_args = config.to_trainer_args()
    mcts_config = config.to_mcts_config()

    # Get game and model
    GameClass = GameRegistry.get_game('tictactoe')
    game = GameClass()

    ModelClass = ModelRegistry.get_model('TicTacToeMLPNet')
    model = ModelClass(hidden=64)

    # Create MCTS and Trainer directly (without factory)
    device = torch.device('cpu')
    mcts = MCTS(game, model, device=device, cfg=mcts_config)
    trainer = Trainer(game, model, mcts, trainer_args)

    # Validate
    assert trainer.args.iterations == 10
    assert mcts.cfg.num_sims == 5
