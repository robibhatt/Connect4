"""Tests for TicTacToeMLPNet model."""

import pytest
import torch

from src.games.tictactoe.models.mlp import TicTacToeMLPNet
from src.models.base import GameNet
from src.models.registry import ModelRegistry


# ===== Test Fixtures =====

@pytest.fixture
def tictactoe_model():
    """TicTacToeMLPNet with default settings."""
    return TicTacToeMLPNet(hidden=64)


@pytest.fixture
def tictactoe_model_small():
    """Tiny TicTacToeMLPNet for faster tests."""
    return TicTacToeMLPNet(hidden=8)


@pytest.fixture
def sample_tictactoe_input():
    """Sample input tensor [B, 3, 3]."""
    return torch.randn(4, 3, 3)


# ===== Initialization Tests =====

def test_model_default_hidden_size():
    """Model should use default hidden size (64)."""
    model = TicTacToeMLPNet()
    assert model.hidden == 64


def test_model_custom_hidden_size():
    """Model should accept custom hidden size."""
    model = TicTacToeMLPNet(hidden=128)
    assert model.hidden == 128


def test_model_class_attributes():
    """Model should have correct class attributes."""
    assert TicTacToeMLPNet.game_name == "tictactoe"
    assert TicTacToeMLPNet.input_shape == (3, 3)
    assert TicTacToeMLPNet.action_size == 9


# ===== Forward Pass Shape Tests =====

def test_forward_output_shapes(tictactoe_model):
    """Forward pass should return correct shapes."""
    x = torch.randn(4, 3, 3)
    logits, value = tictactoe_model(x)

    assert logits.shape == (4, 9)
    assert value.shape == (4, 1)


def test_forward_output_types(tictactoe_model):
    """Forward pass should return float32 tensors."""
    x = torch.randn(4, 3, 3)
    logits, value = tictactoe_model(x)

    assert logits.dtype == torch.float32
    assert value.dtype == torch.float32


def test_forward_value_range(tictactoe_model):
    """Value should be in [-1, 1] range (due to tanh)."""
    x = torch.randn(4, 3, 3)
    _, value = tictactoe_model(x)

    assert (value >= -1).all()
    assert (value <= 1).all()


def test_forward_batch_dimension(tictactoe_model):
    """Forward pass should preserve batch dimension."""
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 3, 3)
        logits, value = tictactoe_model(x)

        assert logits.shape[0] == batch_size
        assert value.shape[0] == batch_size


# ===== Input Validation Tests =====

def test_forward_accepts_correct_input_shape(tictactoe_model):
    """Forward pass should accept [B, 3, 3] input."""
    x = torch.randn(4, 3, 3)
    tictactoe_model(x)  # Should not raise


def test_forward_raises_on_wrong_input_shape(tictactoe_model):
    """Forward pass should raise on wrong input shape."""
    x = torch.randn(4, 2, 2)  # Wrong shape

    with pytest.raises(ValueError, match="Expected input shape"):
        tictactoe_model(x)


# ===== Integration Tests =====

def test_model_forward_doesnt_crash(tictactoe_model):
    """Basic smoke test - forward pass doesn't crash."""
    x = torch.randn(1, 3, 3)
    logits, value = tictactoe_model(x)

    assert logits is not None
    assert value is not None


def test_model_is_registered():
    """Model should be auto-registered in ModelRegistry."""
    model_class = ModelRegistry.get_model('TicTacToeMLPNet')
    assert model_class is TicTacToeMLPNet


def test_model_parameters_are_trainable(tictactoe_model):
    """Model should have trainable parameters."""
    params = list(tictactoe_model.parameters())

    assert len(params) > 0

    # All parameters should have requires_grad=True by default
    for param in params:
        assert param.requires_grad


def test_model_is_gamenet_subclass(tictactoe_model):
    """Model should be a GameNet subclass."""
    assert isinstance(tictactoe_model, GameNet)


# ===== Architecture Tests =====

def test_model_has_correct_layers():
    """Model should have expected layers."""
    model = TicTacToeMLPNet(hidden=64)

    assert hasattr(model, 'fc1')
    assert hasattr(model, 'fc2')
    assert hasattr(model, 'policy')
    assert hasattr(model, 'value')


def test_model_layer_sizes():
    """Model layers should have correct sizes."""
    hidden = 128
    model = TicTacToeMLPNet(hidden=hidden)

    assert model.fc1.out_features == hidden
    assert model.fc2.in_features == hidden
    assert model.fc2.out_features == hidden
    assert model.policy.in_features == hidden
    assert model.policy.out_features == 9
    assert model.value.in_features == hidden
    assert model.value.out_features == 1


# ===== Functional Tests =====

def test_model_with_different_hidden_sizes():
    """Model should work with various hidden sizes."""
    for hidden in [8, 16, 32, 64, 128]:
        model = TicTacToeMLPNet(hidden=hidden)
        x = torch.randn(2, 3, 3)
        logits, value = model(x)

        assert logits.shape == (2, 9)
        assert value.shape == (2, 1)


def test_model_gradient_flow():
    """Model should support gradient flow (basic check)."""
    model = TicTacToeMLPNet(hidden=16)
    x = torch.randn(2, 3, 3)
    logits, value = model(x)

    # Compute a simple loss
    loss = logits.sum() + value.sum()
    loss.backward()

    # Check that gradients exist
    for param in model.parameters():
        assert param.grad is not None


def test_model_with_zero_input(tictactoe_model):
    """Model should handle zero input (empty board)."""
    x = torch.zeros(1, 3, 3)
    logits, value = tictactoe_model(x)

    assert logits.shape == (1, 9)
    assert value.shape == (1, 1)
    assert (value >= -1).all() and (value <= 1).all()
