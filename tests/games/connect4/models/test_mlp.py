"""Tests for Connect4MLPNet model."""

import pytest
import torch

from src.games.connect4.models.mlp import Connect4MLPNet
from src.models.base import GameNet
from src.models.registry import ModelRegistry


# ===== Test Fixtures =====

@pytest.fixture
def connect4_model():
    """Connect4MLPNet with default settings."""
    return Connect4MLPNet(hidden=128)


@pytest.fixture
def connect4_model_small():
    """Tiny Connect4MLPNet for faster tests."""
    return Connect4MLPNet(hidden=16)


@pytest.fixture
def sample_connect4_input():
    """Sample input tensor [B, 6, 7]."""
    return torch.randn(4, 6, 7)


# ===== Initialization Tests =====

def test_model_default_hidden_size():
    """Model should use default hidden size (128)."""
    model = Connect4MLPNet()
    assert model.hidden == 128


def test_model_custom_hidden_size():
    """Model should accept custom hidden size."""
    model = Connect4MLPNet(hidden=256)
    assert model.hidden == 256


def test_model_class_attributes():
    """Model should have correct class attributes."""
    assert Connect4MLPNet.game_name == "connect4"
    assert Connect4MLPNet.input_shape == (6, 7)
    assert Connect4MLPNet.action_size == 7


# ===== Forward Pass Shape Tests =====

def test_forward_output_shapes(connect4_model):
    """Forward pass should return correct shapes."""
    x = torch.randn(4, 6, 7)
    logits, value = connect4_model(x)

    assert logits.shape == (4, 7)
    assert value.shape == (4, 1)


def test_forward_output_types(connect4_model):
    """Forward pass should return float32 tensors."""
    x = torch.randn(4, 6, 7)
    logits, value = connect4_model(x)

    assert logits.dtype == torch.float32
    assert value.dtype == torch.float32


def test_forward_value_range(connect4_model):
    """Value should be in [-1, 1] range (due to tanh)."""
    x = torch.randn(4, 6, 7)
    _, value = connect4_model(x)

    assert (value >= -1).all()
    assert (value <= 1).all()


def test_forward_batch_dimension(connect4_model):
    """Forward pass should preserve batch dimension."""
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 6, 7)
        logits, value = connect4_model(x)

        assert logits.shape[0] == batch_size
        assert value.shape[0] == batch_size


# ===== Input Validation Tests =====

def test_forward_accepts_correct_input_shape(connect4_model):
    """Forward pass should accept [B, 6, 7] input."""
    x = torch.randn(4, 6, 7)
    connect4_model(x)  # Should not raise


def test_forward_raises_on_wrong_input_shape(connect4_model):
    """Forward pass should raise on wrong input shape."""
    x = torch.randn(4, 3, 3)  # Wrong shape (TicTacToe size)

    with pytest.raises(ValueError, match="Expected input shape"):
        connect4_model(x)


# ===== Integration Tests =====

def test_model_forward_doesnt_crash(connect4_model):
    """Basic smoke test - forward pass doesn't crash."""
    x = torch.randn(1, 6, 7)
    logits, value = connect4_model(x)

    assert logits is not None
    assert value is not None


def test_model_is_registered():
    """Model should be auto-registered in ModelRegistry."""
    model_class = ModelRegistry.get_model('Connect4MLPNet')
    assert model_class is Connect4MLPNet


def test_model_parameters_are_trainable(connect4_model):
    """Model should have trainable parameters."""
    params = list(connect4_model.parameters())

    assert len(params) > 0

    # All parameters should have requires_grad=True by default
    for param in params:
        assert param.requires_grad


def test_model_is_gamenet_subclass(connect4_model):
    """Model should be a GameNet subclass."""
    assert isinstance(connect4_model, GameNet)


# ===== Architecture Tests =====

def test_model_has_correct_layers():
    """Model should have expected layers."""
    model = Connect4MLPNet(hidden=128)

    assert hasattr(model, 'fc1')
    assert hasattr(model, 'fc2')
    assert hasattr(model, 'policy')
    assert hasattr(model, 'value')


def test_model_layer_sizes():
    """Model layers should have correct sizes."""
    hidden = 256
    model = Connect4MLPNet(hidden=hidden)

    assert model.fc1.in_features == 42  # 6*7 = 42
    assert model.fc1.out_features == hidden
    assert model.fc2.in_features == hidden
    assert model.fc2.out_features == hidden
    assert model.policy.in_features == hidden
    assert model.policy.out_features == 7
    assert model.value.in_features == hidden
    assert model.value.out_features == 1


# ===== Functional Tests =====

def test_model_with_different_hidden_sizes():
    """Model should work with various hidden sizes."""
    for hidden in [16, 32, 64, 128, 256]:
        model = Connect4MLPNet(hidden=hidden)
        x = torch.randn(2, 6, 7)
        logits, value = model(x)

        assert logits.shape == (2, 7)
        assert value.shape == (2, 1)


def test_model_gradient_flow():
    """Model should support gradient flow (basic check)."""
    model = Connect4MLPNet(hidden=32)
    x = torch.randn(2, 6, 7)
    logits, value = model(x)

    # Compute a simple loss
    loss = logits.sum() + value.sum()
    loss.backward()

    # Check that gradients exist
    for param in model.parameters():
        assert param.grad is not None


def test_model_with_zero_input(connect4_model):
    """Model should handle zero input (empty board)."""
    x = torch.zeros(1, 6, 7)
    logits, value = connect4_model(x)

    assert logits.shape == (1, 7)
    assert value.shape == (1, 1)
    assert (value >= -1).all() and (value <= 1).all()
