"""Tests for GameNet base class."""

import pytest
import torch
import torch.nn as nn

from src.models.base import GameNet


# ===== Test Fixtures =====

@pytest.fixture
def concrete_gamenet():
    """Minimal concrete GameNet for testing."""
    class TestGameNet(GameNet):
        game_name = "test"
        input_shape = (3, 3)
        action_size = 9

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(9, 10)

        def forward(self, x):
            B = x.shape[0]
            logits = torch.zeros(B, 9)
            value = torch.zeros(B, 1)
            return logits, value

    return TestGameNet()


@pytest.fixture
def gamenet_no_validation():
    """GameNet with None attributes (no validation)."""
    class NoValidationNet(GameNet):
        game_name = None
        input_shape = None
        action_size = None

        def forward(self, x):
            B = x.shape[0]
            return torch.zeros(B, 5), torch.zeros(B, 1)

    return NoValidationNet()


# ===== Abstract Base Class Tests =====

def test_cannot_call_forward_on_gamenet_directly():
    """GameNet.forward() raises NotImplementedError."""
    # GameNet can be instantiated but forward() raises NotImplementedError
    net = GameNet()
    x = torch.randn(1, 3, 3)

    with pytest.raises(NotImplementedError):
        net.forward(x)


def test_subclass_must_implement_forward():
    """Subclass without forward() raises NotImplementedError."""
    class IncompleteGameNet(GameNet):
        game_name = "incomplete"
        input_shape = (3, 3)
        action_size = 9
        # Missing forward() implementation - will use base class version

    net = IncompleteGameNet()
    x = torch.randn(1, 3, 3)

    with pytest.raises(NotImplementedError):
        net.forward(x)


# ===== Class Attribute Tests =====

def test_default_attributes_are_none():
    """game_name, input_shape, action_size default to None."""
    class MinimalNet(GameNet):
        def forward(self, x):
            return torch.zeros(1, 1), torch.zeros(1, 1)

    # Class attributes should default to None
    assert MinimalNet.game_name is None
    assert MinimalNet.input_shape is None
    assert MinimalNet.action_size is None


def test_subclass_can_override_attributes(concrete_gamenet):
    """Subclass can override class attributes."""
    assert concrete_gamenet.game_name == "test"
    assert concrete_gamenet.input_shape == (3, 3)
    assert concrete_gamenet.action_size == 9


# ===== Input Validation Tests =====

def test_validate_input_passes_correct_shape(concrete_gamenet):
    """validate_input() should pass for correct shape."""
    x = torch.randn(4, 3, 3)  # Batch of 4
    concrete_gamenet.validate_input(x)  # Should not raise


def test_validate_input_raises_wrong_shape(concrete_gamenet):
    """validate_input() should raise for wrong shape."""
    x = torch.randn(4, 2, 2)  # Wrong shape
    with pytest.raises(ValueError, match="Expected input shape"):
        concrete_gamenet.validate_input(x)


def test_validate_input_skips_if_input_shape_none(gamenet_no_validation):
    """validate_input() should skip if input_shape is None."""
    x = torch.randn(4, 5, 5)  # Any shape should be ok
    gamenet_no_validation.validate_input(x)  # Should not raise


def test_validate_input_batch_dimension_flexible(concrete_gamenet):
    """validate_input() should work with different batch sizes."""
    for batch_size in [1, 2, 8, 16]:
        x = torch.randn(batch_size, 3, 3)
        concrete_gamenet.validate_input(x)  # Should not raise


# ===== Output Validation Tests =====

def test_validate_output_passes_correct_shapes(concrete_gamenet):
    """validate_output() should pass for correct shapes."""
    logits = torch.randn(4, 9)
    value = torch.randn(4, 1)
    concrete_gamenet.validate_output(logits, value)  # Should not raise


def test_validate_output_raises_wrong_logits_shape(concrete_gamenet):
    """validate_output() should raise for wrong logits shape."""
    logits = torch.randn(4, 7)  # Wrong action_size
    value = torch.randn(4, 1)
    with pytest.raises(ValueError, match="Expected logits shape"):
        concrete_gamenet.validate_output(logits, value)


def test_validate_output_raises_wrong_value_shape(concrete_gamenet):
    """validate_output() should raise for wrong value shape."""
    logits = torch.randn(4, 9)
    value = torch.randn(4, 3)  # Wrong shape
    with pytest.raises(ValueError, match="Expected value shape"):
        concrete_gamenet.validate_output(logits, value)


def test_validate_output_accepts_1d_value(concrete_gamenet):
    """validate_output() should accept both (B, 1) and (B,) for value."""
    logits = torch.randn(4, 9)

    # Test (B, 1) shape
    value_2d = torch.randn(4, 1)
    concrete_gamenet.validate_output(logits, value_2d)  # Should not raise

    # Test (B,) shape
    value_1d = torch.randn(4)
    concrete_gamenet.validate_output(logits, value_1d)  # Should not raise


def test_validate_output_skips_if_action_size_none(gamenet_no_validation):
    """validate_output() should skip if action_size is None."""
    logits = torch.randn(4, 7)  # Any action size
    value = torch.randn(4, 1)
    gamenet_no_validation.validate_output(logits, value)  # Should not raise


# ===== Integration Tests =====

def test_concrete_gamenet_forward_pass(concrete_gamenet):
    """Concrete GameNet should have working forward pass."""
    x = torch.randn(4, 3, 3)
    logits, value = concrete_gamenet(x)

    assert logits.shape == (4, 9)
    assert value.shape == (4, 1)


def test_gamenet_is_nn_module(concrete_gamenet):
    """GameNet should be a PyTorch nn.Module."""
    assert isinstance(concrete_gamenet, nn.Module)


def test_gamenet_has_parameters(concrete_gamenet):
    """GameNet should have trainable parameters."""
    params = list(concrete_gamenet.parameters())
    assert len(params) > 0


def test_multiple_gamenet_subclasses():
    """Can create multiple different GameNet subclasses."""
    class Net1(GameNet):
        game_name = "game1"
        input_shape = (3, 3)
        action_size = 9

        def forward(self, x):
            return torch.zeros(x.shape[0], 9), torch.zeros(x.shape[0], 1)

    class Net2(GameNet):
        game_name = "game2"
        input_shape = (6, 7)
        action_size = 7

        def forward(self, x):
            return torch.zeros(x.shape[0], 7), torch.zeros(x.shape[0], 1)

    net1 = Net1()
    net2 = Net2()

    assert net1.game_name == "game1"
    assert net2.game_name == "game2"
    assert net1.action_size == 9
    assert net2.action_size == 7
