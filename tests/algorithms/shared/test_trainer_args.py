"""Tests for shared TrainerArgs configuration."""

import pytest

from src.algorithms.shared.trainer_args import TrainerArgs


class TestTrainerArgs:
    """Test TrainerArgs fields, defaults, and serialization."""

    def test_defaults_and_override(self):
        """Test all fields have correct defaults and can be overridden."""
        # Check defaults
        default_args = TrainerArgs()
        assert default_args.num_test_games == 10
        assert default_args.device == "cpu"
        assert default_args.random_seed is None
        assert default_args.verbose is True

        # Check override
        custom_args = TrainerArgs(
            num_test_games=20,
            device="cuda",
            random_seed=42,
            verbose=False,
        )
        assert custom_args.num_test_games == 20
        assert custom_args.device == "cuda"
        assert custom_args.random_seed == 42
        assert custom_args.verbose is False

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict produce equivalent objects."""
        original = TrainerArgs(num_test_games=7, device="cuda", random_seed=42, verbose=False)
        d = original.to_dict()
        restored = TrainerArgs.from_dict(d)
        assert restored == original
