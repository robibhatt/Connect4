"""Tests for shared TrainerArgs configuration."""

import pytest
from dataclasses import asdict

from src.algorithms.shared.trainer_args import TrainerArgs


class TestTrainerArgsFieldsAndDefaults:
    """Test that TrainerArgs has expected fields with correct defaults."""

    def test_has_num_test_games_field(self):
        args = TrainerArgs()
        assert hasattr(args, "num_test_games")
        assert args.num_test_games == 10

    def test_has_device_field(self):
        args = TrainerArgs()
        assert hasattr(args, "device")
        assert args.device == "cpu"

    def test_has_random_seed_field(self):
        args = TrainerArgs()
        assert hasattr(args, "random_seed")
        assert args.random_seed is None

    def test_has_verbose_field(self):
        args = TrainerArgs()
        assert hasattr(args, "verbose")
        assert args.verbose is True

    def test_can_override_defaults(self):
        args = TrainerArgs(
            num_test_games=20,
            device="cuda",
            random_seed=42,
            verbose=False,
        )
        assert args.num_test_games == 20
        assert args.device == "cuda"
        assert args.random_seed == 42
        assert args.verbose is False


class TestTrainerArgsSerialization:
    """Test TrainerArgs serialization to/from dict."""

    def test_to_dict(self):
        args = TrainerArgs(num_test_games=5, device="cuda", random_seed=123, verbose=False)
        d = args.to_dict()
        assert d == {
            "num_test_games": 5,
            "device": "cuda",
            "random_seed": 123,
            "verbose": False,
        }

    def test_from_dict(self):
        d = {
            "num_test_games": 15,
            "device": "mps",
            "random_seed": 999,
            "verbose": True,
        }
        args = TrainerArgs.from_dict(d)
        assert args.num_test_games == 15
        assert args.device == "mps"
        assert args.random_seed == 999
        assert args.verbose is True

    def test_roundtrip(self):
        original = TrainerArgs(num_test_games=7, device="cuda", random_seed=42, verbose=False)
        d = original.to_dict()
        restored = TrainerArgs.from_dict(d)
        assert restored == original
