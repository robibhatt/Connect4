"""Shared contract tests for Game.key() method.

All Game subclasses must return np.int64 keys that satisfy these requirements.
"""
import pytest
import numpy as np


# ===== Type Contract Tests =====

def test_key_returns_int64_type_connect4(connect4_game):
    """key() must return np.int64 type for Connect4."""
    state = connect4_game.reset()
    key = connect4_game.key(state)
    assert isinstance(key, np.int64), f"Expected np.int64, got {type(key)}"


def test_key_returns_int64_type_tictactoe(tictactoe_game):
    """key() must return np.int64 type for TicTacToe."""
    state = tictactoe_game.reset()
    key = tictactoe_game.key(state)
    assert isinstance(key, np.int64), f"Expected np.int64, got {type(key)}"


# ===== Range Tests =====

def test_key_returns_positive_value_connect4(connect4_game):
    """key() must return positive value (fits in signed int64) for Connect4."""
    state = connect4_game.reset()
    key = connect4_game.key(state)
    assert key >= 0, f"Key must be non-negative, got {key}"
    assert key <= np.iinfo(np.int64).max, f"Key exceeds int64 max"


def test_key_returns_positive_value_tictactoe(tictactoe_game):
    """key() must return positive value (fits in signed int64) for TicTacToe."""
    state = tictactoe_game.reset()
    key = tictactoe_game.key(state)
    assert key >= 0, f"Key must be non-negative, got {key}"
    assert key <= np.iinfo(np.int64).max, f"Key exceeds int64 max"


# ===== Dictionary Compatibility Tests =====

def test_key_usable_as_dict_key_connect4(connect4_game):
    """key() result must work as Python dictionary key for Connect4."""
    state = connect4_game.reset()
    key = connect4_game.key(state)

    table = {key: "test_value"}
    assert table[key] == "test_value"
    assert key in table


def test_key_usable_as_dict_key_tictactoe(tictactoe_game):
    """key() result must work as Python dictionary key for TicTacToe."""
    state = tictactoe_game.reset()
    key = tictactoe_game.key(state)

    table = {key: "test_value"}
    assert table[key] == "test_value"
    assert key in table


# ===== NumPy Array Compatibility Tests =====

def test_key_storable_in_int64_array_connect4(connect4_game):
    """key() result must be storable in np.int64 array for Connect4."""
    states = [connect4_game.reset()]
    for i in range(3):
        states.append(connect4_game.next_state(states[-1], i))

    keys = np.array([connect4_game.key(s) for s in states], dtype=np.int64)
    assert keys.dtype == np.int64
    assert len(keys) == 4


def test_key_storable_in_int64_array_tictactoe(tictactoe_game):
    """key() result must be storable in np.int64 array for TicTacToe."""
    states = [tictactoe_game.reset()]
    for i in range(3):
        states.append(tictactoe_game.next_state(states[-1], i))

    keys = np.array([tictactoe_game.key(s) for s in states], dtype=np.int64)
    assert keys.dtype == np.int64
    assert len(keys) == 4


# ===== Determinism Tests =====

def test_key_deterministic_connect4(connect4_game):
    """key() should be deterministic (no randomness between calls) for Connect4."""
    state = connect4_game.reset()
    keys = [connect4_game.key(state) for _ in range(100)]
    assert all(k == keys[0] for k in keys), "Keys should be identical for same state"


def test_key_deterministic_tictactoe(tictactoe_game):
    """key() should be deterministic (no randomness between calls) for TicTacToe."""
    state = tictactoe_game.reset()
    keys = [tictactoe_game.key(state) for _ in range(100)]
    assert all(k == keys[0] for k in keys), "Keys should be identical for same state"
