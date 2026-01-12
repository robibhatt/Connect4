"""Test that simulate_match.py uses registries instead of hardcoded values."""
import pytest
from scripts.simulate_match import validate_agent_type, validate_game_exists
from src.algorithms.registry import AlgorithmRegistry
from src.games.core.registry import GameRegistry


class TestSimulateMatchUsesRegistries:
    """Tests that simulate_match uses registry-driven validation."""

    def test_valid_agent_types_from_registry(self):
        """validate_agent_type should accept all registered algorithms."""
        for algo in AlgorithmRegistry.get_all_algorithms():
            # Should not raise
            validate_agent_type(algo, agent_num=1)

    def test_invalid_agent_type_shows_registered_algorithms(self):
        """Error message should list algorithms from registry."""
        with pytest.raises(ValueError) as exc_info:
            validate_agent_type('nonexistent_algo', agent_num=1)

        error_msg = str(exc_info.value)
        # Should dynamically list registered algorithms, not hardcoded list
        for algo in AlgorithmRegistry.get_all_algorithms():
            assert algo in error_msg

    def test_valid_games_from_registry(self):
        """validate_game_exists should accept all registered games."""
        for game in GameRegistry.list_games():
            # Should not raise
            validate_game_exists(game)


class TestCheckpointValidationUsesMetadata:
    """Tests that checkpoint validation uses AlgorithmMetadata."""

    def test_checkpoint_required_uses_metadata(self):
        """Checkpoint requirement should come from AlgorithmMetadata."""
        # Random should not require checkpoint
        random_meta = AlgorithmRegistry.get_metadata('random')
        assert random_meta.requires_checkpoint is False

        # VanillaMCTS should require checkpoint
        vmcts_meta = AlgorithmRegistry.get_metadata('vanilla_mcts')
        assert vmcts_meta.requires_checkpoint is True

    def test_checkpoint_files_from_metadata(self):
        """Required checkpoint files should come from metadata."""
        vmcts_meta = AlgorithmRegistry.get_metadata('vanilla_mcts')
        assert 'model.pt' not in vmcts_meta.checkpoint_files
        assert 'agent.yaml' in vmcts_meta.checkpoint_files
