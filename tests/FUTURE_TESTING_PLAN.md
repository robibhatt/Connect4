# Future Testing Plan - Phases 4-5

This document contains the detailed plan for Phase 4 (Agent & Persistence Testing) and Phase 5 (Model Testing), to be implemented after Phase 1-3 are complete.

**Status:** Not yet implemented
**Prerequisites:** Phase 1-3 must be complete (foundation, game logic, registries)

---

## Phase 4: Agent & Persistence Testing

**Goal:** Test agent implementations and checkpoint save/load functionality

### 4.1 Agent Fixtures

**File:** `tests/agents/conftest.py`

Create agent-specific shared fixtures:

```python
"""Pytest fixtures for agent tests."""
import pytest
import torch
import numpy as np
from unittest.mock import Mock

from src.agents.agent import RandomAgent
from src.agents.alphazero_agent_config import AlphaZeroAgentConfig


@pytest.fixture
def random_agent_tictactoe(tictactoe_game, numpy_rng):
    """Deterministic RandomAgent for TicTacToe."""
    return RandomAgent(tictactoe_game, numpy_rng)


@pytest.fixture
def random_agent_connect4(connect4_game, numpy_rng):
    """Deterministic RandomAgent for Connect4."""
    return RandomAgent(connect4_game, numpy_rng)


@pytest.fixture
def mock_mcts():
    """Mock MCTS for agent testing."""
    mcts = Mock()
    mcts.play_move = Mock(return_value=0)
    mcts.clear = Mock()
    mcts.model = Mock(spec=torch.nn.Module)
    return mcts


@pytest.fixture
def sample_agent_config():
    """Sample AlphaZeroAgentConfig for testing."""
    return AlphaZeroAgentConfig(
        model_class='TicTacToeMLPNet',
        model_kwargs={'hidden': 64},
        num_sims=50,
        c_puct=1.25,
        device='cpu'
    )
```

---

### 4.2 Test RandomAgent

**File:** `tests/agents/test_agent.py`
**Tests:** `src/agents/agent.py`

**Key test cases:**

1. **Agent ABC**
   - Test cannot instantiate directly (abstract class)
   - Test subclass must implement `act()`
   - Test `game` attribute is stored correctly

2. **RandomAgent - Basic Behavior**
   ```python
   def test_random_agent_returns_legal_action(random_agent_tictactoe, tictactoe_game):
       """RandomAgent should only return legal actions."""
       state = tictactoe_game.reset()
       action = random_agent_tictactoe.act(state)
       legal_actions = tictactoe_game.legal_actions(state)
       assert legal_actions[action], "Action should be legal"

   def test_random_agent_distribution_is_uniform(random_agent_tictactoe, tictactoe_game):
       """RandomAgent should sample uniformly over many trials."""
       state = tictactoe_game.reset()
       actions = [random_agent_tictactoe.act(state) for _ in range(1000)]
       # Check roughly uniform distribution (chi-square test or simple histogram)
   ```

3. **RandomAgent - Edge Cases**
   ```python
   def test_random_agent_with_single_legal_action():
       """With one legal action, RandomAgent should be deterministic."""
       # Create state with only one legal action
       # Verify agent always returns that action

   def test_random_agent_raises_on_no_legal_actions():
       """RandomAgent should raise RuntimeError when no legal actions."""
       # Mock game.legal_actions to return all False
       # Verify RuntimeError is raised
   ```

4. **RandomAgent - RNG Control**
   ```python
   def test_random_agent_with_custom_rng():
       """RandomAgent should use provided RNG for reproducibility."""
       rng1 = np.random.default_rng(42)
       rng2 = np.random.default_rng(42)
       agent1 = RandomAgent(game, rng1)
       agent2 = RandomAgent(game, rng2)
       # Verify both agents produce same sequence
   ```

**Estimated:** ~8-10 test functions

---

### 4.3 Test Checkpoint Utilities

**File:** `tests/agents/test_checkpoint_utils.py`
**Tests:** `src/agents/checkpoint_utils.py`

**Key test cases:**

1. **Save Checkpoint**
   ```python
   def test_save_creates_directory_with_timestamp(temp_checkpoint_dir):
       """save_agent_checkpoint should create timestamped directory."""
       # Save agent, verify directory name format

   def test_save_creates_model_pt(temp_checkpoint_dir, sample_agent):
       """save_agent_checkpoint should save model.pt via to_checkpoint()."""
       # Save agent, verify model.pt exists and is loadable

   def test_save_creates_agent_yaml(temp_checkpoint_dir, sample_agent):
       """save_agent_checkpoint should save agent.yaml with metadata."""
       # Save agent, verify YAML contains: game, agent_class, mcts_config, etc.

   def test_save_includes_training_config_if_provided(temp_checkpoint_dir):
       """training_config should be saved if provided."""
       # Save with training_config, verify it's in YAML
   ```

2. **Load Checkpoint**
   ```python
   def test_load_reconstructs_agent(checkpoint_dir_with_agent):
       """load_agent_checkpoint should reconstruct agent from files."""
       # Load agent, verify it's the correct class

   def test_load_restores_model_weights(checkpoint_dir_with_agent):
       """Loaded agent should have same model weights."""
       # Compare model state_dict before/after save-load

   def test_load_with_device_override(checkpoint_dir_with_agent):
       """Device can be overridden during load."""
       # Load with device='cpu', verify agent uses CPU

   def test_load_sets_eval_mode(checkpoint_dir_with_agent):
       """Loaded agent should be in eval mode."""
       # Verify model.training == False
   ```

3. **Round-Trip Consistency**
   ```python
   def test_save_load_roundtrip(temp_checkpoint_dir, sample_agent):
       """Agent should behave identically after save-load."""
       # Save agent, load it, verify both produce same actions
   ```

4. **Error Handling**
   ```python
   def test_load_raises_on_missing_directory():
       """load should raise FileNotFoundError for missing directory."""

   def test_load_raises_on_missing_yaml():
       """load should raise FileNotFoundError for missing agent.yaml."""

   def test_load_raises_on_unregistered_agent():
       """load should raise KeyError for unregistered agent class."""
   ```

**Fixtures needed:**
```python
@pytest.fixture
def sample_agent(tictactoe_game):
    """Create a simple trained agent for testing."""
    # Return TicTacToeAlphaZeroAgent with minimal setup

@pytest.fixture
def checkpoint_dir_with_agent(tmp_path, sample_agent):
    """Create valid checkpoint structure."""
    # Save sample_agent, return directory path
```

**Estimated:** ~12-15 test functions

---

### 4.4 Test AlphaZero Agent Config

**File:** `tests/agents/test_alphazero_agent_config.py`
**Tests:** `src/agents/alphazero_agent_config.py`

**Key test cases:**

1. **Config Construction**
   ```python
   def test_create_minimal_config():
       """Can create config with minimal required fields."""
       config = AlphaZeroAgentConfig(...)
       assert config is not None

   def test_create_full_config():
       """Can create config with all fields specified."""
       # Create config with all optional fields

   def test_config_defaults():
       """Default values are applied correctly."""
       # Verify defaults for num_sims, c_puct, etc.
   ```

2. **Serialization**
   ```python
   def test_config_to_dict():
       """Config can be serialized to dict."""
       # Create config, convert to dict, verify fields

   def test_config_from_dict():
       """Config can be deserialized from dict."""
       # Create dict, load config, verify values

   def test_config_roundtrip():
       """Config survives dict roundtrip."""
       # Create config, to_dict, from_dict, compare
   ```

**Estimated:** ~6-8 test functions

---

### 4.5 Test TicTacToe AlphaZero Agent

**File:** `tests/agents/test_tictactoe_alphazero_agent.py`
**Tests:** `src/agents/tictactoe_alphazero_agent.py`

**Key test cases:**

1. **Initialization**
   ```python
   def test_agent_initialization(tictactoe_game, mock_mcts):
       """Agent stores game and MCTS correctly."""
       agent = TicTacToeAlphaZeroAgent(tictactoe_game, mock_mcts)
       assert agent.game is tictactoe_game
       assert agent.mcts is mock_mcts

   def test_agent_implements_agent_interface():
       """Agent should be instance of Agent ABC."""
       # Verify isinstance(agent, Agent)
   ```

2. **Action Selection**
   ```python
   def test_act_delegates_to_mcts(mock_mcts):
       """act() should call MCTS.play_move()."""
       agent = TicTacToeAlphaZeroAgent(game, mock_mcts)
       state = game.reset()
       action = agent.act(state)
       mock_mcts.play_move.assert_called_once()

   def test_act_returns_valid_action():
       """act() should return legal action."""
       # Verify action is in legal_actions
   ```

3. **Game Lifecycle**
   ```python
   def test_start_clears_mcts_tree(mock_mcts):
       """start() should clear MCTS tree."""
       agent = TicTacToeAlphaZeroAgent(game, mock_mcts)
       agent.start()
       mock_mcts.clear.assert_called_once()
   ```

4. **Checkpoint Save**
   ```python
   def test_to_checkpoint_saves_model(tmp_path, sample_agent):
       """to_checkpoint() should save model.pt."""
       checkpoint_path = tmp_path / "model.pt"
       sample_agent.to_checkpoint(checkpoint_path)
       assert checkpoint_path.exists()

   def test_to_checkpoint_uses_torch_save(tmp_path, sample_agent):
       """to_checkpoint() should use torch.save correctly."""
       # Verify saved file is loadable with torch.load
   ```

5. **Checkpoint Load**
   ```python
   def test_from_checkpoint_reconstructs_agent(checkpoint_dir):
       """from_checkpoint() should reconstruct agent."""
       agent = TicTacToeAlphaZeroAgent.from_checkpoint(checkpoint_dir)
       assert isinstance(agent, TicTacToeAlphaZeroAgent)

   def test_from_checkpoint_loads_model_weights(checkpoint_dir):
       """Loaded agent should have correct model weights."""
       # Compare model parameters

   def test_from_checkpoint_creates_mcts(checkpoint_dir):
       """Loaded agent should have MCTS with correct config."""
       agent = TicTacToeAlphaZeroAgent.from_checkpoint(checkpoint_dir)
       assert agent.mcts is not None

   def test_from_checkpoint_device_override(checkpoint_dir):
       """Device can be overridden during load."""
       agent = TicTacToeAlphaZeroAgent.from_checkpoint(checkpoint_dir, device='cpu')
       # Verify model is on CPU

   def test_from_checkpoint_eval_mode(checkpoint_dir):
       """Loaded model should be in eval mode."""
       agent = TicTacToeAlphaZeroAgent.from_checkpoint(checkpoint_dir)
       assert not agent.mcts.model.training
   ```

**Fixtures needed:**
```python
@pytest.fixture
def mock_mcts():
    """Mock MCTS instance."""
    # Return Mock with play_move, clear methods

@pytest.fixture
def checkpoint_dir(tmp_path):
    """Create valid checkpoint directory."""
    # Create directory with model.pt and agent.yaml
```

**Estimated:** ~10-12 test functions

---

### 4.6 Test Connect4 AlphaZero Agent

**File:** `tests/agents/test_connect4_alphazero_agent.py`
**Tests:** `src/agents/connect4_alphazero_agent.py`

**Key test cases:**
- Same as TicTacToe agent tests, adapted for Connect4
- Test game-specific model loading (Connect4MLPNet)
- Verify Connect4-specific behavior

**Estimated:** ~10-12 test functions

---

### 4.7 Test Base AlphaZero MCTS Agent

**File:** `tests/agents/test_alphazero_mcts_agent.py`
**Tests:** `src/agents/alphazero_mcts_agent.py`

**Key test cases:**

1. **Direct Construction**
   ```python
   def test_init_stores_game_and_mcts():
       """__init__ should store game and MCTS."""
       agent = AlphaZeroMCTSAgent(game, mcts)
       assert agent.game is game
       assert agent.mcts is mcts
   ```

2. **Agent Interface**
   ```python
   def test_act_delegates_to_mcts():
       """act() should delegate to MCTS."""
       # Similar to game-specific agent tests

   def test_start_clears_mcts():
       """start() should clear MCTS tree."""
       # Similar to game-specific agent tests
   ```

3. **Checkpoint Loading (Alternative Path)**
   ```python
   def test_from_checkpoint_loads_model_and_config():
       """from_checkpoint() should load model and create MCTS."""
       # Test alternative checkpoint loading path

   def test_from_checkpoint_mcts_override():
       """MCTS params can be overridden during load."""
       # Load with custom num_sims, c_puct

   def test_from_checkpoint_device_selection():
       """Device selection works during load."""
       # Test CPU vs GPU selection

   def test_from_checkpoint_model_kwargs():
       """model_kwargs are passed to model constructor."""
       # Verify custom model kwargs are used

   def test_from_checkpoint_missing_files():
       """Raises FileNotFoundError for missing files."""
       # Test error handling
   ```

**Estimated:** ~8-10 test functions

---

## Phase 5: Model Testing

**Goal:** Test neural network implementations with basic coverage

**Scope:** Basic testing only - initialization, forward pass, shapes. No gradient flow or advanced testing.

---

### 5.1 Test GameNet Base Class

**File:** `tests/models/test_base.py`
**Tests:** `src/models/base.py`

**Key test cases:**

1. **Abstract Base Class**
   ```python
   def test_cannot_instantiate_gamenet_directly():
       """GameNet is abstract and cannot be instantiated."""
       with pytest.raises(TypeError):
           GameNet()

   def test_subclass_must_implement_forward():
       """Subclass without forward() should fail."""
       # Create incomplete subclass, verify error
   ```

2. **Class Attributes**
   ```python
   def test_default_attributes_are_none():
       """game_name, input_shape, action_size default to None."""
       class TestNet(GameNet):
           def forward(self, x): pass

       assert TestNet.game_name is None
       assert TestNet.input_shape is None
       assert TestNet.action_size is None
   ```

3. **Input Validation**
   ```python
   def test_validate_input_passes_correct_shape(concrete_game_net):
       """validate_input() should pass for correct shape."""
       x = torch.randn(4, 3, 3)  # Batch of 4
       concrete_game_net.validate_input(x)  # Should not raise

   def test_validate_input_raises_wrong_shape(concrete_game_net):
       """validate_input() should raise for wrong shape."""
       x = torch.randn(4, 2, 2)  # Wrong shape
       with pytest.raises(ValueError):
           concrete_game_net.validate_input(x)

   def test_validate_input_skips_if_none():
       """validate_input() should skip if input_shape is None."""
       # Create net with input_shape=None, verify no error
   ```

4. **Output Validation**
   ```python
   def test_validate_output_passes_correct_shapes(concrete_game_net):
       """validate_output() should pass for correct shapes."""
       logits = torch.randn(4, 9)
       value = torch.randn(4, 1)
       concrete_game_net.validate_output(logits, value)  # Should not raise

   def test_validate_output_raises_wrong_logits_shape(concrete_game_net):
       """validate_output() should raise for wrong logits shape."""
       logits = torch.randn(4, 7)  # Wrong action_size
       value = torch.randn(4, 1)
       with pytest.raises(ValueError):
           concrete_game_net.validate_output(logits, value)

   def test_validate_output_raises_wrong_value_shape(concrete_game_net):
       """validate_output() should raise for wrong value shape."""
       logits = torch.randn(4, 9)
       value = torch.randn(4, 3)  # Wrong shape
       with pytest.raises(ValueError):
           concrete_game_net.validate_output(logits, value)

   def test_validate_output_accepts_1d_value():
       """validate_output() should accept both (B, 1) and (B,) for value."""
       logits = torch.randn(4, 9)
       value = torch.randn(4)  # 1D value
       concrete_game_net.validate_output(logits, value)  # Should not raise
   ```

**Fixtures needed:**
```python
@pytest.fixture
def concrete_game_net():
    """Concrete GameNet implementation for testing."""
    class TestNet(GameNet):
        game_name = "test"
        input_shape = (3, 3)
        action_size = 9

        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(9, 10)

        def forward(self, x):
            B = x.shape[0]
            return torch.zeros(B, 9), torch.zeros(B, 1)

    return TestNet()
```

**Estimated:** ~10-12 test functions

---

### 5.2 Test TicTacToe MLP Model

**File:** `tests/games/tictactoe/models/test_mlp.py`
**Tests:** `src/games/tictactoe/models/mlp.py`

**Key test cases:**

1. **Initialization**
   ```python
   def test_model_default_hidden_size():
       """Model should use default hidden size (64)."""
       model = TicTacToeMLPNet()
       # Verify hidden size is 64

   def test_model_custom_hidden_size():
       """Model should accept custom hidden size."""
       model = TicTacToeMLPNet(hidden=128)
       # Verify hidden size is 128

   def test_model_class_attributes():
       """Model should have correct class attributes."""
       assert TicTacToeMLPNet.game_name == "tictactoe"
       assert TicTacToeMLPNet.input_shape == (3, 3)
       assert TicTacToeMLPNet.action_size == 9
   ```

2. **Forward Pass**
   ```python
   def test_forward_output_shapes(model):
       """Forward pass should return correct shapes."""
       x = torch.randn(4, 3, 3)
       logits, value = model(x)
       assert logits.shape == (4, 9)
       assert value.shape == (4, 1)

   def test_forward_output_types(model):
       """Forward pass should return float32 tensors."""
       x = torch.randn(4, 3, 3)
       logits, value = model(x)
       assert logits.dtype == torch.float32
       assert value.dtype == torch.float32

   def test_forward_value_range(model):
       """Value should be in [-1, 1] range (due to tanh)."""
       x = torch.randn(4, 3, 3)
       _, value = model(x)
       assert (value >= -1).all()
       assert (value <= 1).all()

   def test_forward_batch_dimension(model):
       """Forward pass should preserve batch dimension."""
       for batch_size in [1, 4, 16]:
           x = torch.randn(batch_size, 3, 3)
           logits, value = model(x)
           assert logits.shape[0] == batch_size
           assert value.shape[0] == batch_size
   ```

3. **Input Validation**
   ```python
   def test_forward_accepts_correct_input_shape(model):
       """Forward pass should accept [B, 3, 3] input."""
       x = torch.randn(4, 3, 3)
       model(x)  # Should not raise

   def test_forward_raises_on_wrong_input_shape(model):
       """Forward pass should raise on wrong input shape."""
       x = torch.randn(4, 2, 2)  # Wrong shape
       with pytest.raises(Exception):  # May be RuntimeError or ValueError
           model(x)
   ```

4. **Integration**
   ```python
   def test_model_forward_doesnt_crash(model):
       """Basic smoke test - forward pass doesn't crash."""
       x = torch.randn(1, 3, 3)
       logits, value = model(x)
       assert logits is not None
       assert value is not None

   def test_model_is_registered():
       """Model should be auto-registered in ModelRegistry."""
       from src.models.registry import ModelRegistry
       model_class = ModelRegistry.get_model('TicTacToeMLPNet')
       assert model_class is TicTacToeMLPNet
   ```

**Fixtures needed:**
```python
@pytest.fixture
def model():
    """TicTacToeMLPNet instance with default settings."""
    return TicTacToeMLPNet(hidden=64)

@pytest.fixture
def sample_input():
    """Sample input tensor for testing."""
    return torch.randn(4, 3, 3)
```

**Estimated:** ~10-12 test functions

**NOT included (as requested):**
- Gradient flow testing
- Save/load model testing
- Training mode vs eval mode
- Advanced initialization testing

---

### 5.3 Test Connect4 MLP Model

**File:** `tests/games/connect4/models/test_mlp.py`
**Tests:** `src/games/connect4/models/mlp.py`

**Key test cases:**
- Same as TicTacToe model tests, adapted for Connect4
- Test input shape [B, 6, 7]
- Test output shape [B, 7] for logits
- Test default hidden size (128)
- Test class attributes (game_name="connect4", action_size=7)

**Fixtures needed:**
```python
@pytest.fixture
def model():
    """Connect4MLPNet instance with default settings."""
    return Connect4MLPNet(hidden=128)

@pytest.fixture
def sample_input():
    """Sample input tensor for Connect4."""
    return torch.randn(4, 6, 7)
```

**Estimated:** ~10-12 test functions

---

## Summary - Phase 4-5

### Total New Test Files: 10

**Phase 4 (Agents & Persistence):** 7 files
1. `tests/agents/conftest.py` - fixtures
2. `tests/agents/test_agent.py` - RandomAgent
3. `tests/agents/test_checkpoint_utils.py` - save/load
4. `tests/agents/test_alphazero_agent_config.py` - config
5. `tests/agents/test_tictactoe_alphazero_agent.py` - TicTacToe agent
6. `tests/agents/test_connect4_alphazero_agent.py` - Connect4 agent
7. `tests/agents/test_alphazero_mcts_agent.py` - base MCTS agent

**Phase 5 (Models):** 3 files
8. `tests/models/test_base.py` - GameNet base
9. `tests/games/tictactoe/models/test_mlp.py` - TicTacToe model
10. `tests/games/connect4/models/test_mlp.py` - Connect4 model

### Estimated Test Functions
- **Phase 4:** ~60-70 test functions
- **Phase 5:** ~30-36 test functions
- **Total:** ~90-106 new test functions

### Coverage Goals
- **Agents:** >85% coverage
- **Models:** >80% coverage (basic testing only)
- **Checkpoint I/O:** >90% coverage

---

## Notes for Future Implementation

1. **Mocking Strategy:** Use `unittest.mock.Mock` for MCTS in agent tests to avoid running expensive tree searches
2. **Determinism:** Always use seeded RNGs (`numpy_rng` fixture) for reproducible tests
3. **Device Testing:** All tests run on CPU to avoid GPU dependencies
4. **Checkpoint Tests:** Always use `tmp_path` fixture, never write to real directories
5. **Model Testing:** Keep it simple - no gradient flow or training tests (per user request)

---

**Last Updated:** 2026-01-02
**Status:** Awaiting Phase 1-3 completion
