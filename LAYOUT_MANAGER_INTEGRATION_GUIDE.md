# Layout Manager Integration Guide

## Overview

This guide explains how to complete the UI layout fix by implementing the stub methods in `layout_manager.py` and integrating it into `agent_match_ui.py`.

## Current Status

✅ **Completed**:
- `src/games/core/ui/layout_manager.py` - Created with stub implementations
- `tests/games/core/ui/test_layout_manager.py` - 16 comprehensive tests
- `tests/games/core/ui/test_agent_match_ui.py` - Enhanced with 5 additional tests

❌ **TODO** (for next agent):
- Implement 5 stub methods in `layout_manager.py`
- Integrate layout manager into `agent_match_ui.py`
- Verify all tests pass
- Test with `view_match.py` to confirm UI fix

## Step 1: Implement Stub Methods

### File: `src/games/core/ui/layout_manager.py`

Implement these 5 methods marked with `TODO`:

### 1.1 `TextElement.overlaps_with()`

**Location**: Line ~59
**Purpose**: Detect if two text elements' bounding boxes overlap
**Algorithm**:
```python
def overlaps_with(self, other: 'TextElement', margin: int = 0) -> bool:
    # Get bounds for both elements
    x1, y1, w1, h1 = self.get_bounds()
    x2, y2, w2, h2 = other.get_bounds()

    # Expand by margin
    x1 -= margin
    y1 -= margin
    w1 += 2 * margin
    h1 += 2 * margin

    # AABB collision detection
    # No overlap if:
    # - this is completely to the left of other
    # - this is completely to the right of other
    # - this is completely above other
    # - this is completely below other
    if x1 + w1 < x2:  # this is left of other
        return False
    if x1 > x2 + w2:  # this is right of other
        return False
    if y1 + h1 < y2:  # this is above other
        return False
    if y1 > y2 + h2:  # this is below other
        return False

    # Otherwise, they overlap
    return True
```

**Tests**: Tests 2-5 in `test_layout_manager.py`

### 1.2 `UILayoutManager.truncate_text_to_fit()`

**Location**: Line ~124
**Purpose**: Shorten text with "..." if it exceeds max_width
**Algorithm**:
```python
def truncate_text_to_fit(self, text: str, font: pg.font.Font, max_width: int) -> str:
    # Measure full text
    surf = font.render(text, True, (0, 0, 0))
    if surf.get_width() <= max_width:
        return text  # Fits as-is

    # Try progressively shorter versions with "..."
    ellipsis = "..."
    for i in range(len(text) - 1, 0, -1):
        truncated = text[:i] + ellipsis
        surf = font.render(truncated, True, (0, 0, 0))
        if surf.get_width() <= max_width:
            return truncated

    # If even "..." doesn't fit, return empty or just "..."
    return ellipsis if font.render(ellipsis, True, (0, 0, 0)).get_width() <= max_width else ""
```

**Tests**: Test 16 in `test_layout_manager.py`

### 1.3 `UILayoutManager.validate_no_overlaps()`

**Location**: Line ~142
**Purpose**: Check all element pairs for overlaps, raise AssertionError if found
**Algorithm**:
```python
def validate_no_overlaps(self, min_spacing: int = 20) -> None:
    from itertools import combinations

    element_list = list(self.elements.items())
    overlaps = []

    for (name1, elem1), (name2, elem2) in combinations(element_list, 2):
        if elem1.overlaps_with(elem2, margin=min_spacing):
            overlaps.append(f"{name1} overlaps with {name2}")

    if overlaps:
        raise AssertionError(f"Layout has {len(overlaps)} overlap(s): " + "; ".join(overlaps))
```

**Tests**: Tests 6-7, 10 in `test_layout_manager.py`

### 1.4 `UILayoutManager.validate_within_bounds()`

**Location**: Line ~155
**Purpose**: Verify all elements fit within window, raise AssertionError if overflow
**Algorithm**:
```python
def validate_within_bounds(self, margin: int = 10) -> None:
    violations = []

    for name, elem in self.elements.items():
        x, y, w, h = elem.get_bounds()

        if x < margin:
            violations.append(f"{name} x={x} < margin={margin}")
        if y < margin:
            violations.append(f"{name} y={y} < margin={margin}")
        if x + w > self.window_size - margin:
            violations.append(f"{name} right edge {x+w} > {self.window_size - margin}")
        if y + h > self.window_size - margin:
            violations.append(f"{name} bottom edge {y+h} > {self.window_size - margin}")

    if violations:
        raise AssertionError(f"Layout has {len(violations)} bounds violation(s): " + "; ".join(violations))
```

**Tests**: Tests 8-9, 10 in `test_layout_manager.py`

### 1.5 `AgentMatchUILayout.calculate_layout()` ⭐ MAIN TASK

**Location**: Line ~175
**Purpose**: Calculate non-overlapping positions for all UI elements
**Current**: Naive fixed positions (causes overlaps)
**Required**: Smart adaptive layout

**Algorithm**:
```python
def calculate_layout(self) -> None:
    """Calculate non-overlapping positions for all elements."""

    # Constants
    LEFT_MARGIN = 20
    RIGHT_MARGIN = 10
    VERTICAL_SPACING = 5
    MIN_HORIZ_GAP = 20  # Gap between left and right columns

    # === LEFT COLUMN (stacked vertically) ===
    left_x = LEFT_MARGIN
    current_y = 15  # Start position

    # 1. Main message
    if 'main_message' in self.elements:
        elem = self.elements['main_message']
        elem.x = left_x
        elem.y = current_y
        bounds = elem.get_bounds()
        current_y += bounds[3] + VERTICAL_SPACING  # Move down by height + spacing

    # 2. Hint text
    if 'hint_text' in self.elements:
        elem = self.elements['hint_text']
        elem.x = left_x
        elem.y = current_y
        bounds = elem.get_bounds()
        current_y += bounds[3] + VERTICAL_SPACING

    # 3. Agent labels (can be left-aligned or centered)
    if 'agent_labels' in self.elements:
        elem = self.elements['agent_labels']
        # Option A: Left-aligned
        elem.x = left_x
        elem.y = current_y

        # Option B: Centered (commented out)
        # bounds = elem.get_bounds()
        # elem.x = (self.window_size - bounds[2]) // 2
        # elem.y = current_y

    # === RIGHT COLUMN (score, right-aligned) ===
    # Measure widest score element first
    score_names = ['score_games', 'score_agent1', 'score_agent2', 'score_draws']
    max_score_width = 0

    for name in score_names:
        if name in self.elements:
            bounds = self.elements[name].get_bounds()
            max_score_width = max(max_score_width, bounds[2])

    # Position scores right-aligned
    score_x = self.window_size - RIGHT_MARGIN - max_score_width
    score_y = 10

    for name in score_names:
        if name in self.elements:
            elem = self.elements[name]
            # Right-align by adjusting x based on actual width
            bounds = elem.get_bounds()
            elem.x = self.window_size - RIGHT_MARGIN - bounds[2]
            elem.y = score_y
            score_y += 25  # Fixed line height for scores

    # === COLLISION PREVENTION ===
    # Ensure left column doesn't overlap right column
    # Measure max width of left elements
    max_left_width = 0
    for name in ['main_message', 'hint_text', 'agent_labels']:
        if name in self.elements:
            bounds = self.elements[name].get_bounds()
            max_left_width = max(max_left_width, bounds[2])

    left_end_x = LEFT_MARGIN + max_left_width

    # If collision, need to truncate or adjust
    # (Truncation should happen before layout in _setup_text_elements())
    # Here we just verify
    if left_end_x + MIN_HORIZ_GAP > score_x:
        # Layout is too tight - this will be caught by validate_no_overlaps()
        pass
```

**Tests**: Tests 11-15 in `test_layout_manager.py`

**Key Requirements**:
- No overlaps between any elements (20px minimum spacing)
- All elements within window bounds (10px margin)
- Works on 600px, 700px, 800px windows
- Adapts to different agent name lengths

## Step 2: Integrate into AgentMatchUI

### File: `src/games/core/ui/agent_match_ui.py`

### 2.1 Add Import (top of file, ~line 15)

```python
from src.games.core.ui.layout_manager import AgentMatchUILayout, TextElement
```

### 2.2 Initialize Layout Manager in `__init__` (~line 70)

Add after `super().__init__()`:
```python
# Layout manager for text positioning
self.layout_manager = AgentMatchUILayout(
    window_size=self.cfg.window_size,
    top_bar_height=self.cfg.top_bar_height
)
```

### 2.3 Replace `_render()` Method (~line 221-246)

Replace entire method:
```python
def _render(self) -> None:
    """Render frame using layout manager for text positioning"""
    assert self.screen is not None
    assert self.big_font is not None
    assert self.font is not None

    # Background
    self.screen.fill(self.cfg.bg_color)

    # Create text elements
    self._setup_text_elements()

    # Calculate layout (positions)
    self.layout_manager.calculate_layout()

    # Validate layout (raises AssertionError if invalid)
    self.layout_manager.validate_no_overlaps(min_spacing=20)
    self.layout_manager.validate_within_bounds(margin=10)

    # Render all text elements
    for name, elem in self.layout_manager.elements.items():
        surf = elem.font.render(elem.content, True, elem.color)
        self.screen.blit(surf, (elem.x, elem.y))

    # Board + pieces (game-specific)
    self._render_board()
    self._render_pieces()

    if pg.get_init() and pg.display.get_surface() is not None:
        pg.display.flip()
```

### 2.4 Add Helper Method `_setup_text_elements()` (new method)

Add after `_render()`:
```python
def _setup_text_elements(self) -> None:
    """Create TextElement objects for current game state"""
    self.layout_manager.elements.clear()

    # Main message
    self.layout_manager.add_element('main_message', TextElement(
        content=self.msg,
        font=self.big_font,
        color=self.cfg.text_color
    ))

    # Hint text
    self.layout_manager.add_element('hint_text', TextElement(
        content="SPACE=step, N=new game, Q/ESC=quit",
        font=self.font,
        color=(70, 70, 70)
    ))

    # Agent labels (with truncation for long names)
    max_label_width = self.cfg.window_size - 40  # 20px margin on each side
    label = f"{self.agent1_name} ({self.agent1_symbol}) vs {self.agent2_name} ({self.agent2_symbol})"
    label = self.layout_manager.truncate_text_to_fit(label, self.font, max_label_width)
    self.layout_manager.add_element('agent_labels', TextElement(
        content=label,
        font=self.font,
        color=(70, 70, 70)
    ))

    # Score lines
    score_lines = [
        f"Games: {self.games_played}",
        f"{self.agent1_name}: {self.wins1}",
        f"{self.agent2_name}: {self.wins2}",
        f"Draws: {self.draws}",
    ]
    for i, (name, line) in enumerate(zip(
        ['score_games', 'score_agent1', 'score_agent2', 'score_draws'],
        score_lines
    )):
        self.layout_manager.add_element(name, TextElement(
            content=line,
            font=self.font,
            color=self.cfg.text_color
        ))
```

### 2.5 Remove Old Methods

Delete these methods (they're replaced by layout manager):
- `_render_score_overlay()` (lines ~248-266)
- `_render_agent_labels()` (lines ~268-282)

## Step 3: Run Tests

### 3.1 Run Layout Manager Tests

```bash
pytest tests/games/core/ui/test_layout_manager.py -v
```

**Expected**: All 16 tests should PASS after implementing stubs

### 3.2 Run AgentMatchUI Tests

```bash
pytest tests/games/core/ui/test_agent_match_ui.py::test_vertical_spacing_between_hint_and_agent_labels -v
pytest tests/games/core/ui/test_agent_match_ui.py::test_main_message_has_minimum_20px_gap_to_score -v
pytest tests/games/core/ui/test_agent_match_ui.py::test_all_elements_visible_on_600px_window -v
```

**Expected**: Should PASS after integration

### 3.3 Run Full Test Suite

```bash
pytest tests/games/core/ui/test_agent_match_ui.py -v
```

**Expected**: All 48 tests should PASS

## Step 4: Verify UI Fix

### 4.1 Run view_match.py

```bash
python scripts/view_match.py
```

**Expected Behavior**:
- ✅ No text overlaps visible
- ✅ Main message doesn't collide with score overlay
- ✅ Hint text has clear spacing from agent labels
- ✅ Score is right-aligned with margin
- ✅ All text within window bounds on 600px window

### 4.2 Test Different Scenarios

Edit `scripts/view_match.yaml`:

**Scenario 1: Long agent names**
```yaml
agent1:
  type: random
  name: "VeryLongAgentNameThatTestsTruncation"
agent2:
  type: random
  name: "AnotherVeryLongAgentName"

ui:
  window_size: 600
```

**Expected**: Agent names truncated with "...", no overlaps

**Scenario 2: Large window**
```yaml
ui:
  window_size: 800
```

**Expected**: More spacing, everything proportionally spaced

## Success Criteria

✅ All 48+ tests pass
✅ No text overlaps on 600px window
✅ Score overlay properly right-aligned
✅ Long agent names handled gracefully
✅ Layout adapts to different window sizes
✅ `view_match.py` displays correctly

## Debugging Tips

If tests fail:

1. **Check overlap detection**: Print bounding boxes in `overlaps_with()`
2. **Verify measurements**: Print `get_bounds()` results in `calculate_layout()`
3. **Test incrementally**: Implement and test one method at a time
4. **Visual debugging**: Temporarily draw rectangles around text elements in pygame

## Implementation Order

Recommended sequence:

1. ✅ `TextElement.overlaps_with()` - Foundation for collision detection
2. ✅ `UILayoutManager.truncate_text_to_fit()` - Needed for long names
3. ✅ `UILayoutManager.validate_no_overlaps()` - Uses overlaps_with()
4. ✅ `UILayoutManager.validate_within_bounds()` - Independent check
5. ✅ `AgentMatchUILayout.calculate_layout()` - Main logic
6. ✅ Integrate into `agent_match_ui.py`
7. ✅ Run tests and iterate

Good luck! 🚀
