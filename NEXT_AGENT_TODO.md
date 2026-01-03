# Next Agent TODO: Implement Layout Manager to Fix UI Overlaps

## Quick Start

**Your mission**: Implement 5 stub methods in `layout_manager.py` to fix UI text overlaps.

**Time estimate**: ~1-2 hours

**Files to modify**:
1. `src/games/core/ui/layout_manager.py` - Implement 5 TODO methods
2. `src/games/core/ui/agent_match_ui.py` - Integrate layout manager

**How to verify**:
```bash
# Run tests (should go from 14 skipped → 16 passed)
pytest tests/games/core/ui/test_layout_manager.py -v

# View fixed UI
python scripts/view_match.py
```

## Problem Summary

The UI in `view_match.py` has severe text overlaps (see screenshot):
- Main message overlaps with score overlay
- Hint text collides with agent labels
- Hard-coded positions don't adapt to text width or window size

**Root cause**: `agent_match_ui.py` uses fixed positions (x=20, x=window-200, etc.) without measuring actual text dimensions.

## Solution Architecture

A **test-driven layout system** has been created with:
- ✅ `TextElement` class - Represents text with bounds
- ✅ `UILayoutManager` - Abstract base for layout validation
- ✅ `AgentMatchUILayout` - Concrete layout for agent match UI
- ✅ 21 comprehensive tests (16 new + 5 enhanced existing)

**Your job**: Fill in the 5 stub methods marked `TODO` to make tests pass.

## Implementation Checklist

### ☐ Step 1: Implement Collision Detection

**File**: `src/games/core/ui/layout_manager.py`, line ~59
**Method**: `TextElement.overlaps_with()`
**Purpose**: Detect if two text elements overlap using AABB collision
**Hint**: Compare bounding boxes (x, y, width, height) with margin
**Tests**: 2-5 in `test_layout_manager.py`

<details>
<summary>Algorithm Hint</summary>

```python
def overlaps_with(self, other: 'TextElement', margin: int = 0) -> bool:
    x1, y1, w1, h1 = self.get_bounds()
    x2, y2, w2, h2 = other.get_bounds()

    # Expand by margin (treat it as "personal space")
    x1 -= margin; y1 -= margin
    w1 += 2*margin; h1 += 2*margin

    # Check if rectangles DON'T overlap (easier logic)
    # Return True if they DO overlap
    # Hint: Use AABB collision formula
```
</details>

### ☐ Step 2: Implement Text Truncation

**File**: `src/games/core/ui/layout_manager.py`, line ~124
**Method**: `UILayoutManager.truncate_text_to_fit()`
**Purpose**: Shorten text with "..." if too wide
**Tests**: Test 16 in `test_layout_manager.py`

<details>
<summary>Algorithm Hint</summary>

```python
def truncate_text_to_fit(self, text: str, font: pg.font.Font, max_width: int) -> str:
    # 1. Measure full text
    # 2. If fits, return as-is
    # 3. Otherwise, try progressively shorter text + "..."
    # 4. Return first version that fits
```
</details>

### ☐ Step 3: Implement Overlap Validation

**File**: `src/games/core/ui/layout_manager.py`, line ~142
**Method**: `UILayoutManager.validate_no_overlaps()`
**Purpose**: Check all element pairs, raise AssertionError if overlap found
**Tests**: Tests 6-7, 10 in `test_layout_manager.py`

<details>
<summary>Algorithm Hint</summary>

```python
def validate_no_overlaps(self, min_spacing: int = 20) -> None:
    from itertools import combinations

    # Check all pairs
    for (name1, elem1), (name2, elem2) in combinations(self.elements.items(), 2):
        if elem1.overlaps_with(elem2, margin=min_spacing):
            # Collect violations and raise AssertionError
```
</details>

### ☐ Step 4: Implement Bounds Validation

**File**: `src/games/core/ui/layout_manager.py`, line ~155
**Method**: `UILayoutManager.validate_within_bounds()`
**Purpose**: Verify all elements within window, raise AssertionError if overflow
**Tests**: Tests 8-9, 10 in `test_layout_manager.py`

<details>
<summary>Algorithm Hint</summary>

```python
def validate_within_bounds(self, margin: int = 10) -> None:
    for name, elem in self.elements.items():
        x, y, w, h = elem.get_bounds()
        # Check: x >= margin, y >= margin
        # Check: x+w <= window_size-margin, y+h <= window_size-margin
        # Raise AssertionError if violations
```
</details>

### ☐ Step 5: Implement Smart Layout Algorithm ⭐

**File**: `src/games/core/ui/layout_manager.py`, line ~175
**Method**: `AgentMatchUILayout.calculate_layout()`
**Purpose**: Calculate positions for all UI elements without overlap
**Tests**: Tests 11-15 in `test_layout_manager.py`

This is the **main task**. The current implementation uses naive fixed positions that cause overlaps.

**Requirements**:
- Left column: main_message, hint_text, agent_labels stacked vertically
- Right column: score elements right-aligned
- 20px minimum spacing between elements
- 10px margin from window edges
- Works on 600px, 700px, 800px windows

<details>
<summary>Algorithm Hint</summary>

```python
def calculate_layout(self) -> None:
    # Constants
    LEFT_MARGIN = 20
    RIGHT_MARGIN = 10
    VERTICAL_SPACING = 5

    # LEFT COLUMN: Stack vertically
    current_y = 15
    for name in ['main_message', 'hint_text']:
        if name in self.elements:
            elem = self.elements[name]
            elem.x = LEFT_MARGIN
            elem.y = current_y
            current_y += elem.get_bounds()[3] + VERTICAL_SPACING

    # Agent labels (left-aligned or centered)
    if 'agent_labels' in self.elements:
        elem = self.elements['agent_labels']
        elem.x = LEFT_MARGIN
        elem.y = current_y

    # RIGHT COLUMN: Right-align scores
    # 1. Measure widest score element
    # 2. Calculate score_x = window_size - margin - max_width
    # 3. Position each score at (score_x, score_y + i*25)
    # 4. Adjust each score's x based on its actual width for right alignment
```
</details>

### ☐ Step 6: Integrate into AgentMatchUI

**File**: `src/games/core/ui/agent_match_ui.py`

Follow the integration guide in `LAYOUT_MANAGER_INTEGRATION_GUIDE.md`:

1. Add import (line ~15)
2. Initialize layout manager in `__init__` (line ~70)
3. Replace `_render()` method (line ~221)
4. Add `_setup_text_elements()` method
5. Delete `_render_score_overlay()` and `_render_agent_labels()`

### ☐ Step 7: Verify Fix

```bash
# Run layout tests
pytest tests/games/core/ui/test_layout_manager.py -v
# Expected: 16 passed (no skips)

# Run enhanced tests
pytest tests/games/core/ui/test_agent_match_ui.py::test_main_message_has_minimum_20px_gap_to_score -v
pytest tests/games/core/ui/test_agent_match_ui.py::test_all_elements_visible_on_600px_window -v
# Expected: PASS

# Visual verification
python scripts/view_match.py
# Expected: No overlaps, clean layout
```

## Test Results (Current vs Target)

**Current** (with stubs):
```
16 tests: 2 passed, 14 skipped
```

**Target** (after implementation):
```
16 tests: 16 passed, 0 skipped
```

## Resources

- **Integration Guide**: `LAYOUT_MANAGER_INTEGRATION_GUIDE.md` - Detailed algorithms
- **Plan**: `.claude/plans/glittery-chasing-owl.md` - Original design plan
- **Screenshot**: `Screenshot 2026-01-03 at 1.01.31 PM.png` - Shows current UI issues
- **Tests**: `tests/games/core/ui/test_layout_manager.py` - Your success criteria

## Success Criteria

✅ All 16 layout manager tests pass
✅ All 48 agent match UI tests pass (including 5 new ones)
✅ `view_match.py` displays without text overlaps
✅ Works on 600px window with long agent names
✅ Score overlay properly right-aligned

## Tips

1. **Start simple**: Implement methods 1-4 first (they're straightforward)
2. **Test incrementally**: Run tests after each method
3. **Visual debug**: Print bounding boxes in `calculate_layout()` to see positions
4. **Use the tests**: They define exactly what's needed
5. **Reference guide**: Full algorithm details in `LAYOUT_MANAGER_INTEGRATION_GUIDE.md`

## Estimated Difficulty

- Steps 1-4: ⭐ Easy (30-45 min total) - Straightforward algorithms
- Step 5: ⭐⭐ Medium (30-45 min) - Requires thinking about layout strategy
- Step 6: ⭐ Easy (15 min) - Copy-paste integration code
- Step 7: ⭐ Easy (10 min) - Run tests

**Total**: ~1.5-2 hours

Good luck! The tests will guide you to success. 🚀
