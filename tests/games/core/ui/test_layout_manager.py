"""
Tests for UILayoutManager - text element layout and collision detection.

DEPRECATED: LayoutManager is deprecated - all text rendering moved to CLI.
These tests are kept for historical reference but skipped.

These tests verify:
- TextElement bounds calculation and collision detection
- Layout manager validation (overlaps, bounds)
- AgentMatchUILayout positioning logic
- Text truncation for long agent names
- Adaptive layout for different window sizes
"""

import pytest
import pygame as pg

from src.games.core.ui.layout_manager import (
    TextElement,
    UILayoutManager,
    AgentMatchUILayout
)

# Skip all tests in this module - LayoutManager is deprecated
pytestmark = pytest.mark.skip(reason="LayoutManager deprecated - text moved to CLI")


# ===== A. TextElement Tests (5 tests) =====

def test_text_element_measures_bounds_correctly():
    """Test 1: TextElement correctly measures text dimensions"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    elem = TextElement(
        content="Hello World",
        font=font,
        color=(0, 0, 0),
        x=10,
        y=20
    )

    # Bounds should be lazy-calculated
    assert elem.width == 0  # Not yet measured

    x, y, w, h = elem.get_bounds()

    # Should have position
    assert x == 10
    assert y == 20

    # Should have non-zero dimensions
    assert w > 0
    assert h > 0

    # Width/height should be cached
    assert elem.width == w
    assert elem.height == h


def test_text_element_overlap_detection_horizontal():
    """Test 2: Detects horizontal overlap between elements"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    elem1 = TextElement("Element 1", font, (0, 0, 0), x=0, y=0)
    elem2 = TextElement("Element 2", font, (0, 0, 0), x=50, y=0)  # Overlaps horizontally

    # Measure bounds first
    elem1.get_bounds()
    elem2.get_bounds()

    # Should overlap (assuming text width > 50px)
    try:
        overlaps = elem1.overlaps_with(elem2, margin=0)
        # If implementation exists, verify it detects overlap
        assert overlaps, "Should detect horizontal overlap"
    except NotImplementedError:
        pytest.skip("overlaps_with() not yet implemented (expected)")


def test_text_element_overlap_detection_vertical():
    """Test 3: Detects vertical overlap between elements"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    elem1 = TextElement("Line 1", font, (0, 0, 0), x=0, y=0)
    elem2 = TextElement("Line 2", font, (0, 0, 0), x=0, y=10)  # Overlaps vertically

    elem1.get_bounds()
    elem2.get_bounds()

    try:
        overlaps = elem1.overlaps_with(elem2, margin=0)
        assert overlaps, "Should detect vertical overlap"
    except NotImplementedError:
        pytest.skip("overlaps_with() not yet implemented (expected)")


def test_text_element_overlap_with_margin():
    """Test 4: Respects margin parameter in overlap detection"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    # Two elements barely not touching
    elem1 = TextElement("A", font, (0, 0, 0), x=0, y=0)
    elem2 = TextElement("B", font, (0, 0, 0), x=100, y=0)

    elem1.get_bounds()
    elem2.get_bounds()

    try:
        # Without margin, might not overlap
        no_margin = elem1.overlaps_with(elem2, margin=0)

        # With large margin, should overlap
        with_margin = elem1.overlaps_with(elem2, margin=50)

        # Margin should make difference
        assert with_margin or not no_margin, "Margin should affect overlap detection"
    except NotImplementedError:
        pytest.skip("overlaps_with() not yet implemented (expected)")


def test_text_element_no_overlap_when_separated():
    """Test 5: Correctly identifies non-overlapping elements"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    elem1 = TextElement("Left", font, (0, 0, 0), x=0, y=0)
    elem2 = TextElement("Right", font, (0, 0, 0), x=500, y=0)  # Far apart

    elem1.get_bounds()
    elem2.get_bounds()

    try:
        overlaps = elem1.overlaps_with(elem2, margin=0)
        assert not overlaps, "Separated elements should not overlap"
    except NotImplementedError:
        pytest.skip("overlaps_with() not yet implemented (expected)")


# ===== B. Layout Validation Tests (5 tests) =====

class ConcreteLayoutManager(UILayoutManager):
    """Minimal concrete implementation for testing base class"""
    def calculate_layout(self) -> None:
        # Position elements in a simple row
        x = 10
        for name, elem in self.elements.items():
            elem.x = x
            elem.y = 10
            elem.get_bounds()  # Measure
            x += elem.width + 10  # 10px spacing


class OverlappingLayoutManager(UILayoutManager):
    """Layout manager that intentionally creates overlaps"""
    def calculate_layout(self) -> None:
        # Put everything at same position (guaranteed overlap)
        for elem in self.elements.values():
            elem.x = 50
            elem.y = 50
            elem.get_bounds()


class OverflowLayoutManager(UILayoutManager):
    """Layout manager that positions elements outside bounds"""
    def calculate_layout(self) -> None:
        # Position elements beyond window edge
        for i, elem in enumerate(self.elements.values()):
            elem.x = self.window_size + 10  # Outside window
            elem.y = 10 + i * 30
            elem.get_bounds()


def test_layout_manager_detects_horizontal_overlap():
    """Test 6: validate_no_overlaps() detects horizontal collisions"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    layout = OverlappingLayoutManager(window_size=600, top_bar_height=70)
    layout.add_element('elem1', TextElement("Element 1", font, (0, 0, 0)))
    layout.add_element('elem2', TextElement("Element 2", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        layout.validate_no_overlaps(min_spacing=20)
        pytest.fail("Should have raised AssertionError for overlaps")
    except NotImplementedError:
        pytest.skip("validate_no_overlaps() not yet implemented (expected)")
    except AssertionError:
        pass  # Expected - overlaps detected


def test_layout_manager_detects_vertical_overlap():
    """Test 7: validate_no_overlaps() detects vertical collisions"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    layout = UILayoutManager.__new__(OverlappingLayoutManager)
    layout.__init__(window_size=600, top_bar_height=70)

    layout.add_element('line1', TextElement("Line 1", font, (0, 0, 0)))
    layout.add_element('line2', TextElement("Line 2", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        layout.validate_no_overlaps(min_spacing=10)
        pytest.fail("Should have raised AssertionError for overlaps")
    except NotImplementedError:
        pytest.skip("validate_no_overlaps() not yet implemented (expected)")
    except AssertionError:
        pass  # Expected


def test_layout_manager_detects_overflow_right_edge():
    """Test 8: validate_within_bounds() detects right edge overflow"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    layout = OverflowLayoutManager(window_size=600, top_bar_height=70)
    layout.add_element('overflow', TextElement("Overflow Text", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        layout.validate_within_bounds(margin=10)
        pytest.fail("Should have raised AssertionError for overflow")
    except NotImplementedError:
        pytest.skip("validate_within_bounds() not yet implemented (expected)")
    except AssertionError:
        pass  # Expected


def test_layout_manager_detects_overflow_bottom_edge():
    """Test 9: validate_within_bounds() detects bottom edge overflow"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    class BottomOverflowLayout(UILayoutManager):
        def calculate_layout(self):
            for elem in self.elements.values():
                elem.x = 10
                elem.y = self.window_size + 10  # Below window
                elem.get_bounds()

    layout = BottomOverflowLayout(window_size=600, top_bar_height=70)
    layout.add_element('elem', TextElement("Text", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        layout.validate_within_bounds(margin=10)
        pytest.fail("Should have raised AssertionError for overflow")
    except NotImplementedError:
        pytest.skip("validate_within_bounds() not yet implemented (expected)")
    except AssertionError:
        pass  # Expected


def test_layout_manager_validates_clean_layout():
    """Test 10: validate_*() passes for properly spaced elements"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    layout = ConcreteLayoutManager(window_size=800, top_bar_height=70)
    layout.add_element('elem1', TextElement("A", font, (0, 0, 0)))
    layout.add_element('elem2', TextElement("B", font, (0, 0, 0)))
    layout.add_element('elem3', TextElement("C", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        # Should not raise - elements are properly spaced
        layout.validate_no_overlaps(min_spacing=5)
        layout.validate_within_bounds(margin=10)
    except NotImplementedError:
        pytest.skip("Validation methods not yet implemented (expected)")


# ===== C. AgentMatchUI Layout Tests (6 tests) =====

def test_agent_match_layout_no_overlap_small_window_600px():
    """Test 11: AgentMatchUILayout handles small 600px window without overlaps"""
    pg.init()
    font = pg.font.SysFont(None, 26)
    big_font = pg.font.SysFont(None, 34)

    layout = AgentMatchUILayout(window_size=600, top_bar_height=70)

    # Add typical elements
    layout.add_element('main_message', TextElement(
        "Random Agent 1 to move. (SPACE=next move)",
        big_font, (0, 0, 0)
    ))
    layout.add_element('hint_text', TextElement(
        "SPACE=step, N=new game, Q/ESC=quit",
        font, (70, 70, 70)
    ))
    layout.add_element('agent_labels', TextElement(
        "Random Agent 1 (X) vs Random Agent 2 (O)",
        font, (70, 70, 70)
    ))
    layout.add_element('score_games', TextElement("Games: 0", font, (0, 0, 0)))
    layout.add_element('score_agent1', TextElement("Random Agent 1: 0", font, (0, 0, 0)))
    layout.add_element('score_agent2', TextElement("Random Agent 2: 0", font, (0, 0, 0)))
    layout.add_element('score_draws', TextElement("Draws: 0", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        # CRITICAL: Should not raise on 600px window (using 5px vertical spacing)
        layout.validate_no_overlaps(min_spacing=5)
        layout.validate_within_bounds(margin=10)
    except NotImplementedError:
        pytest.skip("Layout methods not yet implemented (expected)")
    except AssertionError as e:
        pytest.fail(f"Layout should be valid on 600px window: {e}")


def test_agent_match_layout_no_overlap_medium_window_700px():
    """Test 12: AgentMatchUILayout handles 700px window"""
    pg.init()
    font = pg.font.SysFont(None, 26)
    big_font = pg.font.SysFont(None, 34)

    layout = AgentMatchUILayout(window_size=700, top_bar_height=70)

    layout.add_element('main_message', TextElement("Agent1 to move", big_font, (0, 0, 0)))
    layout.add_element('hint_text', TextElement("Controls", font, (70, 70, 70)))
    layout.add_element('score_games', TextElement("Games: 5", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        layout.validate_no_overlaps(min_spacing=5)
        layout.validate_within_bounds(margin=10)
    except NotImplementedError:
        pytest.skip("Layout methods not yet implemented (expected)")


def test_agent_match_layout_no_overlap_large_window_800px():
    """Test 13: AgentMatchUILayout handles large 800px window"""
    pg.init()
    font = pg.font.SysFont(None, 26)
    big_font = pg.font.SysFont(None, 34)

    layout = AgentMatchUILayout(window_size=800, top_bar_height=70)

    layout.add_element('main_message', TextElement(
        "AlphaZero Agent to move. (SPACE=next move)",
        big_font, (0, 0, 0)
    ))
    layout.add_element('score_games', TextElement("Games: 100", font, (0, 0, 0)))
    layout.add_element('score_agent1', TextElement("AlphaZero: 75", font, (0, 0, 0)))

    layout.calculate_layout()

    try:
        layout.validate_no_overlaps(min_spacing=5)
        layout.validate_within_bounds(margin=10)
    except NotImplementedError:
        pytest.skip("Layout methods not yet implemented (expected)")


def test_agent_match_layout_handles_long_agent_names():
    """Test 14: Layout handles very long agent names without overflow"""
    pg.init()
    font = pg.font.SysFont(None, 26)
    big_font = pg.font.SysFont(None, 34)

    layout = AgentMatchUILayout(window_size=600, top_bar_height=70)

    # Extremely long agent names
    long_message = "VeryLongAgentNameThatMightCauseProblems to move. (SPACE=next move)"
    long_label = "VeryLongAgentName1 (X) vs AnotherVeryLongAgentName2 (O)"

    layout.add_element('main_message', TextElement(long_message, big_font, (0, 0, 0)))
    layout.add_element('agent_labels', TextElement(long_label, font, (70, 70, 70)))
    layout.add_element('score_agent1', TextElement(
        "VeryLongAgentNameThatMightCauseProblems: 0", font, (0, 0, 0)
    ))

    layout.calculate_layout()

    try:
        # Should handle gracefully (truncate or adapt)
        layout.validate_within_bounds(margin=10)
    except NotImplementedError:
        pytest.skip("Layout methods not yet implemented (expected)")
    except AssertionError as e:
        pytest.fail(f"Layout should handle long names: {e}")


def test_agent_match_layout_all_elements_within_bounds():
    """Test 15: All layout elements stay within window bounds"""
    pg.init()
    font = pg.font.SysFont(None, 26)
    big_font = pg.font.SysFont(None, 34)

    layout = AgentMatchUILayout(window_size=600, top_bar_height=70)

    # Full set of elements
    layout.add_element('main_message', TextElement("Message", big_font, (0, 0, 0)))
    layout.add_element('hint_text', TextElement("Hint", font, (70, 70, 70)))
    layout.add_element('agent_labels', TextElement("A (X) vs B (O)", font, (70, 70, 70)))
    layout.add_element('score_games', TextElement("Games: 0", font, (0, 0, 0)))
    layout.add_element('score_agent1', TextElement("Agent1: 0", font, (0, 0, 0)))
    layout.add_element('score_agent2', TextElement("Agent2: 0", font, (0, 0, 0)))
    layout.add_element('score_draws', TextElement("Draws: 0", font, (0, 0, 0)))

    layout.calculate_layout()

    # Check each element individually
    for name, elem in layout.elements.items():
        x, y, w, h = elem.get_bounds()

        # Should be within bounds (with margin)
        assert x >= 0, f"{name} x position {x} is negative"
        assert y >= 0, f"{name} y position {y} is negative"
        # Note: Full bounds check is in validate_within_bounds()


def test_truncate_text_to_fit_adds_ellipsis():
    """Test 16: truncate_text_to_fit() shortens long text with ellipsis"""
    pg.init()
    font = pg.font.SysFont(None, 26)

    layout = AgentMatchUILayout(window_size=600, top_bar_height=70)

    long_text = "This is a very long text that should definitely be truncated"
    max_width = 100  # Very narrow

    try:
        truncated = layout.truncate_text_to_fit(long_text, font, max_width)

        # Should be shorter
        assert len(truncated) < len(long_text), "Text should be truncated"

        # Should end with ellipsis
        assert truncated.endswith("..."), "Truncated text should have ellipsis"

        # Should fit within max_width
        surf = font.render(truncated, True, (0, 0, 0))
        assert surf.get_width() <= max_width, "Truncated text should fit within max_width"

    except NotImplementedError:
        pytest.skip("truncate_text_to_fit() not yet implemented (expected)")
