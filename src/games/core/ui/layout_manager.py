"""
DEPRECATED: Layout manager for UI text elements.

⚠️  This module is deprecated as of 2026-01-03.
⚠️  All text rendering has been moved to CLI status display.
⚠️  This file is kept for backward compatibility only.
⚠️  New code should use CLIStatusDisplay instead.

Provides collision detection, bounds validation, and automatic text positioning
to prevent overlapping text elements in pygame UIs.

This module is generic and reusable across all game UIs (TicTacToe, Connect4, etc.).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pygame as pg


@dataclass
class TextElement:
    """
    Represents a UI text element with position and bounds.

    Attributes:
        content: The text string to display
        font: Pygame font object for rendering
        color: RGB tuple for text color
        x: Horizontal position (set by layout manager)
        y: Vertical position (set by layout manager)
        width: Text width in pixels (lazy-calculated from font)
        height: Text height in pixels (lazy-calculated from font)
    """
    content: str
    font: pg.font.Font
    color: tuple[int, int, int]
    x: int = 0  # Position (set by layout manager)
    y: int = 0
    width: int = 0  # Bounds (calculated from font)
    height: int = 0

    def get_bounds(self) -> tuple[int, int, int, int]:
        """
        Returns bounding box (x, y, width, height).

        Lazy-calculates width and height on first call.
        """
        if self.width == 0:  # Lazy measurement
            surf = self.font.render(self.content, True, self.color)
            self.width = surf.get_width()
            self.height = surf.get_height()
        return (self.x, self.y, self.width, self.height)

    def overlaps_with(self, other: 'TextElement', margin: int = 0) -> bool:
        """
        Check if this element overlaps with another (with optional margin).

        Args:
            other: Another TextElement to check collision with
            margin: Additional spacing requirement (in pixels)

        Returns:
            True if elements overlap (considering margin), False otherwise
        """
        # Get bounds for both elements
        x1, y1, w1, h1 = self.get_bounds()
        x2, y2, w2, h2 = other.get_bounds()

        # Expand first element's bounds by margin
        x1 -= margin
        y1 -= margin
        w1 += 2 * margin
        h1 += 2 * margin

        # AABB collision detection
        # Check if NOT overlapping (easier logic), then negate
        # Use <= to allow exact touching to be considered non-overlapping
        if x1 + w1 <= x2:  # this is left of other
            return False
        if x1 >= x2 + w2:  # this is right of other
            return False
        if y1 + h1 <= y2:  # this is above other
            return False
        if y1 >= y2 + h2:  # this is below other
            return False

        # Otherwise, they overlap
        return True


class UILayoutManager(ABC):
    """
    Abstract layout manager for UI text elements.

    Generic, reusable across all game UIs (TicTacToe, Connect4, etc.).
    Validates layout constraints and raises errors on violations.

    Usage:
        1. Create concrete subclass implementing calculate_layout()
        2. Add text elements via add_element()
        3. Call calculate_layout() to position elements
        4. Call validate_*() methods to ensure layout is valid
        5. Render elements at their calculated positions
    """

    def __init__(self, window_size: int, top_bar_height: int):
        """
        Initialize layout manager.

        Args:
            window_size: Window dimension in pixels (assumes square window)
            top_bar_height: Height reserved for top bar (board starts below this)
        """
        warnings.warn(
            "UILayoutManager is deprecated. Use CLIStatusDisplay instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.window_size = window_size
        self.top_bar_height = top_bar_height
        self.elements: dict[str, TextElement] = {}

    @abstractmethod
    def calculate_layout(self) -> None:
        """
        Calculate positions for all elements.

        Must set x, y coordinates on each element in self.elements.
        Implementation should ensure:
        - No overlaps between elements
        - All elements within window bounds
        - Proper spacing between elements
        """
        pass

    def add_element(self, name: str, element: TextElement) -> None:
        """Register a text element for layout calculation."""
        self.elements[name] = element

    def truncate_text_to_fit(self, text: str, font: pg.font.Font, max_width: int) -> str:
        """
        Truncate text with ellipsis if it exceeds max_width.

        Args:
            text: Original text string
            font: Font used to measure text
            max_width: Maximum allowed width in pixels

        Returns:
            Truncated string that fits within max_width (with "..." if truncated)
        """
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

        # Edge case: If even "..." doesn't fit, return empty or just "..."
        return ellipsis if font.render(ellipsis, True, (0, 0, 0)).get_width() <= max_width else ""

    def validate_no_overlaps(self, min_spacing: int = 20) -> None:
        """
        Validate no elements overlap.

        Args:
            min_spacing: Minimum required spacing between elements (pixels)

        Raises:
            AssertionError: If any elements overlap, with details about which elements
        """
        from itertools import combinations

        element_list = list(self.elements.items())
        overlaps = []

        for (name1, elem1), (name2, elem2) in combinations(element_list, 2):
            if elem1.overlaps_with(elem2, margin=min_spacing):
                overlaps.append(f"{name1} overlaps with {name2}")

        if overlaps:
            raise AssertionError(f"Layout has {len(overlaps)} overlap(s): " + "; ".join(overlaps))

    def validate_within_bounds(self, margin: int = 10) -> None:
        """
        Validate all elements fit within window bounds.

        Args:
            margin: Required margin from window edge (pixels)

        Raises:
            AssertionError: If any element overflows bounds, with details
        """
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


class AgentMatchUILayout(UILayoutManager):
    """
    Layout manager for AgentMatchUI.

    Manages text elements for agent vs agent match visualization:
    - main_message: Status message (e.g., "Agent1 to move")
    - hint_text: Control hints (e.g., "SPACE=step, N=new, Q=quit")
    - agent_labels: Agent symbols (e.g., "Agent1 (X) vs Agent2 (O)")
    - score_*: Four score text elements (games, agent1, agent2, draws)
    """

    def calculate_layout(self) -> None:
        """
        Calculate non-overlapping positions for all elements.

        Layout Strategy:
        1. Left column: Stack main_message, hint_text, agent_labels vertically
        2. Right column: Right-align score elements with proper spacing
        3. Constraints: 20px min spacing, 10px edge margin
        """
        # Constants
        LEFT_MARGIN = 20
        RIGHT_MARGIN = 10
        VERTICAL_SPACING = 5         # Vertical gap between stacked elements
        MIN_HORIZONTAL_SPACING = 20  # Horizontal gap between left and right columns
        TOP_MARGIN = 15

        # First, measure score column width (right column)
        score_names = ['score_games', 'score_agent1', 'score_agent2', 'score_draws']
        max_score_width = 0
        for name in score_names:
            if name in self.elements:
                bounds = self.elements[name].get_bounds()
                max_score_width = max(max_score_width, bounds[2])

        # Calculate maximum width for left column elements
        # Left column can use: window_size - LEFT_MARGIN - score_width - RIGHT_MARGIN - MIN_HORIZONTAL_SPACING
        if max_score_width > 0:
            max_left_width = self.window_size - LEFT_MARGIN - max_score_width - RIGHT_MARGIN - MIN_HORIZONTAL_SPACING
        else:
            max_left_width = self.window_size - LEFT_MARGIN - RIGHT_MARGIN

        # Truncate left column text if necessary
        left_column_names = ['main_message', 'hint_text', 'agent_labels']
        for name in left_column_names:
            if name in self.elements:
                elem = self.elements[name]
                bounds = elem.get_bounds()
                if bounds[2] > max_left_width:
                    # Text too wide, truncate it
                    elem.content = self.truncate_text_to_fit(elem.content, elem.font, max_left_width)
                    # Reset width to force re-measurement
                    elem.width = 0
                    elem.height = 0

        # LEFT COLUMN: Stack vertically with proper spacing
        current_y = TOP_MARGIN

        if 'main_message' in self.elements:
            elem = self.elements['main_message']
            elem.x = LEFT_MARGIN
            elem.y = current_y
            bounds = elem.get_bounds()
            current_y += bounds[3] + VERTICAL_SPACING

        if 'hint_text' in self.elements:
            elem = self.elements['hint_text']
            elem.x = LEFT_MARGIN
            elem.y = current_y
            bounds = elem.get_bounds()
            current_y += bounds[3] + VERTICAL_SPACING

        if 'agent_labels' in self.elements:
            elem = self.elements['agent_labels']
            elem.x = LEFT_MARGIN
            elem.y = current_y

        # RIGHT COLUMN: Right-align scores with proper spacing
        score_y = TOP_MARGIN  # Start scores at same level as left column (y=15)

        for name in score_names:
            if name in self.elements:
                elem = self.elements[name]
                bounds = elem.get_bounds()
                # Right-align: position x based on actual text width
                elem.x = self.window_size - RIGHT_MARGIN - bounds[2]
                elem.y = score_y
                # Use text height + min spacing for next position
                score_y += bounds[3] + VERTICAL_SPACING
