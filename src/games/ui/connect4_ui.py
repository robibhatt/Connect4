"""
Connect4 UI implementation.
Provides 6x7 board rendering and column-based input mapping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame as pg

from src.games.ui.game_ui import GameUI, UIConfig
from src.games.game import Game
from src.agents.agent import Agent


class Connect4UI(GameUI):
    """
    Pygame UI for Connect4.
    - 6×7 grid layout (6 rows, 7 columns)
    - Colored discs (red for human, yellow for agent)
    - Click column to drop piece (gravity)
    """

    def __init__(
        self,
        game: Game,
        agent: Agent,
        pause_seconds: float = 0.4,
        rng: Optional[np.random.Generator] = None,
        cfg: Optional[UIConfig] = None,
    ):
        super().__init__(game, agent, pause_seconds, rng, cfg)

        # Connect4-specific geometry (set by _compute_board_geometry)
        self.cell_width: int = 0
        self.cell_height: int = 0
        self.board_x0: int = 0
        self.board_y0: int = 0
        self.board_x1: int = 0
        self.board_y1: int = 0

        # Colors for pieces (hardcoded for simplicity)
        self.human_color = (200, 0, 0)    # Red
        self.agent_color = (255, 220, 0)  # Yellow

    def _compute_board_geometry(self) -> None:
        """Compute 6×7 grid geometry"""
        # Connect4 is 7 wide, 6 tall
        inner_h = self.cfg.window_size - self.cfg.top_bar_height - self.cfg.margin
        inner_w = self.cfg.window_size - 2 * self.cfg.margin

        # Compute cell size
        cell_w = inner_w // 7
        cell_h = inner_h // 6
        cell = min(cell_w, cell_h)

        self.cell_width = cell
        self.cell_height = cell

        # Compute board dimensions
        board_w = 7 * cell
        board_h = 6 * cell

        # Center the board horizontally
        self.board_x0 = (self.cfg.window_size - board_w) // 2
        self.board_y0 = self.cfg.top_bar_height
        self.board_x1 = self.board_x0 + board_w
        self.board_y1 = self.board_y0 + board_h

    def _render_board(self) -> None:
        """Draw Connect4 board with blue background and grid lines"""
        assert self.screen is not None

        # Draw blue board background
        pg.draw.rect(
            self.screen, (0, 100, 200),
            (self.board_x0, self.board_y0, self.board_x1 - self.board_x0, self.board_y1 - self.board_y0)
        )

        # Draw vertical grid lines (8 lines for 7 columns)
        for col in range(8):
            x = self.board_x0 + col * self.cell_width
            pg.draw.line(
                self.screen, (0, 0, 0),
                (x, self.board_y0),
                (x, self.board_y1),
                2
            )

        # Draw horizontal grid lines (7 lines for 6 rows)
        for row in range(7):
            y = self.board_y0 + row * self.cell_height
            pg.draw.line(
                self.screen, (0, 0, 0),
                (self.board_x0, y),
                (self.board_x1, y),
                2
            )

    def _render_pieces(self) -> None:
        """Draw colored discs in the board"""
        assert self.screen is not None
        assert self.state is not None

        b = self.state.board  # canonical board (6, 7)
        radius = int(self.cell_width * 0.4)

        for r in range(6):
            for c in range(7):
                v = int(b[r, c])
                if v == 0:
                    continue

                # Canonical ownership:
                # +1 belongs to player-to-move, -1 belongs to opponent.
                # human_to_move tells us whether player-to-move is HUMAN or AGENT.
                if v == +1:
                    owner_is_human = self.human_to_move
                else:  # v == -1
                    owner_is_human = not self.human_to_move

                color = self.human_color if owner_is_human else self.agent_color

                # Draw disc at center of cell
                center_x = self.board_x0 + c * self.cell_width + self.cell_width // 2
                center_y = self.board_y0 + r * self.cell_height + self.cell_height // 2

                pg.draw.circle(self.screen, color, (center_x, center_y), radius)

    def _screen_pos_to_action(self, pos: tuple[int, int]) -> Optional[int]:
        """
        Convert pixel coordinates to column action (0-6).
        Y-coordinate doesn't matter (gravity determines row).
        """
        x, y = pos

        # Check if click is within board bounds (x and y)
        if not (self.board_x0 <= x < self.board_x1 and self.board_y0 <= y < self.board_y1):
            return None

        # Compute column (only x matters for Connect4)
        col = (x - self.board_x0) // self.cell_width

        # Validate column
        if not (0 <= col < 7):
            return None

        # Action is just the column index
        return int(col)
