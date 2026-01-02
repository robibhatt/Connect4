"""
TicTacToe UI implementation.
Provides 3x3 grid rendering and click-to-cell input mapping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame as pg

from src.games.core.ui.game_ui import GameUI, UIConfig
from src.games.core.game import Game
from src.agents.agent import Agent


class TicTacToeUI(GameUI):
    """
    Pygame UI for TicTacToe.
    - 3×3 grid layout
    - X and O symbols (black)
    - Click any cell to place mark
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

        # TicTacToe-specific geometry (set by _compute_board_geometry)
        self.cell: int = 0
        self.board_x0: int = 0
        self.board_y0: int = 0
        self.board_x1: int = 0
        self.board_y1: int = 0

    def _compute_board_geometry(self) -> None:
        """Compute 3×3 grid geometry with square cells"""
        # Board is a 3x3 grid below the top bar
        inner_h = self.cfg.window_size - self.cfg.top_bar_height - self.cfg.margin
        inner_w = self.cfg.window_size - 2 * self.cfg.margin
        board_side = min(inner_w, inner_h)

        self.cell = board_side // 3
        self.board_x0 = self.cfg.margin
        self.board_y0 = self.cfg.top_bar_height
        self.board_x1 = self.board_x0 + 3 * self.cell
        self.board_y1 = self.board_y0 + 3 * self.cell

    def _render_board(self) -> None:
        """Draw 3×3 grid lines"""
        assert self.screen is not None

        for i in (1, 2):
            # Vertical lines
            x = self.board_x0 + i * self.cell
            pg.draw.line(
                self.screen, (0, 0, 0),
                (x, self.board_y0),
                (x, self.board_y1),
                self.cfg.line_width,
            )

            # Horizontal lines
            y = self.board_y0 + i * self.cell
            pg.draw.line(
                self.screen, (0, 0, 0),
                (self.board_x0, y),
                (self.board_x1, y),
                self.cfg.line_width,
            )

    def _render_pieces(self) -> None:
        """Draw X and O marks in cells"""
        assert self.screen is not None
        assert self.state is not None

        b = self.state.board  # canonical board (3, 3)
        pad = int(self.cell * 0.2)

        for r in range(3):
            for c in range(3):
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

                symbol = self.human_symbol if owner_is_human else self.agent_symbol

                x0 = self.board_x0 + c * self.cell
                y0 = self.board_y0 + r * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell

                if symbol == "X":
                    # Draw X as two diagonal lines
                    pg.draw.line(
                        self.screen, (0, 0, 0),
                        (x0 + pad, y0 + pad),
                        (x1 - pad, y1 - pad),
                        self.cfg.mark_width
                    )
                    pg.draw.line(
                        self.screen, (0, 0, 0),
                        (x0 + pad, y1 - pad),
                        (x1 - pad, y0 + pad),
                        self.cfg.mark_width
                    )
                else:  # "O"
                    # Draw O as a circle
                    center = ((x0 + x1) // 2, (y0 + y1) // 2)
                    radius = (self.cell // 2) - pad
                    pg.draw.circle(
                        self.screen, (0, 0, 0),
                        center, radius,
                        self.cfg.mark_width
                    )

    def _screen_pos_to_action(self, pos: tuple[int, int]) -> Optional[int]:
        """Convert pixel coordinates to grid cell action (0-8)"""
        x, y = pos

        # Check if click is within board bounds
        if not (self.board_x0 <= x < self.board_x1 and self.board_y0 <= y < self.board_y1):
            return None

        # Compute grid cell
        c = (x - self.board_x0) // self.cell
        r = (y - self.board_y0) // self.cell

        # Convert to action (row-major: 0-8)
        return int(3 * r + c)
