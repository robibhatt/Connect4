"""
TicTacToe Match UI implementation for agent vs agent visualization.
Extends AgentMatchUI with TicTacToe-specific rendering (3x3 grid, X and O symbols).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame as pg

from src.games.core.ui.game_ui import UIConfig
from src.games.core.ui.agent_match_ui import AgentMatchUI
from src.games.core.game import Game
from src.agents.agent import Agent


class TicTacToeMatchUI(AgentMatchUI):
    """
    Pygame UI for TicTacToe agent vs agent matches.
    - 3×3 grid layout
    - X and O symbols (black)
    - Manual stepping with SPACE key
    - Score tracking across games
    """

    def __init__(
        self,
        game: Game,
        agent1: Agent,
        agent2: Agent,
        agent1_name: Optional[str] = None,
        agent2_name: Optional[str] = None,
        pause_after_move: float = 0.3,
        rng: Optional[np.random.Generator] = None,
        cfg: Optional[UIConfig] = None,
    ):
        super().__init__(game, agent1, agent2, agent1_name, agent2_name,
                         pause_after_move, rng, cfg)

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
        """
        Draw X and O marks in cells.

        CRITICAL: Handle agent perspective correctly!
        In canonical state, +1 = current player, -1 = opponent.
        We need to determine which agent owns each piece.
        """
        assert self.screen is not None
        assert self.state is not None

        b = self.state.board  # canonical board (3, 3)
        pad = int(self.cell * 0.2)

        for r in range(3):
            for c in range(3):
                v = int(b[r, c])
                if v == 0:
                    continue

                # Determine owner (agent1 or agent2)
                # +1 = current player, -1 = opponent
                if v == +1:
                    owner_is_agent1 = (self.current_agent_index == 1)
                else:  # v == -1
                    owner_is_agent1 = (self.current_agent_index == 2)

                # Get symbol for this agent
                symbol = self.agent1_symbol if owner_is_agent1 else self.agent2_symbol

                # Compute cell bounds
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
        """
        Convert pixel coordinates to grid cell action (0-8).
        Not needed for agent match UI, but required by abstract base class.
        """
        return None
