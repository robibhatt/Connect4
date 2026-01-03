"""
Connect4 Match UI implementation for agent vs agent visualization.
Extends AgentMatchUI with Connect4-specific rendering (6x7 grid, colored discs).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame as pg

from src.games.core.ui.game_ui import UIConfig
from src.games.core.ui.agent_match_ui import AgentMatchUI
from src.games.core.game import Game
from src.agents.agent import Agent


class Connect4MatchUI(AgentMatchUI):
    """
    Pygame UI for Connect4 agent vs agent matches.
    - 6×7 grid layout (6 rows, 7 columns)
    - Colored discs (red for agent1, yellow for agent2)
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

        # Connect4-specific geometry (set by _compute_board_geometry)
        self.cell_width: int = 0
        self.cell_height: int = 0
        self.board_x0: int = 0
        self.board_y0: int = 0
        self.board_x1: int = 0
        self.board_y1: int = 0

        # Colors for pieces (hardcoded for simplicity)
        self.agent1_color = (200, 0, 0)    # Red
        self.agent2_color = (255, 220, 0)  # Yellow

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
        """
        Draw colored discs in the board.

        CRITICAL: Handle agent perspective correctly!
        In canonical state, +1 = current player, -1 = opponent.
        We need to determine which agent owns each piece.
        """
        assert self.screen is not None
        assert self.state is not None

        b = self.state.board  # canonical board (6, 7)
        radius = int(self.cell_width * 0.4)

        for r in range(6):
            for c in range(7):
                v = int(b[r, c])
                if v == 0:
                    continue

                # Determine owner (agent1 or agent2)
                # +1 = current player, -1 = opponent
                if v == +1:
                    owner_is_agent1 = (self.current_agent_index == 1)
                else:  # v == -1
                    owner_is_agent1 = (self.current_agent_index == 2)

                # Get color for this agent
                color = self.agent1_color if owner_is_agent1 else self.agent2_color

                # Draw disc at center of cell
                center_x = self.board_x0 + c * self.cell_width + self.cell_width // 2
                center_y = self.board_y0 + r * self.cell_height + self.cell_height // 2

                pg.draw.circle(self.screen, color, (center_x, center_y), radius)

    def _screen_pos_to_action(self, pos: tuple[int, int]) -> Optional[int]:
        """
        Convert pixel coordinates to column action (0-6).
        Not needed for agent match UI, but required by abstract base class.
        """
        return None
