"""
Abstract base class for game UIs.
Provides shared game loop, turn management, and agent coordination.
Subclasses implement game-specific rendering and input mapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame as pg

from games.game import Game, State
from agents.agent import Agent


@dataclass
class UIConfig:
    """Configuration parameters for game UI"""
    window_size: int = 600
    top_bar_height: int = 70
    margin: int = 30
    line_width: int = 6
    mark_width: int = 10
    fps: int = 60
    bg_color: tuple[int, int, int] = (245, 245, 245)
    text_color: tuple[int, int, int] = (0, 0, 0)


class GameUI(ABC):
    """
    Abstract base class for human vs agent gameplay with pygame UI.

    Provides:
    - Game loop (event handling, update, render)
    - Turn management (human vs agent)
    - Agent coordination (non-blocking moves)
    - Terminal detection
    - Keyboard controls

    Subclasses must implement:
    - Board geometry computation
    - Board rendering
    - Piece rendering
    - Input mapping (click → action)
    """

    def __init__(
        self,
        game: Game,
        agent: Agent,
        pause_seconds: float = 0.4,
        rng: Optional[np.random.Generator] = None,
        cfg: Optional[UIConfig] = None,
    ):
        self.game = game
        self.agent = agent
        self.pause_seconds = float(pause_seconds)
        self.rng = np.random.default_rng() if rng is None else rng
        self.cfg = cfg or UIConfig()

        # Pygame components (initialized in run())
        self.screen: Optional[pg.Surface] = None
        self.clock: Optional[pg.time.Clock] = None
        self.font: Optional[pg.font.Font] = None
        self.big_font: Optional[pg.font.Font] = None

        # Game state
        self.state: Optional[State] = None
        self.human_symbol = "X"
        self.agent_symbol = "O"
        self.human_to_move = True
        self.game_over = False
        self.msg = ""

        # Agent timing (non-blocking pause)
        self.agent_think_until: Optional[int] = None

    # ===== Abstract Methods (Subclass Implementation Required) =====

    @abstractmethod
    def _compute_board_geometry(self) -> None:
        """
        Compute and store board layout dimensions.
        Should set instance variables like self.board_x0, self.board_y0, etc.
        """
        pass

    @abstractmethod
    def _render_board(self) -> None:
        """Draw the game board (grid lines, columns, background, etc)"""
        pass

    @abstractmethod
    def _render_pieces(self) -> None:
        """Draw the pieces on the board"""
        pass

    @abstractmethod
    def _screen_pos_to_action(self, pos: tuple[int, int]) -> Optional[int]:
        """
        Convert screen pixel coordinates to a game action.
        Returns None if click is outside valid area.
        """
        pass

    # ===== Public API =====

    def run(self) -> None:
        """Main entry point - initialize pygame and run game loop"""
        pg.init()
        self.screen = pg.display.set_mode((self.cfg.window_size, self.cfg.window_size))
        pg.display.set_caption("Human vs Agent")
        self.clock = pg.time.Clock()

        self.font = pg.font.SysFont(None, 26)
        self.big_font = pg.font.SysFont(None, 34)

        self._compute_board_geometry()
        self.new_game()

        running = True
        while running:
            self.clock.tick(self.cfg.fps)
            running = self._handle_events()
            self._update()
            self._render()

        pg.quit()

    def new_game(self) -> None:
        """Start a new game, randomize who goes first"""
        self.agent.start()
        self.state = self.game.reset()
        self.game_over = False

        # Randomize whether human is X. X always starts.
        self.human_symbol = "X" if (self.rng.integers(0, 2) == 1) else "O"
        self.agent_symbol = "O" if self.human_symbol == "X" else "X"
        self.human_to_move = (self.human_symbol == "X")

        self.agent_think_until = None
        self.msg = f"You are {self.human_symbol}. {'Your move' if self.human_to_move else 'Agent move'}."

        # Flush stale clicks from the previous game (only if pygame is initialized)
        if pg.get_init():
            pg.event.clear(pg.MOUSEBUTTONDOWN)

    # ===== Game Loop Components =====

    def _handle_events(self) -> bool:
        """Process pygame events. Returns False to exit."""
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return False

            if e.type == pg.KEYDOWN:
                if e.key in (pg.K_q, pg.K_ESCAPE):
                    return False
                if e.key == pg.K_n:
                    self.new_game()

            if e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                # Only accept clicks on human turns
                if self.game_over or (not self.human_to_move):
                    continue

                action = self._screen_pos_to_action(e.pos)
                if action is None:
                    continue

                self._apply_human_move(action)

        return True

    def _update(self) -> None:
        """Update game state, coordinate agent moves"""
        # Check terminal (once)
        if not self.game_over:
            self._check_terminal()

        # If it's agent's turn, schedule / execute after pause_seconds
        if self.game_over or self.human_to_move:
            return

        now = pg.time.get_ticks() if pg.get_init() else 0
        if self.agent_think_until is None:
            # Start "thinking": clear clicks and set a deadline
            if pg.get_init():
                pg.event.clear(pg.MOUSEBUTTONDOWN)
            self.agent_think_until = now + int(1000.0 * self.pause_seconds)
            return

        if now >= self.agent_think_until:
            self._apply_agent_move()
            if not self.game_over:
                self._check_terminal()

    def _render(self) -> None:
        """Render frame: background, status text, board, pieces"""
        assert self.screen is not None and self.big_font is not None and self.font is not None

        # Background
        self.screen.fill(self.cfg.bg_color)

        # Title/status (top bar)
        title = self.big_font.render(self.msg, True, self.cfg.text_color)
        self.screen.blit(title, (20, 15))

        hint = self.font.render("Click to move. N = new game. Q/Esc = quit.", True, (70, 70, 70))
        self.screen.blit(hint, (20, 45))

        # Board + pieces (game-specific)
        self._render_board()
        self._render_pieces()

        pg.display.flip()

    # ===== Move Execution =====

    def _apply_human_move(self, action: int) -> None:
        """Execute human move, switch to agent turn"""
        if self.game_over or (not self.human_to_move):
            return
        if not self._is_legal(action):
            return

        self.state = self.game.next_state(self.state, action)
        self.human_to_move = False
        self.msg = "Agent move…"
        self.agent_think_until = None

    def _apply_agent_move(self) -> None:
        """Execute agent move, switch to human turn"""
        # Clear any clicks that happened during agent thinking
        if pg.get_init():
            pg.event.clear(pg.MOUSEBUTTONDOWN)

        action = int(self.agent.act(self.state))

        if not self._is_legal(action):
            raise ValueError(f"Agent played illegal move {action}.")

        self.state = self.game.next_state(self.state, action)
        self.human_to_move = True

        # Clear again in case the click arrived during act()/next_state()
        if pg.get_init():
            pg.event.clear(pg.MOUSEBUTTONDOWN)

        self.msg = f"You are {self.human_symbol}. Your move."
        self.agent_think_until = None

    # ===== Helper Methods =====

    def _is_legal(self, action: int) -> bool:
        """Check if action is legal in current state"""
        legal = self.game.legal_actions(self.state).astype(bool)
        return bool(0 <= action < self.game.action_size and legal[action])

    def _check_terminal(self) -> None:
        """Check for game over, update message"""
        done, v = self.game.terminal_value(self.state)
        if not done:
            return

        self.game_over = True
        self.agent_think_until = None

        if v == 0.0:
            self.msg = "Draw. (N = new game)"
            return

        # v is from player-to-move POV in canonical state.
        # If v==+1, player-to-move has won; if v==-1, opponent has won.
        if v > 0:
            winner_is_human = self.human_to_move
        else:
            winner_is_human = not self.human_to_move

        self.msg = ("YOU WIN 🎉" if winner_is_human else "YOU LOSE 💀") + "   (N = new game)"
