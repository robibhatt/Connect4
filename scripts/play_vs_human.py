from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame as pg


@dataclass
class _UIConfig:
    # Window / layout
    size: int = 460          # window width/height in px
    top_bar: int = 70        # reserved space for status text
    margin: int = 30         # board margin (left/right/bottom), top margin is top_bar
    # Drawing
    line_w: int = 6
    mark_w: int = 10
    # Timing
    fps: int = 60


class TicTacToePygameUI:
    """
    Pygame UI for a *canonical* TicTacToe state:

      - s.board is ALWAYS from the perspective of player-to-move
      - +1 = player-to-move marks, -1 = opponent marks, 0 empty

    This UI:
      - Randomizes whether human is X or O each game (X always goes first)
      - Displays a stable human POV board (no visual "flipping")
      - Calls agent.start() at the beginning of each new game
      - Avoids "stale click" bugs by clearing queued mouse events during agent turns
      - Uses a non-blocking agent pause (no time.sleep freeze)
      - (NEW) Optional sanity checks to ensure UI turn-tracking stays aligned with the Game
    """

    def __init__(
        self,
        game,
        agent,
        pause_seconds: float = 0.4,
        rng: Optional[np.random.Generator] = None,
        cfg: Optional[_UIConfig] = None,
        strict_canonical_checks: bool = True,
    ):
        self.game = game
        self.agent = agent
        self.pause_seconds = float(pause_seconds)
        self.rng = np.random.default_rng() if rng is None else rng
        self.cfg = _UIConfig() if cfg is None else cfg
        self.strict_canonical_checks = bool(strict_canonical_checks)

        # Pygame stuff
        self.screen: Optional[pg.Surface] = None
        self.clock: Optional[pg.time.Clock] = None
        self.font: Optional[pg.font.Font] = None
        self.big: Optional[pg.font.Font] = None

        # Geometry
        self.cell: Optional[int] = None
        self.board_x0: Optional[int] = None
        self.board_y0: Optional[int] = None
        self.board_x1: Optional[int] = None
        self.board_y1: Optional[int] = None

        # Game/UI state
        self.s = None
        self.human_symbol = "X"
        self.agent_symbol = "O"
        self.human_to_move = True
        self.game_over = False
        self.msg = ""

        # Agent timing (non-blocking pause)
        self.agent_think_until: Optional[int] = None  # pygame ticks (ms)

    # ----------------------------
    # Public entrypoint
    # ----------------------------
    def run(self) -> None:
        pg.init()
        self.screen = pg.display.set_mode((self.cfg.size, self.cfg.size))
        pg.display.set_caption("TicTacToe vs Agent")
        self.clock = pg.time.Clock()

        self.font = pg.font.SysFont(None, 26)
        self.big = pg.font.SysFont(None, 34)

        self._compute_geometry()
        self.new_game()

        running = True
        while running:
            self.clock.tick(self.cfg.fps)
            running = self.handle_events()
            self.update()
            self.render()

        pg.quit()

    # ----------------------------
    # Game control
    # ----------------------------
    def new_game(self) -> None:
        self.agent.start()
        self.s = self.game.reset()
        self.game_over = False

        # Randomize whether human is X. X always starts.
        self.human_symbol = "X" if (self.rng.integers(0, 2) == 1) else "O"
        self.agent_symbol = "O" if self.human_symbol == "X" else "X"
        self.human_to_move = (self.human_symbol == "X")

        self.agent_think_until = None
        self.msg = f"You are {self.human_symbol}. {'Your move' if self.human_to_move else 'Agent move'}."

        # Flush stale clicks from the previous game
        pg.event.clear(pg.MOUSEBUTTONDOWN)

        # Sanity check: in canonical form, player-to-move should be represented as +1 on board
        self._assert_canonical_and_turn_sync()

    def apply_human_move(self, a: int) -> None:
        if self.game_over or (not self.human_to_move):
            return
        if not self._is_legal(a):
            return

        self._assert_canonical_and_turn_sync()

        self.s = self.game.next_state(self.s, a)
        self.human_to_move = False
        self.msg = "Agent move…"
        self.agent_think_until = None  # will be set in update()

        self._assert_canonical_and_turn_sync()

    def apply_agent_move(self) -> None:
        # Clear any clicks that happened during agent thinking; prevents stale-click bug.
        pg.event.clear(pg.MOUSEBUTTONDOWN)

        self._assert_canonical_and_turn_sync()

        a = int(self.agent.act(self.s))
        if not self._is_legal(a):
            raise ValueError(f"Agent played illegal move {a}.")

        self.s = self.game.next_state(self.s, a)
        self.human_to_move = True

        # Clear again in case the click arrived during act()/next_state()
        pg.event.clear(pg.MOUSEBUTTONDOWN)

        self.msg = f"You are {self.human_symbol}. Your move."
        self.agent_think_until = None

        self._assert_canonical_and_turn_sync()

    # ----------------------------
    # Main loop pieces
    # ----------------------------
    def handle_events(self) -> bool:
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
                a = self.pos_to_action(e.pos)
                if a is None:
                    continue
                self.apply_human_move(a)

        return True

    def update(self) -> None:
        # Check terminal (once)
        if not self.game_over:
            self._maybe_finish_game()

        # If it's agent's turn, schedule / execute after pause_seconds
        if self.game_over or self.human_to_move:
            return

        now = pg.time.get_ticks()
        if self.agent_think_until is None:
            # Start "thinking": clear clicks and set a deadline
            pg.event.clear(pg.MOUSEBUTTONDOWN)
            self.agent_think_until = now + int(1000.0 * self.pause_seconds)
            return

        if now >= self.agent_think_until:
            self.apply_agent_move()
            if not self.game_over:
                self._maybe_finish_game()

    def render(self) -> None:
        assert self.screen is not None and self.big is not None and self.font is not None

        # background
        self.screen.fill((245, 245, 245))

        # title/status (top bar)
        title = self.big.render(self.msg, True, (0, 0, 0))
        self.screen.blit(title, (20, 15))

        hint = self.font.render("Click to move. N = new game. Q/Esc = quit.", True, (70, 70, 70))
        self.screen.blit(hint, (20, 45))

        # board + marks
        self.draw_grid()
        self.draw_marks()

        pg.display.flip()

    # ----------------------------
    # Drawing
    # ----------------------------
    def draw_grid(self) -> None:
        assert self.screen is not None
        assert self.cell is not None and self.board_x0 is not None and self.board_y0 is not None and self.board_x1 is not None and self.board_y1 is not None

        for i in (1, 2):
            # vertical
            x = self.board_x0 + i * self.cell
            pg.draw.line(
                self.screen, (0, 0, 0),
                (x, self.board_y0),
                (x, self.board_y1),
                self.cfg.line_w,
            )
            # horizontal
            y = self.board_y0 + i * self.cell
            pg.draw.line(
                self.screen, (0, 0, 0),
                (self.board_x0, y),
                (self.board_x1, y),
                self.cfg.line_w,
            )

    def draw_marks(self) -> None:
        assert self.screen is not None
        assert self.cell is not None and self.board_x0 is not None and self.board_y0 is not None

        b = self.s.board  # canonical board
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
                    pg.draw.line(self.screen, (0, 0, 0), (x0 + pad, y0 + pad), (x1 - pad, y1 - pad), self.cfg.mark_w)
                    pg.draw.line(self.screen, (0, 0, 0), (x0 + pad, y1 - pad), (x1 - pad, y0 + pad), self.cfg.mark_w)
                else:
                    center = ((x0 + x1) // 2, (y0 + y1) // 2)
                    radius = (self.cell // 2) - pad
                    pg.draw.circle(self.screen, (0, 0, 0), center, radius, self.cfg.mark_w)

    # ----------------------------
    # Geometry / helpers
    # ----------------------------
    def _compute_geometry(self) -> None:
        # Board is a 3x3 grid below the top bar.
        inner_h = self.cfg.size - self.cfg.top_bar - self.cfg.margin
        inner_w = self.cfg.size - 2 * self.cfg.margin
        board_side = min(inner_w, inner_h)

        self.cell = board_side // 3
        self.board_x0 = self.cfg.margin
        self.board_y0 = self.cfg.top_bar
        self.board_x1 = self.board_x0 + 3 * self.cell
        self.board_y1 = self.board_y0 + 3 * self.cell

    def pos_to_action(self, pos) -> Optional[int]:
        assert self.cell is not None and self.board_x0 is not None and self.board_y0 is not None and self.board_x1 is not None and self.board_y1 is not None

        x, y = pos
        if not (self.board_x0 <= x < self.board_x1 and self.board_y0 <= y < self.board_y1):
            return None
        c = (x - self.board_x0) // self.cell
        r = (y - self.board_y0) // self.cell
        return int(3 * r + c)

    def _is_legal(self, a: int) -> bool:
        legal = self.game.legal_actions(self.s).astype(bool)
        return bool(0 <= a < self.game.action_size and legal[a])

    def _maybe_finish_game(self) -> None:
        done, v = self.game.terminal_value(self.s)
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

    # ----------------------------
    # NEW: sanity checks
    # ----------------------------
    def _assert_canonical_and_turn_sync(self) -> None:
        """
        If your Game truly uses canonical states (player-to-move is represented as +1 on board),
        then `to_play(s)` should be +1 (if implemented that way), and the UI's `human_to_move`
        should match who is actually to move given the "X starts" convention.
        """
        if not self.strict_canonical_checks:
            return
        if self.s is None:
            return

        # 1) If the game exposes to_play(s), ensure it matches the canonical convention.
        if hasattr(self.game, "to_play"):
            try:
                tp = int(self.game.to_play(self.s))
            except Exception:
                tp = None
            if tp is not None and tp != +1:
                raise RuntimeError(
                    f"[UI] Expected canonical state with game.to_play(s)==+1, got {tp}. "
                    "Either your Game is not canonical, or this UI should derive turns from to_play(s) "
                    "instead of tracking human_to_move manually."
                )

        # 2) Optional: ensure board contains only -1/0/+1.
        b = getattr(self.s, "board", None)
        if b is not None:
            u = np.unique(b)
            if not set(map(int, u)).issubset({-1, 0, 1}):
                raise RuntimeError(f"[UI] Board has unexpected values: {u} (expected subset of {{-1,0,1}}).")


def play_ttt_human_vs_agent_click(game, agent, pause_seconds=0.4, rng=None):
    """
    Backwards-compatible entrypoint: call this from your other script.
    Blocks until window closed / user quits.
    """
    ui = TicTacToePygameUI(game=game, agent=agent, pause_seconds=pause_seconds, rng=rng)
    ui.run()
