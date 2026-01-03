"""
Abstract base class for agent vs agent match visualization.
Provides manual stepping controls and score tracking across multiple games.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np
import pygame as pg

from src.games.core.game import Game
from src.games.core.ui.game_ui import GameUI, UIConfig
from src.games.core.ui.cli_status import CLIStatusDisplay
from src.agents.agent import Agent


class AgentMatchUI(GameUI):
    """
    Abstract base class for agent vs agent match visualization with manual stepping.

    Extends GameUI to support watching two AI agents play against each other:
    - Manual stepping: SPACE key advances each move
    - Score tracking: Cumulative wins/draws/games across multiple games
    - N key: Start new game while preserving score
    - Agent labels: Shows which agent has which symbol/color

    Subclasses must implement the same abstract methods as GameUI:
    - _compute_board_geometry()
    - _render_board()
    - _render_pieces()  (must handle agent perspective correctly!)
    - _screen_pos_to_action()  (can return None, not used)
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
        # Store both agents
        self.agent1 = agent1
        self.agent2 = agent2

        # Agent names (default to class names)
        self.agent1_name = agent1_name if agent1_name else agent1.__class__.__name__
        self.agent2_name = agent2_name if agent2_name else agent2.__class__.__name__

        # Match-level state (persists across games)
        self.wins1 = 0
        self.wins2 = 0
        self.draws = 0
        self.games_played = 0

        # Game-level state (reset each game)
        self.waiting_for_step = True
        self.current_agent_index = 1  # 1 or 2
        self.agent1_symbol = "X"
        self.agent2_symbol = "O"
        self.pause_after_move = float(pause_after_move)

        # Call parent constructor with dummy agent (we override agent logic)
        # We pass agent1 as the "agent" parameter just to satisfy parent's __init__
        super().__init__(game, agent1, pause_seconds=pause_after_move, rng=rng, cfg=cfg)

        # Override CLI display for agent match mode
        self.cli_display = CLIStatusDisplay(mode="agent_vs_agent")

    def run(self) -> None:
        """Main entry point - initialize pygame and run game loop"""
        pg.init()
        self.screen = pg.display.set_mode((self.cfg.window_size, self.cfg.window_size))
        pg.display.set_caption("Agent vs Agent Match")
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
        """Start a new game, randomize who goes first. Preserves cumulative scores."""
        # Start both agents
        self.agent1.start()
        self.agent2.start()

        # Reset game state
        self.state = self.game.reset()
        self.game_over = False

        # Randomize starting player (who is X)
        if self.rng.integers(0, 2) == 0:
            self.agent1_symbol = "X"
            self.agent2_symbol = "O"
            self.current_agent_index = 1
        else:
            self.agent1_symbol = "O"
            self.agent2_symbol = "X"
            self.current_agent_index = 2

        # Wait for SPACE to start
        self.waiting_for_step = True

        # Update message
        current_agent_name = self.agent1_name if self.current_agent_index == 1 else self.agent2_name
        self.msg = f"{current_agent_name} to move. (SPACE=next move)"

        # Clear events
        if pg.get_init():
            pg.event.clear()

    def _handle_events(self) -> bool:
        """Process pygame events. Returns False to exit."""
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return False

            if e.type == pg.KEYDOWN:
                # Q/ESC: Quit
                if e.key in (pg.K_q, pg.K_ESCAPE):
                    return False

                # N: New game (preserves score)
                if e.key == pg.K_n:
                    self.new_game()
                    continue

                # SPACE: Advance move or start next game
                if e.key == pg.K_SPACE:
                    if self.game_over:
                        # Start next game
                        self.new_game()
                    elif self.waiting_for_step:
                        # Clear waiting flag to allow move execution
                        self.waiting_for_step = False

        return True

    def _update(self) -> None:
        """Execute agent moves when not waiting for input"""
        # Don't do anything if game is over or waiting for SPACE
        if self.game_over or self.waiting_for_step:
            return

        # Get current agent
        agent = self.agent1 if self.current_agent_index == 1 else self.agent2

        # Get action
        action = int(agent.act(self.state))

        # Validate action
        if not self._is_legal(action):
            raise ValueError(
                f"Agent {self.current_agent_index} ({agent.__class__.__name__}) "
                f"played illegal action {action}"
            )

        # Apply move
        self.state = self.game.next_state(self.state, action)

        # Toggle agent (1 → 2, 2 → 1)
        self.current_agent_index = 3 - self.current_agent_index

        # Check if game ended
        self._check_terminal()

        # Wait for next SPACE press
        self.waiting_for_step = True

        # Update message if game not over
        if not self.game_over:
            current_agent_name = self.agent1_name if self.current_agent_index == 1 else self.agent2_name
            self.msg = f"{current_agent_name} to move. (SPACE=next move)"
            self._update_cli_status()

    def _check_terminal(self) -> None:
        """Check for game over, update scores and message"""
        done, v = self.game.terminal_value(self.state)
        if not done:
            return

        self.game_over = True
        self.games_played += 1

        if v == 0.0:
            # Draw
            self.draws += 1
            self.msg = f"Draw. Game {self.games_played}. (SPACE=next, N=new, Q=quit)"
        else:
            # Win - determine which agent won
            # v is from current player's perspective in canonical state
            # v > 0 means current player won, v < 0 means opponent won
            if v > 0:
                # Current player won
                winner_index = self.current_agent_index
            else:
                # Opponent won (switch to other agent)
                winner_index = 3 - self.current_agent_index

            if winner_index == 1:
                self.wins1 += 1
                winner_name = self.agent1_name
            else:
                self.wins2 += 1
                winner_name = self.agent2_name

            self.msg = f"{winner_name} wins! (SPACE=next game)"

        self._update_cli_status()

    def _render(self) -> None:
        """Render frame: background, board, pieces only (no text)"""
        assert self.screen is not None

        # Background
        self.screen.fill(self.cfg.bg_color)

        # Board + pieces (game-specific)
        self._render_board()
        self._render_pieces()

        if pg.get_init() and pg.display.get_surface() is not None:
            pg.display.flip()

    def _update_cli_status(self) -> None:
        """Update CLI status display with current match state"""
        game_name = self.game.__class__.__name__

        self.cli_display.update_status(
            game_name=game_name,
            agent1_name=self.agent1_name,
            agent2_name=self.agent2_name,
            agent1_symbol=self.agent1_symbol,
            agent2_symbol=self.agent2_symbol,
            status_msg=self.msg,
            games_played=self.games_played,
            wins1=self.wins1,
            wins2=self.wins2,
            draws=self.draws,
            control_hints="SPACE=step, N=new game, Q/ESC=quit"
        )

    # Abstract methods from GameUI - subclasses must implement
    @abstractmethod
    def _compute_board_geometry(self) -> None:
        """Compute and store board layout dimensions"""
        pass

    @abstractmethod
    def _render_board(self) -> None:
        """Draw the game board"""
        pass

    @abstractmethod
    def _render_pieces(self) -> None:
        """
        Draw the pieces on the board.

        CRITICAL: Must handle agent perspective correctly!
        The game state uses canonical representation (current player = +1).
        To determine which agent owns a piece:

        if piece_value == +1:
            owner_is_agent1 = (self.current_agent_index == 1)
        else:  # piece_value == -1
            owner_is_agent1 = (self.current_agent_index == 2)

        Then use self.agent1_symbol/agent2_symbol or agent1_color/agent2_color.
        """
        pass

    @abstractmethod
    def _screen_pos_to_action(self, pos: tuple[int, int]) -> Optional[int]:
        """
        Convert screen position to action.
        Not needed for agent match UI, can return None.
        """
        pass
