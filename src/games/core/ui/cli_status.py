"""
CLI status display for game UIs.

Provides ANSI-based in-place status updates for game state, scores, and controls.
Handles both TTY and non-TTY environments gracefully.
"""

from __future__ import annotations

import sys
from typing import Optional


class CLIStatusDisplay:
    """
    Command-line status display with ANSI escape code support.

    Displays game status, scores, and controls in a clean multi-line block
    that updates in-place (no scrolling spam) when running in a TTY.

    Supports two display modes:
    - human_vs_agent: Simple status for human vs agent games
    - agent_vs_agent: Detailed status with score tracking for agent matches
    """

    def __init__(self, mode: str, tty_enabled: Optional[bool] = None):
        """
        Initialize CLI status display.

        Args:
            mode: Display mode - "human_vs_agent" or "agent_vs_agent"
            tty_enabled: Override TTY detection. If None, auto-detect.
        """
        if mode not in ("human_vs_agent", "agent_vs_agent"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'human_vs_agent' or 'agent_vs_agent'")

        self.mode = mode
        self._tty_enabled = tty_enabled if tty_enabled is not None else self._is_tty()
        self._last_block_height = 0
        self._first_render = True

    def _is_tty(self) -> bool:
        """Check if stdout is connected to a terminal"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def update_status(self, **kwargs) -> None:
        """
        Update the status display with current game state.

        For human_vs_agent mode, expects:
            - game_name: str
            - human_symbol: str
            - agent_symbol: str
            - status_msg: str
            - control_hints: str

        For agent_vs_agent mode, expects:
            - game_name: str
            - agent1_name: str
            - agent2_name: str
            - agent1_symbol: str
            - agent2_symbol: str
            - status_msg: str
            - games_played: int
            - wins1: int
            - wins2: int
            - draws: int
            - control_hints: str
        """
        if self.mode == "human_vs_agent":
            lines = self._format_human_vs_agent(**kwargs)
        else:
            lines = self._format_agent_vs_agent(**kwargs)

        self._render_block(lines)

    def clear(self) -> None:
        """Clear the status display"""
        if self._tty_enabled and self._last_block_height > 0:
            # Move cursor up and clear all lines
            sys.stdout.write(f"\033[{self._last_block_height}A")
            for _ in range(self._last_block_height):
                sys.stdout.write("\033[K\n")
            sys.stdout.flush()
        self._last_block_height = 0
        self._first_render = True

    def _format_human_vs_agent(
        self,
        game_name: str,
        human_symbol: str,
        agent_symbol: str,
        status_msg: str,
        control_hints: str,
    ) -> list[str]:
        """Format status block for human vs agent mode"""
        lines = [
            "═" * 45,
            f"  {game_name}: You ({human_symbol}) vs Agent ({agent_symbol})",
            "─" * 45,
            f"  Status: {status_msg}",
            "─" * 45,
            f"  {control_hints}",
            "═" * 45,
        ]
        return lines

    def _format_agent_vs_agent(
        self,
        game_name: str,
        agent1_name: str,
        agent2_name: str,
        agent1_symbol: str,
        agent2_symbol: str,
        status_msg: str,
        games_played: int,
        wins1: int,
        wins2: int,
        draws: int,
        control_hints: str,
    ) -> list[str]:
        """Format status block for agent vs agent mode"""
        lines = [
            "═" * 45,
            f"  {game_name}: {agent1_name} ({agent1_symbol}) vs {agent2_name} ({agent2_symbol})",
            "─" * 45,
            f"  Status: {status_msg}",
            "─" * 45,
            "  Score:",
            f"    Games:  {games_played}",
            f"    {agent1_name}: {wins1} wins",
            f"    {agent2_name}: {wins2} wins",
            f"    Draws: {draws}",
            "─" * 45,
            f"  {control_hints}",
            "═" * 45,
        ]
        return lines

    def _render_block(self, lines: list[str]) -> None:
        """
        Render multi-line status block.

        In TTY mode: Uses ANSI escape codes to update in-place
        In non-TTY mode: Prints with separators
        """
        if self._tty_enabled:
            # Move cursor up to start of previous block (skip on first render)
            if not self._first_render and self._last_block_height > 0:
                sys.stdout.write(f"\033[{self._last_block_height}A")

            # Render each line with clear
            for line in lines:
                sys.stdout.write(f"\033[K{line}\n")

            # Pad with empty lines if block shrunk
            if not self._first_render:
                for _ in range(self._last_block_height - len(lines)):
                    sys.stdout.write("\033[K\n")

            sys.stdout.flush()
            self._last_block_height = len(lines)
            self._first_render = False
        else:
            # Non-TTY: just print all lines
            if not self._first_render:
                print()  # Add spacing between updates
            for line in lines:
                print(line)
            sys.stdout.flush()
            self._first_render = False
