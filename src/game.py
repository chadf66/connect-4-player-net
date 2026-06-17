"""Connect Four game engine.

Board convention: 6 rows x 7 columns. Row 0 is the bottom. Pieces fall down,
so dropping in a column fills the lowest empty row. Players are 1 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
P1 = 1
P2 = 2


@dataclass
class Board:
    grid: np.ndarray = field(default_factory=lambda: np.zeros((ROWS, COLS), dtype=np.int8))
    current_player: int = P1
    move_count: int = 0
    history: list[int] = field(default_factory=list)
    _winner: int | None = None
    _terminal: bool = False

    def copy(self) -> "Board":
        return Board(
            grid=self.grid.copy(),
            current_player=self.current_player,
            move_count=self.move_count,
            history=list(self.history),
            _winner=self._winner,
            _terminal=self._terminal,
        )

    def legal_moves(self) -> list[int]:
        return [c for c in range(COLS) if self.grid[ROWS - 1, c] == EMPTY]

    def is_legal(self, col: int) -> bool:
        return 0 <= col < COLS and self.grid[ROWS - 1, col] == EMPTY

    def play(self, col: int) -> "Board":
        if self._terminal:
            raise ValueError("Game is over; cannot play.")
        if not self.is_legal(col):
            raise ValueError(f"Illegal move: column {col}.")
        new = self.copy()
        row = int(np.argmax(new.grid[:, col] == EMPTY))
        new.grid[row, col] = new.current_player
        new.move_count += 1
        new.history.append(col)
        if _is_winning_move(new.grid, row, col, new.current_player):
            new._winner = new.current_player
            new._terminal = True
        elif new.move_count == ROWS * COLS:
            new._terminal = True
        else:
            new.current_player = P2 if new.current_player == P1 else P1
        return new

    def winner(self) -> int | None:
        return self._winner

    def terminal(self) -> bool:
        return self._terminal

    def outcome_for(self, player: int) -> float:
        """+1 if `player` won, -1 if they lost, 0 otherwise (draw or non-terminal)."""
        if self._winner is None:
            return 0.0
        return 1.0 if self._winner == player else -1.0

    def __str__(self) -> str:
        symbols = {EMPTY: ".", P1: "X", P2: "O"}
        rows = [" ".join(symbols[int(v)] for v in self.grid[r]) for r in range(ROWS - 1, -1, -1)]
        return "\n".join(rows) + "\n" + " ".join(str(c) for c in range(COLS))


def _is_winning_move(grid: np.ndarray, row: int, col: int, player: int) -> bool:
    """Check if placing `player` at (row, col) completes a 4-in-a-row.

    Scans the four directions through the placed piece.
    """
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and grid[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
                if count >= 4:
                    return True
        if count >= 4:
            return True
    return False


def new_game() -> Board:
    return Board()
