"""Baseline opponents for evaluation.

These are not trained — they're frozen players the network can be measured
against. Win-rate vs random and vs heuristic are the simplest progress signals.
"""

from __future__ import annotations

import random
from typing import Protocol

from src.game import COLS, Board


class Player(Protocol):
    """Anything callable as `player(board) -> int (column)`."""

    def choose(self, board: Board) -> int: ...


class RandomPlayer:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, board: Board) -> int:
        return self.rng.choice(board.legal_moves())


# Center-most-first preference order: 3, then 2/4, then 1/5, then 0/6.
_CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


class HeuristicPlayer:
    """Three-rule baseline:

    1. If a winning move is available, play it.
    2. If the opponent threatens an immediate win, block it.
    3. Otherwise prefer columns closer to the center.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, board: Board) -> int:
        legal = board.legal_moves()
        me = board.current_player

        # Rule 1: win if we can.
        for c in legal:
            after = board.play(c)
            if after.winner() == me:
                return c

        # Rule 2: block opponent's immediate win.
        # Simulate "what if opponent moved here?" by playing on a copy with
        # opponent as current_player. We can't just call play() — current_player
        # is us. So we manually check: if opponent placed in column c, would
        # they win?
        opp_board = board.copy()
        opp_board.current_player = 3 - me  # P1<->P2
        for c in legal:
            after = opp_board.play(c)
            if after.winner() == (3 - me):
                return c

        # Rule 3: center preference. Pick the legal column closest to center 3.
        for c in _CENTER_ORDER:
            if c in legal:
                return c

        # Fallback (shouldn't reach if there's any legal move).
        return self.rng.choice(legal)
