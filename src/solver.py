"""Wrapper around the Pascal Pons perfect-play Connect Four solver.

Used only for evaluation — never as a training target. Returns the optimal
move and exact game-theoretic score for any position.

The C++ solver expects moves as a string of digits 1..7 (1-indexed columns).
Our internal Board uses 0..6, so we translate.

Solver output format (per line):
    <input_line> <score>                  (default mode)
    <input_line> <score_col1> ... <score_col7>   (-a analyze mode)

Score convention: positive = current player wins, 0 = draw, negative = loses.
Magnitude relates to how soon (closer to root = larger |score|), but we only
use sign + argmax for our purposes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.game import COLS, Board

REPO_ROOT = Path(__file__).resolve().parent.parent
SOLVER_BINARY = REPO_ROOT / "third_party" / "connect4-solver" / "c4solver"
SOLVER_CWD = SOLVER_BINARY.parent  # so opening book can be found if present


def _moves_to_pons(history: list[int]) -> str:
    return "".join(str(c + 1) for c in history)


class Solver:
    """Persistent solver process. Reuse for many queries to amortize startup."""

    def __init__(self, analyze: bool = True, weak: bool = False) -> None:
        if not SOLVER_BINARY.exists():
            raise FileNotFoundError(
                f"Solver binary missing at {SOLVER_BINARY}. "
                f"Run `make` in third_party/connect4-solver/."
            )
        args = [str(SOLVER_BINARY)]
        if analyze:
            args.append("-a")
        if weak:
            args.append("-w")
        self._analyze = analyze
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(SOLVER_CWD),
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)

    def __enter__(self) -> "Solver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _query(self, line: str) -> list[str]:
        if self._proc.poll() is not None:
            raise RuntimeError("Solver subprocess has exited.")
        # The empty starting position needs to be sent as a newline alone.
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        out = self._proc.stdout.readline().rstrip("\n")
        if not out:
            err = self._proc.stderr.readline().strip()
            raise RuntimeError(f"Solver returned empty output (stderr: {err!r}).")
        return out.split(" ")

    def analyze(self, board: Board) -> list[int | None]:
        """Return per-column scores. None for illegal columns (full).

        Score sign: + = current player wins, 0 = draw, − = loses.
        """
        if board.terminal():
            raise ValueError("Cannot analyze a terminal position.")
        if not self._analyze:
            raise RuntimeError("Solver was not started in analyze mode.")
        line = _moves_to_pons(board.history)
        parts = self._query(line)
        # Format: "<line> <s1> <s2> ... <s7>". For the empty root position,
        # parts[0] is "" but the rest is the same. parts[1:] is always the scores.
        scores_str = parts[1:]
        # Pons returns a special value for illegal columns. Checking exact source:
        # in Solver::analyze, illegal columns get Solver::INVALID_MOVE = -1000.
        scores: list[int | None] = []
        for c in range(COLS):
            s = int(scores_str[c])
            if not board.is_legal(c):
                scores.append(None)
            else:
                scores.append(s)
        return scores

    def score(self, board: Board) -> int:
        """Return the game-theoretic score of `board` (current player's POV)."""
        if board.terminal():
            raise ValueError("Cannot score a terminal position.")
        line = _moves_to_pons(board.history)
        if self._analyze:
            scores = self.analyze(board)
            return max(s for s in scores if s is not None)
        parts = self._query(line)
        return int(parts[1] if line else parts[0])

    def optimal_move(self, board: Board) -> int:
        """Return any optimal column (argmax over legal scores)."""
        scores = self.analyze(board)
        best_score = max(s for s in scores if s is not None)
        for c, s in enumerate(scores):
            if s == best_score:
                return c
        raise RuntimeError("No legal move found.")

    def optimal_moves(self, board: Board) -> list[int]:
        """Return ALL columns tied for the best score (for match-rate scoring)."""
        scores = self.analyze(board)
        best = max(s for s in scores if s is not None)
        return [c for c, s in enumerate(scores) if s == best]
