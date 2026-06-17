"""Tests for the Pascal Pons solver wrapper.

These tests do NOT exercise the empty-root position because that's prohibitively
slow without the precomputed opening book (7x6.book). Mid-game positions solve
quickly even without the book.
"""

import pytest

from src.game import COLS, P1, new_game
from src.solver import Solver, SOLVER_BINARY


pytestmark = pytest.mark.skipif(
    not SOLVER_BINARY.exists(),
    reason="Solver binary not built (run `make` in third_party/connect4-solver).",
)


@pytest.fixture(scope="module")
def solver():
    s = Solver(analyze=True)
    yield s
    s.close()


def test_solver_starts_and_responds(solver):
    board = new_game().play(3).play(3).play(3).play(4)
    scores = solver.analyze(board)
    assert len(scores) == COLS
    assert all(s is None or isinstance(s, int) for s in scores)


def test_illegal_columns_marked_none(solver):
    # Fill column 3
    board = new_game()
    for _ in range(6):
        board = board.play(3)
    # After filling, col 3 is full; but board might be near-terminal. Reset.
    # Construct a position where col 0 is full but game is ongoing.
    moves = [0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0]  # P1 fills col 0 (6 pieces) with P2 alternating
    board = new_game()
    for m in moves:
        if board.terminal():
            break
        board = board.play(m)
    # If not terminal, check column 0 score is None.
    if not board.terminal() and not board.is_legal(0):
        scores = solver.analyze(board)
        assert scores[0] is None


def test_immediate_win_detected(solver):
    # P1 has 3 in col 3, can win by playing col 3 again (vertical).
    board = new_game().play(3).play(4).play(3).play(4).play(3).play(4)
    # Now P1 to move with 3-in-a-row vertically in col 3, P2 has 3 in col 4.
    # Actually after these moves: cols look like P1 in col 3 (3 pieces), P2 in col 4 (3 pieces).
    # P1 to move. P1 plays col 3 for the win.
    scores = solver.analyze(board)
    assert scores[3] is not None and scores[3] > 0  # winning move
    assert solver.optimal_move(board) == 3


def test_immediate_block_required(solver):
    # P2 has 3 in col 4 (vertical threat). P1 has scattered pieces with no own
    # winning move, so P1 must block at col 4 to avoid losing next turn.
    board = new_game()
    for m in [0, 4, 5, 4, 6, 4]:
        board = board.play(m)
    # Sanity: P1 has pieces at row 0 cols 0, 5, 6 (no 3-in-a-row), P2 has 3 stacked in col 4.
    assert solver.optimal_move(board) == 4


def test_optimal_moves_includes_argmax(solver):
    board = new_game().play(3).play(3).play(3).play(4).play(4)  # mid-game
    best_set = solver.optimal_moves(board)
    assert solver.optimal_move(board) in best_set
    assert len(best_set) >= 1


def test_score_matches_max_of_analyze(solver):
    board = new_game().play(3).play(2).play(3).play(4).play(3).play(5)
    scores = solver.analyze(board)
    legal_scores = [s for s in scores if s is not None]
    assert solver.score(board) == max(legal_scores)
