"""MCTS correctness tests.

These run with a *randomly initialised* network (seeded for reproducibility). That
is deliberate: a random net gives uninformative priors and noisy value estimates, so
if the search still finds a forced win or a forced block, it can only be because the
tree search and — most importantly — the terminal-value backup and sign convention
are correct. No training is involved.
"""

import numpy as np
import torch

from src.game import COLS, P1, P2, new_game
from src.mcts import MCTSConfig, run_mcts, run_mcts_batched, visit_distribution
from src.model import ConnectFourNet

# Tests run on CPU: small model, deterministic, no device-mismatch surprises.
DEVICE = torch.device("cpu")


def play_sequence(cols: list[int]):
    board = new_game()
    for c in cols:
        board = board.play(c)
    return board


def fresh_model(seed: int = 0) -> ConnectFourNet:
    torch.manual_seed(seed)
    model = ConnectFourNet().to(DEVICE)
    model.eval()
    return model


def test_mate_in_one():
    # P1 has three on the bottom row (cols 0,1,2) and it is P1's turn.
    # Playing column 3 completes four-in-a-row.
    board = play_sequence([0, 0, 1, 1, 2, 2])
    assert board.current_player == P1

    model = fresh_model()
    counts = run_mcts(
        model, board, DEVICE, MCTSConfig(num_simulations=200), add_noise=False
    )
    assert int(counts.argmax()) == 3, f"expected winning move 3, got visits {counts}"


def test_must_block():
    # P2 has three on the bottom row (cols 0,1,2); only column 3 completes it.
    # P1 (to move) has no immediate win and must block at column 3.
    board = play_sequence([6, 0, 6, 1, 5, 2])
    assert board.current_player == P1

    model = fresh_model()
    counts = run_mcts(
        model, board, DEVICE, MCTSConfig(num_simulations=400), add_noise=False
    )
    assert int(counts.argmax()) == 3, f"expected blocking move 3, got visits {counts}"


def test_illegal_columns_never_visited():
    # Fill column 0 (alternating, so no win) — it becomes illegal.
    board = play_sequence([0, 0, 0, 0, 0, 0])
    assert 0 not in board.legal_moves()

    model = fresh_model()
    counts = run_mcts(
        model, board, DEVICE, MCTSConfig(num_simulations=50), add_noise=False
    )
    assert counts[0] == 0.0
    # Every visit must land on a legal column.
    for c in range(COLS):
        if c not in board.legal_moves():
            assert counts[c] == 0.0


def test_visit_distribution_temperatures():
    counts = np.array([0, 10, 0, 30, 0, 0, 0], dtype=np.float64)

    greedy = visit_distribution(counts, temperature=0.0)
    assert greedy.argmax() == 3
    assert greedy.sum() == 1.0
    assert greedy[3] == 1.0  # one-hot at the most-visited move

    soft = visit_distribution(counts, temperature=1.0)
    np.testing.assert_allclose(soft.sum(), 1.0)
    # At τ=1, π ∝ N, so column 3 (30 visits) gets 3× column 1 (10 visits).
    np.testing.assert_allclose(soft[3], 0.75)
    np.testing.assert_allclose(soft[1], 0.25)


def test_batched_matches_sequential():
    # The whole correctness claim of parallel self-play: batching evaluations across
    # games must not change any individual game's search. With noise off and a fixed
    # network, each tree's visit counts must match the sequential run_mcts exactly.
    boards = [
        new_game(),
        play_sequence([3, 3, 2]),
        play_sequence([0, 1, 0, 2, 6]),
        play_sequence([0, 0, 1, 1, 2, 2]),  # the mate-in-1 position
    ]
    model = fresh_model()
    cfg = MCTSConfig(num_simulations=80)

    sequential = [run_mcts(model, b, DEVICE, cfg, add_noise=False) for b in boards]
    batched = run_mcts_batched(model, boards, DEVICE, cfg, add_noise=False)

    for seq, bat in zip(sequential, batched):
        np.testing.assert_array_equal(seq, bat)


def test_counts_total_matches_simulations():
    # Root visit counts should sum to the number of simulations (one per descent).
    board = new_game()
    model = fresh_model()
    sims = 64
    counts = run_mcts(
        model, board, DEVICE, MCTSConfig(num_simulations=sims), add_noise=False
    )
    assert counts.sum() == sims
