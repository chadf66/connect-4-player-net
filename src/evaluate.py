"""Evaluation: head-to-head matches and solver move-match-rate.

Three benchmark types:
  1. play_match(a, b, n) — head-to-head win counts (always plays both colors).
  2. solver_match_rate(model, positions) — fraction of positions where the
     network's argmax move matches a solver-optimal move.
  3. CLI: `python -m src.evaluate --checkpoint X --vs random` runs all of these.

The 1000-position test suite is generated once via random self-play to varying
depths, then saved to disk so it's a stable benchmark across checkpoints.
"""

from __future__ import annotations

import argparse
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from src.encode import encode, legal_mask
from src.game import Board, P1, P2, new_game
from src.model import ConnectFourNet, mask_illegal_logits, select_device
from src.opponents import HeuristicPlayer, Player, RandomPlayer
from src.solver import Solver, SOLVER_BINARY


class NeuralPlayer:
    """Plays by argmax over the model's masked policy logits (no MCTS)."""

    def __init__(self, model: ConnectFourNet, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def choose(self, board: Board) -> int:
        state = encode(board).unsqueeze(0).to(self.device)
        mask = legal_mask(board).unsqueeze(0).to(self.device)
        logits, _ = self.model(state)
        logits = mask_illegal_logits(logits, mask)
        return int(logits.argmax(dim=-1).item())


@dataclass
class MatchResult:
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / self.games if self.games else 0.0

    def __str__(self) -> str:
        return (
            f"A wins: {self.a_wins} ({self.a_win_rate:.1%})  "
            f"B wins: {self.b_wins} ({self.b_wins / max(self.games, 1):.1%})  "
            f"Draws: {self.draws}"
        )


def _play_one(player_a: Player, player_b: Player, a_starts: bool) -> int | None:
    """Play one game; return the winning player number (1 or 2) or None for draw.

    `a_starts=True` means player A is P1 (moves first).
    """
    board = new_game()
    a_is_p1 = a_starts
    while not board.terminal():
        if (board.current_player == P1) == a_is_p1:
            move = player_a.choose(board)
        else:
            move = player_b.choose(board)
        board = board.play(move)
    return board.winner()


def play_match(
    player_a: Player,
    player_b: Player,
    n_games: int,
    swap_colors: bool = True,
) -> MatchResult:
    """Play n_games matches. With swap_colors=True (default), half the games
    are A-as-P1 and half are A-as-P2 — removes first-move bias."""
    result = MatchResult()
    for i in range(n_games):
        a_starts = (i % 2 == 0) if swap_colors else True
        winner = _play_one(player_a, player_b, a_starts)
        a_is_p1 = a_starts
        if winner is None:
            result.draws += 1
        elif (winner == P1) == a_is_p1:
            result.a_wins += 1
        else:
            result.b_wins += 1
    return result


def generate_test_positions(
    n_positions: int,
    seed: int = 0,
    min_ply: int = 4,
    max_ply: int = 30,
) -> list[Board]:
    """Generate a fixed evaluation test suite via random rollouts to random depths."""
    rng = random.Random(seed)
    positions: list[Board] = []
    while len(positions) < n_positions:
        board = new_game()
        target_ply = rng.randint(min_ply, max_ply)
        while not board.terminal() and board.move_count < target_ply:
            move = rng.choice(board.legal_moves())
            board = board.play(move)
        if not board.terminal():
            positions.append(board)
    return positions


def load_or_create_test_positions(
    path: Path, n_positions: int = 1000, seed: int = 0
) -> list[Board]:
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    positions = generate_test_positions(n_positions, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(positions, f)
    return positions


def solver_match_rate(
    player: NeuralPlayer,
    positions: list[Board],
    solver: Solver,
    progress: bool = False,
) -> float:
    """Fraction of positions where the player's move is in the solver's set of
    optimal moves (ties counted as matches)."""
    iterator = positions
    if progress:
        from tqdm import tqdm
        iterator = tqdm(positions, desc="solver match-rate")
    matches = 0
    for board in iterator:
        player_move = player.choose(board)
        optimal = solver.optimal_moves(board)
        if player_move in optimal:
            matches += 1
    return matches / len(positions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vs", choices=["random", "heuristic", "solver"], required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument(
        "--test-positions-file",
        type=Path,
        default=Path("checkpoints/test_positions.pkl"),
    )
    parser.add_argument(
        "--mcts",
        action="store_true",
        help="Wrap the network in MCTS (otherwise play the raw policy argmax).",
    )
    parser.add_argument("--sims", type=int, default=800, help="MCTS simulations per move (with --mcts).")
    args = parser.parse_args()

    device = select_device()
    model = ConnectFourNet().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    if args.mcts:
        from src.mcts import MCTSPlayer
        player = MCTSPlayer(model, device, num_simulations=args.sims)
        label = f"MCTS net ({args.sims} sims)"
    else:
        player = NeuralPlayer(model, device)
        label = "raw net"

    if args.vs == "random":
        result = play_match(player, RandomPlayer(seed=42), args.games)
        print(f"{label} vs Random ({args.games} games): {result}")
    elif args.vs == "heuristic":
        result = play_match(player, HeuristicPlayer(seed=42), args.games)
        print(f"{label} vs Heuristic ({args.games} games): {result}")
    elif args.vs == "solver":
        if not SOLVER_BINARY.exists():
            raise FileNotFoundError("Solver not built; run `make` in third_party/connect4-solver/.")
        positions = load_or_create_test_positions(args.test_positions_file, args.positions)
        with Solver() as solver:
            rate = solver_match_rate(player, positions, solver, progress=True)
        print(f"{label} solver match-rate over {len(positions)} positions: {rate:.1%}")


if __name__ == "__main__":
    main()
