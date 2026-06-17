"""Play Connect Four against the trained network in the terminal.

    uv run python -m src.play                          # use the latest checkpoint
    uv run python -m src.play --checkpoint path.pt --sims 800 --first ai

The network plays via MCTS (`MCTSPlayer`) — same search used in evaluation, so this
is the model's real playing strength. Single-move search at 400–800 sims takes
~1–2s, which reads as the AI "thinking."

The board is shown from `Board.__str__`: X = player 1, O = player 2, column indices
along the bottom. You move by typing a column number; `q` quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.game import P1, P2, Board, new_game
from src.mcts import MCTSPlayer
from src.model import ConnectFourNet, select_device

SYMBOLS = {P1: "X", P2: "O"}


class HumanPlayer:
    """Reads a legal column from stdin (matches the `Player` protocol)."""

    def choose(self, board: Board) -> int:
        legal = board.legal_moves()
        while True:
            raw = input(f"Your move — columns {legal}: ").strip().lower()
            if raw in ("q", "quit", "exit"):
                raise SystemExit("Bye.")
            try:
                col = int(raw)
            except ValueError:
                print("  Enter a column number (or 'q' to quit).")
                continue
            if col in legal:
                return col
            print(f"  Column {col} isn't a legal move — pick from {legal}.")


def _latest_checkpoint(dir: Path) -> Path | None:
    ckpts = sorted(dir.glob("iter_*.pt")) if dir.exists() else []
    return ckpts[-1] if ckpts else None


def _load_model(checkpoint: Path, device: torch.device) -> ConnectFourNet:
    # Deserialize onto CPU first, then move the (tiny) assembled model to the device.
    # Loading directly with map_location=mps copies each storage to the GPU *during*
    # unpickling, which can block indefinitely if another process is saturating MPS.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ConnectFourNet()
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.to(device)
    model.eval()
    return model


def play(checkpoint: Path, sims: int, human_first: bool) -> None:
    device = select_device()
    model = _load_model(checkpoint, device)
    ai = MCTSPlayer(model, device, num_simulations=sims)
    human = HumanPlayer()

    human_player = P1 if human_first else P2
    print(f"\nLoaded {checkpoint.name} on {device}. You are '{SYMBOLS[human_player]}'; "
          f"AI thinks with {sims} sims.\n")

    board = new_game()
    while not board.terminal():
        print(board)
        print()
        if board.current_player == human_player:
            move = human.choose(board)
        else:
            print("AI thinking...")
            move = ai.choose(board)
            print(f"AI plays column {move}.")
        board = board.play(move)
        print()

    print(board)
    print()
    winner = board.winner()
    if winner is None:
        print("Draw.")
    elif winner == human_player:
        print("You win!")
    else:
        print("The network wins.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load (default: latest in --checkpoints-dir).",
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--sims", type=int, default=400, help="MCTS simulations per AI move.")
    parser.add_argument("--first", choices=["human", "ai"], default="human")
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoints_dir)
    if checkpoint is None or not checkpoint.exists():
        raise SystemExit(
            "No checkpoint found. Train first, or pass --checkpoint explicitly "
            f"(looked in {args.checkpoints_dir}/)."
        )

    play(checkpoint, args.sims, human_first=(args.first == "human"))


if __name__ == "__main__":
    main()
