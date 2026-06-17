"""AlphaZero-style training loop: MCTS self-play → replay buffer → SGD.

Loss = policy cross-entropy (against the MCTS visit distribution) + value MSE.

  policy_loss = -Σ_a π(a) · log p(a|s)     (π = MCTS visit distribution)
  value_loss  = MSE(V(s), z)

Where π is the search's visit distribution at s and z is the game outcome from the
perspective of the player who moved at s. Unlike a REINFORCE target, π carries the
relative strength of *every* candidate move, so this is plain supervised regression
toward the search result — no advantage/baseline machinery needed.

Checkpoints are saved to `checkpoints/iter_NNNN.pt` and contain model weights,
optimizer state, and the replay buffer, so a re-run resumes the latest exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.evaluate import (
    NeuralPlayer,
    load_or_create_test_positions,
    play_match,
    solver_match_rate,
)
from src.model import ConnectFourNet, mask_illegal_logits, select_device
from src.opponents import HeuristicPlayer, RandomPlayer
from src.replay import ReplayBuffer
from src.selfplay import SelfPlayConfig, play_games
from src.solver import SOLVER_BINARY, Solver


@dataclass
class TrainConfig:
    iterations: int = 100
    games_per_iter: int = 50
    steps_per_iter: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    buffer_capacity: int = 100_000
    num_simulations: int = 400
    c_puct: float = 1.5
    eval_every: int = 10
    eval_games: int = 60
    solver_eval_positions: int = 200
    save_every: int = 10
    checkpoints_dir: Path = Path("checkpoints")


def _compute_loss(
    model: ConnectFourNet,
    states: torch.Tensor,
    policies: torch.Tensor,
    outcomes: torch.Tensor,
    legal_masks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_logits, value = model(states)
    policy_logits = mask_illegal_logits(policy_logits, legal_masks)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    # Cross-entropy between the network policy and the MCTS visit distribution.
    # Illegal columns have log_prob = -inf and π = 0; 0 * -inf is NaN, so zero
    # those terms explicitly (π=0 means they contribute nothing to the target).
    terms = torch.where(policies > 0, policies * log_probs, torch.zeros_like(log_probs))
    policy_loss = -terms.sum(dim=1).mean()
    value_loss = F.mse_loss(value, outcomes)
    total = policy_loss + value_loss
    metrics = {
        "loss": total.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "value_mean": value.mean().item(),
    }
    return total, metrics


def _save_checkpoint(
    path: Path,
    model: ConnectFourNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    iteration: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "buffer": buffer.state_dict(),
            "iteration": iteration,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: ConnectFourNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    buffer.load_state_dict(ckpt["buffer"])
    return ckpt["iteration"]


def _latest_checkpoint(dir: Path) -> Path | None:
    if not dir.exists():
        return None
    ckpts = sorted(dir.glob("iter_*.pt"))
    return ckpts[-1] if ckpts else None


def _append_metrics(path: Path, record: dict) -> None:
    """Append one iteration's metrics as a JSON line (plottable after the run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def train(cfg: TrainConfig) -> None:
    device = select_device()
    print(f"device: {device}")

    model = ConnectFourNet().to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity)

    start_iter = 0
    latest = _latest_checkpoint(cfg.checkpoints_dir)
    if latest is not None:
        start_iter = _load_checkpoint(latest, model, optimizer, buffer, device) + 1
        print(f"resumed from {latest} (iteration {start_iter})")

    selfplay_cfg = SelfPlayConfig(
        num_simulations=cfg.num_simulations, c_puct=cfg.c_puct
    )
    rng = np.random.default_rng(int(time.time()) % 2**31)
    metrics_path = cfg.checkpoints_dir / "metrics.jsonl"

    # In-loop solver match-rate needs the compiled solver and a small frozen suite
    # (kept separate from the larger CLI eval suite). Skip gracefully if unbuilt.
    solver_positions = None
    if SOLVER_BINARY.exists():
        solver_positions = load_or_create_test_positions(
            cfg.checkpoints_dir / "train_solver_positions.pkl", cfg.solver_eval_positions
        )
        print(f"in-loop solver match-rate: {len(solver_positions)} positions")
    else:
        print("solver binary not built — skipping in-loop solver match-rate")

    for iteration in range(start_iter, cfg.iterations):
        t0 = time.time()

        # --- Self-play phase ---
        samples = play_games(
            model, device, cfg.games_per_iter, selfplay_cfg, generator=rng, progress=False
        )
        buffer.extend(samples)
        selfplay_dt = time.time() - t0

        # --- Training phase ---
        train_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "value_mean": 0.0}
        steps = min(cfg.steps_per_iter, max(0, len(buffer) // cfg.batch_size))
        model.train()
        for _ in range(steps):
            batch = buffer.sample(cfg.batch_size)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = _compute_loss(
                model,
                batch["states"],
                batch["policies"],
                batch["outcomes"],
                batch["legal_masks"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in metrics.items():
                train_metrics[k] += v
        if steps:
            for k in train_metrics:
                train_metrics[k] /= steps
        train_dt = time.time() - t0 - selfplay_dt

        # --- Logging ---
        avg_game_len = len(samples) / cfg.games_per_iter
        print(
            f"iter {iteration:04d}  "
            f"games={cfg.games_per_iter} avg_len={avg_game_len:.1f}  "
            f"buffer={len(buffer)}  steps={steps}  "
            f"loss={train_metrics['loss']:.3f} (p={train_metrics['policy_loss']:.3f}, v={train_metrics['value_loss']:.3f})  "
            f"v_mean={train_metrics['value_mean']:+.2f}  "
            f"self-play {selfplay_dt:.1f}s + train {train_dt:.1f}s"
        )

        record = {
            "iteration": iteration,
            **train_metrics,
            "avg_len": avg_game_len,
            "buffer": len(buffer),
            "steps": steps,
            "selfplay_s": round(selfplay_dt, 1),
            "train_s": round(train_dt, 1),
        }

        # --- Evaluation (raw-network argmax; understates MCTS-backed strength) ---
        if cfg.eval_every and (iteration + 1) % cfg.eval_every == 0:
            neural = NeuralPlayer(model, device)
            vs_random = play_match(neural, RandomPlayer(seed=42), cfg.eval_games)
            vs_heuristic = play_match(neural, HeuristicPlayer(seed=42), cfg.eval_games)
            record["vs_random"] = vs_random.a_win_rate
            record["vs_heuristic"] = vs_heuristic.a_win_rate
            eval_str = (
                f"vs_random={vs_random.a_win_rate:.1%}  "
                f"vs_heuristic={vs_heuristic.a_win_rate:.1%}"
            )
            if solver_positions is not None:
                with Solver() as solver:
                    match = solver_match_rate(neural, solver_positions, solver)
                record["solver_match"] = match
                eval_str += f"  solver_match={match:.1%}"
            print(f"   eval: {eval_str}")

        _append_metrics(metrics_path, record)

        # --- Save ---
        if cfg.save_every and (iteration + 1) % cfg.save_every == 0:
            path = cfg.checkpoints_dir / f"iter_{iteration + 1:04d}.pt"
            _save_checkpoint(path, model, optimizer, buffer, iteration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--steps-per-iter", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--sims", type=int, default=400, dest="num_simulations")
    parser.add_argument("--c-puct", type=float, default=1.5, dest="c_puct")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=60)
    parser.add_argument("--solver-eval-positions", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
