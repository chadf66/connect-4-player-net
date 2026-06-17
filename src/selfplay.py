"""Self-play game generation, driven by MCTS (AlphaZero-style).

The network plays both sides of every game. At each move:
  1. Run MCTS from the current position to get visit counts over the 7 columns.
  2. Shape the counts into a policy target π ∝ N^(1/τ) (the temperature schedule
     below). This distribution — not the single move played — is the policy label.
  3. Sample the move actually played from π, and record (state, π, player, mask).

After a game ends, walk back through its recorded positions and label each with the
outcome from the perspective of the player who moved there.

**Games are played in parallel.** A pool of up to `max_parallel` games advances in
lockstep, move by move, so that all their MCTS searches share batched network
evaluations (see `run_mcts_batched`). This is the difference between a tractable
overnight run and a multi-day one: the per-move network cost is paid once for the
whole pool instead of once per game. Games terminate at different lengths, so the
active pool shrinks over a step-cycle; when it empties, the next pool starts.

Temperature schedule (per AlphaZero practice):
  - First `temperature_full_moves` plies: full temperature (diverse, exploratory)
  - After that:                           greedy (sharpest move)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.encode import encode, legal_mask
from src.game import new_game
from src.mcts import MCTSConfig, run_mcts_batched, visit_distribution
from src.model import ConnectFourNet
from src.replay import Sample


@dataclass
class SelfPlayConfig:
    num_simulations: int = 400
    c_puct: float = 1.5
    temperature_full: float = 1.0
    temperature_low: float = 1e-6  # effectively greedy (argmax visits)
    temperature_full_moves: int = 8  # apply full temperature for the first N plies
    max_parallel: int = 64  # games advanced in lockstep, sharing batched inference


def _play_pool(
    model: ConnectFourNet,
    device: torch.device,
    pool_size: int,
    cfg: SelfPlayConfig,
    mcts_cfg: MCTSConfig,
    gen: np.random.Generator,
) -> list[Sample]:
    """Play `pool_size` games concurrently in lockstep; return all their samples."""
    boards = [new_game() for _ in range(pool_size)]
    # Per game: list of (state, policy_target, player_to_move, legal_mask).
    trajectories: list[list[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]] = [
        [] for _ in range(pool_size)
    ]
    active = list(range(pool_size))  # indices of games not yet finished

    while active:
        active_boards = [boards[i] for i in active]
        counts_list = run_mcts_batched(
            model, active_boards, device, mcts_cfg, generator=gen, add_noise=True
        )

        still_active = []
        for counts, i in zip(counts_list, active):
            board = boards[i]
            temperature = (
                cfg.temperature_full
                if board.move_count < cfg.temperature_full_moves
                else cfg.temperature_low
            )
            policy = visit_distribution(counts, temperature)
            trajectories[i].append(
                (encode(board), torch.from_numpy(policy).float(), board.current_player, legal_mask(board))
            )
            move = int(gen.choice(len(policy), p=policy))
            boards[i] = board.play(move)
            if not boards[i].terminal():
                still_active.append(i)
        active = still_active

    samples: list[Sample] = []
    for i in range(pool_size):
        final = boards[i]
        for state, policy, player, mask in trajectories[i]:
            samples.append(
                Sample(state=state, policy=policy, outcome=final.outcome_for(player), legal_mask=mask)
            )
    return samples


def play_games(
    model: ConnectFourNet,
    device: torch.device,
    num_games: int,
    config: SelfPlayConfig | None = None,
    generator: np.random.Generator | None = None,
    progress: bool = False,
) -> list[Sample]:
    """Generate `num_games` self-play games, batching inference across each pool."""
    cfg = config or SelfPlayConfig()
    gen = generator or np.random.default_rng()
    mcts_cfg = MCTSConfig(num_simulations=cfg.num_simulations, c_puct=cfg.c_puct)
    model.eval()

    pbar = None
    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=num_games, desc="self-play")

    samples: list[Sample] = []
    remaining = num_games
    while remaining > 0:
        pool_size = min(cfg.max_parallel, remaining)
        samples.extend(_play_pool(model, device, pool_size, cfg, mcts_cfg, gen))
        remaining -= pool_size
        if pbar is not None:
            pbar.update(pool_size)

    if pbar is not None:
        pbar.close()
    return samples


def play_game(
    model: ConnectFourNet,
    device: torch.device,
    config: SelfPlayConfig | None = None,
    generator: np.random.Generator | None = None,
) -> list[Sample]:
    """Play a single self-play game (convenience wrapper around `play_games`)."""
    return play_games(model, device, 1, config, generator)
