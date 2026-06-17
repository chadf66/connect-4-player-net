"""Sliding-window replay buffer for self-play training samples.

Each sample is a triple (state, policy, outcome):
- state:   (3, ROWS, COLS) float tensor, current-player perspective
- policy:  (COLS,) float tensor — the MCTS visit distribution (the policy target)
- outcome: float (+1 / 0 / -1) from this state's player-to-move POV
- legal_mask: (COLS,) bool — recorded so the loss can mask logits at sample time

The policy target is the search's visit distribution rather than the single move
played: a much higher-quality signal than one sampled action (it carries the
relative strength of every candidate move).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch

from src.encode import mirror_policy, mirror_state


@dataclass
class Sample:
    state: torch.Tensor  # (3, ROWS, COLS) float32
    policy: torch.Tensor  # (COLS,) float32 — MCTS visit distribution
    outcome: float
    legal_mask: torch.Tensor  # (COLS,) bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._buf: deque[Sample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, sample: Sample) -> None:
        self._buf.append(sample)

    def extend(self, samples: list[Sample]) -> None:
        self._buf.extend(samples)

    def sample(
        self,
        batch_size: int,
        mirror_prob: float = 0.5,
        rng: random.Random | None = None,
    ) -> dict[str, torch.Tensor]:
        rng = rng or random
        if batch_size > len(self._buf):
            raise ValueError(f"Requested {batch_size} samples but buffer has {len(self._buf)}.")
        picked = rng.sample(list(self._buf), batch_size)

        states = []
        policies = []
        outcomes = []
        masks = []
        for s in picked:
            state = s.state
            policy = s.policy
            mask = s.legal_mask
            # Horizontal mirror is the only valid Connect Four symmetry; flip the
            # state, the policy target, and the legal mask together.
            if rng.random() < mirror_prob:
                state = mirror_state(state)
                policy = mirror_policy(policy)
                mask = mirror_policy(mask.to(torch.float32)).to(torch.bool)
            states.append(state)
            policies.append(policy)
            outcomes.append(s.outcome)
            masks.append(mask)

        return {
            "states": torch.stack(states, dim=0),
            "policies": torch.stack(policies, dim=0),
            "outcomes": torch.tensor(outcomes, dtype=torch.float32),
            "legal_masks": torch.stack(masks, dim=0),
        }

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "samples": [
                {
                    "state": s.state,
                    "policy": s.policy,
                    "outcome": s.outcome,
                    "legal_mask": s.legal_mask,
                }
                for s in self._buf
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        self.capacity = sd["capacity"]
        # Force every sample onto the CPU. A resumed checkpoint may have been loaded
        # with map_location=<gpu>, which puts these buffer tensors on the GPU; mixing
        # them with the CPU tensors from fresh self-play makes torch.stack segfault.
        # The buffer must stay on CPU — training moves each sampled batch to the
        # device itself.
        self._buf = deque(
            (Sample(
                state=d["state"].cpu(),
                policy=d["policy"].cpu(),
                outcome=float(d["outcome"]),
                legal_mask=d["legal_mask"].cpu(),
            ) for d in sd["samples"]),
            maxlen=self.capacity,
        )
