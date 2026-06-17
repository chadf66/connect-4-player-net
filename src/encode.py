"""Board ↔ tensor encoding plus horizontal-mirror data augmentation.

The encoded tensor is oriented from the current player's perspective:
channel 0 = current player's pieces, channel 1 = opponent's pieces,
channel 2 = empty squares. The network never has to learn "which color am I"
because the input is always "me vs opponent."
"""

from __future__ import annotations

import numpy as np
import torch

from src.game import COLS, EMPTY, P1, P2, ROWS, Board


def encode(board: Board) -> torch.Tensor:
    """Return a (3, ROWS, COLS) float tensor from the current player's POV."""
    me = board.current_player
    opp = P2 if me == P1 else P1
    grid = board.grid
    me_plane = (grid == me).astype(np.float32)
    opp_plane = (grid == opp).astype(np.float32)
    empty_plane = (grid == EMPTY).astype(np.float32)
    return torch.from_numpy(np.stack([me_plane, opp_plane, empty_plane], axis=0))


def encode_batch(boards: list[Board]) -> torch.Tensor:
    """Encode a list of boards into a (B, 3, ROWS, COLS) tensor."""
    return torch.stack([encode(b) for b in boards], dim=0)


def mirror_state(state: torch.Tensor) -> torch.Tensor:
    """Horizontal flip of the encoded state tensor along the COLS axis."""
    return state.flip(-1)


def mirror_policy(policy: torch.Tensor) -> torch.Tensor:
    """Horizontal flip of a policy vector (column c ↔ column COLS - 1 - c)."""
    return policy.flip(-1)


def mirror_move(col: int) -> int:
    return COLS - 1 - col


def legal_mask(board: Board) -> torch.Tensor:
    """Return a length-COLS bool tensor: True for legal columns."""
    mask = torch.zeros(COLS, dtype=torch.bool)
    for c in board.legal_moves():
        mask[c] = True
    return mask


def legal_mask_batch(boards: list[Board]) -> torch.Tensor:
    """Return a (B, COLS) bool tensor of legal-move masks."""
    return torch.stack([legal_mask(b) for b in boards], dim=0)
