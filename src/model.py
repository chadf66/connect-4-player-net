"""Residual CNN with separate policy and value heads.

Architecture follows the AlphaZero pattern (smaller, since Connect Four):
- 3 input channels (me, opponent, empty) over a 6x7 grid
- Stem: one 3x3 conv to widen to `channels`
- N residual blocks (default 8)
- Two heads:
    * Policy head: 1x1 conv to 2 channels -> flatten -> linear -> 7 logits
    * Value head:  1x1 conv to 1 channel  -> flatten -> 64 hidden -> tanh scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game import COLS, ROWS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ConnectFourNet(nn.Module):
    def __init__(self, channels: int = 128, num_blocks: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits, value).

        policy_logits: (B, COLS) — apply illegal-move masking + softmax outside.
        value:         (B,)      — already squashed to (-1, 1) by tanh.
        """
        x = self.stem(x)
        x = self.blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


def mask_illegal_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Set logits for illegal columns to -inf so softmax assigns them zero prob."""
    return logits.masked_fill(~legal_mask, float("-inf"))


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
