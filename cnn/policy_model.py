from __future__ import annotations

import torch
import torch.nn as nn

from cnn.encoding import ACTION_SPACE_SIZE
from cnn.policy_head import PolicyHead


class PolicyNetwork(nn.Module):
    """
    Minimal policy network for supervised training.

    It adapts board tensors of shape (batch_size, 13, 8, 8)
    into the 128-channel residual-tower style interface expected
    by the AlphaZero policy head.
    """

    def __init__(self, action_space_size: int = ACTION_SPACE_SIZE) -> None:
        super().__init__()
        self.stem = nn.Conv2d(
            in_channels=13,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(num_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.policy_head = PolicyHead(action_space_size=action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 13, 8, 8)

        x = self.stem(x)
        # x: (batch_size, 128, 8, 8)

        x = self.stem_bn(x)
        # x: (batch_size, 128, 8, 8)

        x = self.relu(x)
        # x: (batch_size, 128, 8, 8)

        logits = self.policy_head(x)
        # logits: (batch_size, action_space_size)

        return logits
