from __future__ import annotations

import torch
import torch.nn as nn

from cnn.encoding import ACTION_SPACE_SIZE
from cnn.policy_head import PolicyHead


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.relu(x)


class SharedBackbone(nn.Module):
    def __init__(self, input_channels: int = 13, channels: int = 128, num_blocks: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(num_blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_bn(x)
        x = self.relu(x)
        return self.blocks(x)


class ValueHead(nn.Module):
    def __init__(self, channels: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.tanh(x)


class PolicyValueNetwork(nn.Module):
    def __init__(self, action_space_size: int = ACTION_SPACE_SIZE) -> None:
        super().__init__()
        self.backbone = SharedBackbone()
        self.policy_head = PolicyHead(action_space_size=action_space_size)
        self.value_head = ValueHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
