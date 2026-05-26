from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn.encoding import ACTION_SPACE_SIZE


class PolicyHead(nn.Module):
    def __init__(self, action_space_size: int = ACTION_SPACE_SIZE) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=2 * 8 * 8, out_features=action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 128, 8, 8)

        x = self.conv(x)
        # x: (batch_size, 2, 8, 8)

        x = self.bn(x)
        # x: (batch_size, 2, 8, 8)

        x = self.relu(x)
        # x: (batch_size, 2, 8, 8)

        x = torch.flatten(x, start_dim=1)
        # x: (batch_size, 128)

        logits = self.fc(x)
        # logits: (batch_size, action_space_size)

        return logits


def supervised_policy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.08,
) -> torch.Tensor:
    # logits: (batch_size, num_classes)
    # targets: (batch_size,)

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")

    num_classes: int = logits.size(1)

    log_probs = F.log_softmax(logits, dim=1)
    # log_probs: (batch_size, 4672)

    with torch.no_grad():
        smoothed_targets = torch.full_like(
            log_probs,
            fill_value=label_smoothing / (num_classes - 1),
        )
        # smoothed_targets: (batch_size, num_classes)

        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        # smoothed_targets: (batch_size, num_classes)

    loss = -(smoothed_targets * log_probs).sum(dim=1).mean()
    # loss: ()

    return loss


if __name__ == "__main__":
    x: torch.Tensor = torch.randn(4, 128, 8, 8)
    # x: (4, 128, 8, 8)

    targets: torch.Tensor = torch.tensor([12, 245, 1024, 4660], dtype=torch.long)
    # targets: (4,)

    policy_head = PolicyHead(action_space_size=ACTION_SPACE_SIZE)

    logits = policy_head(x)
    # logits: (4, 4672)

    probs = F.softmax(logits, dim=1)
    # probs: (4, 4672)

    loss = supervised_policy_loss(
        logits=logits,
        targets=targets,
        label_smoothing=0.08,
    )
    # loss: ()

    print("logits shape:", logits.shape)
    print("probs shape:", probs.shape)
    print("loss:", loss.item())
