from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    AlphaZero-style policy head.

    Input:
        x: Tensor of shape (batch_size, 128, 8, 8)

    Output:
        probs:  Tensor of shape (batch_size, 4672)
        logits: Tensor of shape (batch_size, 4672)
    """

    def __init__(self, action_space_size: int = 4672) -> None:
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        # logits: (batch_size, 4672)

        probs = F.softmax(logits, dim=1)
        # probs: (batch_size, 4672)

        return probs, logits


def policy_loss_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.08,
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing for noisy human chess moves.

    Args:
        logits:
            Tensor of shape (batch_size, 4672)
        targets:
            Long tensor of shape (batch_size,)
            Each entry is the correct action index in [0, 4671].
        smoothing:
            Label smoothing factor.

    Returns:
        Scalar loss tensor.
    """
    # logits: (batch_size, 4672)
    # targets: (batch_size,)

    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1).")

    num_classes: int = logits.size(1)

    log_probs = F.log_softmax(logits, dim=1)
    # log_probs: (batch_size, 4672)

    with torch.no_grad():
        true_dist = torch.full_like(log_probs, fill_value=smoothing / (num_classes - 1))
        # true_dist: (batch_size, 4672)

        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        # true_dist: (batch_size, 4672)

    loss = -(true_dist * log_probs).sum(dim=1).mean()
    # loss: scalar

    return loss


def training_step(
    policy_head: PolicyHead,
    tower_output: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    smoothing: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One supervised behavioral cloning training step.

    Args:
        policy_head:
            Policy head module.
        tower_output:
            Tensor of shape (batch_size, 128, 8, 8)
        targets:
            Long tensor of shape (batch_size,)
        optimizer:
            PyTorch optimizer
        smoothing:
            Label smoothing factor

    Returns:
        probs: Tensor of shape (batch_size, 4672)
        loss: Scalar tensor
    """
    policy_head.train()

    optimizer.zero_grad()

    probs, logits = policy_head(tower_output)
    # probs:  (batch_size, 4672)
    # logits: (batch_size, 4672)

    loss = policy_loss_with_label_smoothing(
        logits=logits,
        targets=targets,
        smoothing=smoothing,
    )
    # loss: scalar

    loss.backward()
    optimizer.step()

    return probs, loss


if __name__ == "__main__":
    batch_size: int = 4
    action_space_size: int = 4672

    policy_head = PolicyHead(action_space_size=action_space_size)

    optimizer = torch.optim.Adam(policy_head.parameters(), lr=1e-3)

    tower_output = torch.randn(batch_size, 128, 8, 8)
    # tower_output: (4, 128, 8, 8)

    targets = torch.randint(low=0, high=action_space_size, size=(batch_size,), dtype=torch.long)
    # targets: (4,)

    probs, loss = training_step(
        policy_head=policy_head,
        tower_output=tower_output,
        targets=targets,
        optimizer=optimizer,
        smoothing=0.08,
    )

    print("Policy probs shape:", probs.shape)   # (4, 4672)
    print("Loss:", loss.item())
