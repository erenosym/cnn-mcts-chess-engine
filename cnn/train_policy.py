from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cnn.data.policy_data import (
    ACTION_SPACE_SIZE,
    ChessPolicyDataset,
    PolicyTrainingConfig,
    load_policy_games_from_pgn,
    split_policy_games,
)
from cnn.policy_head import supervised_policy_loss
from cnn.policy_model import PolicyNetwork


PGN_PATH = PROJECT_ROOT / "lichess_elite_2025-01.pgn"
DEFAULT_BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
DEFAULT_MAX_CPU_THREADS = 4
DEFAULT_NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.08
MODEL_PATH = Path("cnn/chess_policy_model.pt")
BEST_MODEL_PATH = Path("cnn/chess_policy_model_best.pt")


@dataclass
class RuntimeSettings:
    batch_size: int
    max_cpu_threads: int
    num_workers: int


@dataclass
class TrainingSummary:
    epochs_completed: int
    best_epoch: int
    best_val_loss: float
    last_train_loss: float
    last_val_loss: float
    test_loss: float


def make_loader(
    dataset: ChessPolicyDataset,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_runtime_settings(device: torch.device) -> RuntimeSettings:
    cpu_count = os.cpu_count() or 1

    if device.type == "mps":
        return RuntimeSettings(
            batch_size=192,
            max_cpu_threads=min(10, cpu_count),
            num_workers=min(6, cpu_count),
        )

    if device.type == "cuda":
        return RuntimeSettings(
            batch_size=256,
            max_cpu_threads=min(6, cpu_count),
            num_workers=min(4, cpu_count),
        )

    return RuntimeSettings(
        batch_size=DEFAULT_BATCH_SIZE,
        max_cpu_threads=min(DEFAULT_MAX_CPU_THREADS, cpu_count),
        num_workers=min(DEFAULT_NUM_WORKERS, cpu_count),
    )


def run_epoch(
    model: PolicyNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    label_smoothing: float,
    desc: str,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )

    for x, targets in progress:
        x = x.to(device)
        targets = targets.to(device)
        # x: (batch_size, 13, 8, 8)
        # targets: (batch_size,)

        logits = model(x)
        # logits: (batch_size, action_space_size)

        loss = supervised_policy_loss(
            logits=logits,
            targets=targets,
            label_smoothing=label_smoothing,
        )
        # loss: ()

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    progress.close()

    if total_samples == 0:
        return 0.0

    return total_loss / total_samples


def main() -> None:
    if not PGN_PATH.exists():
        raise FileNotFoundError(f"PGN file not found: {PGN_PATH}")

    data_config = PolicyTrainingConfig()

    device = select_device()
    settings = select_runtime_settings(device)
    torch.set_num_threads(settings.max_cpu_threads)

    print(f"Using device: {device}")
    if data_config.max_games is None:
        print(f"Reading all policy games from: {PGN_PATH}")
    else:
        print(f"Reading up to {data_config.max_games} games from: {PGN_PATH}")
    print(
        "Runtime settings -> "
        f"batch_size: {settings.batch_size}, "
        f"cpu_threads: {settings.max_cpu_threads}, "
        f"workers: {settings.num_workers}"
    )
    print(
        "Policy filter -> "
        f"excluded terminations: {sorted(data_config.excluded_terminations)}"
    )
    print(f"Label smoothing: {LABEL_SMOOTHING}")

    games = load_policy_games_from_pgn(PGN_PATH, data_config)
    if not games:
        raise ValueError("No valid policy games were loaded from the PGN file.")

    train_samples, val_samples, test_samples = split_policy_games(games, data_config)

    train_dataset = ChessPolicyDataset(train_samples)
    val_dataset = ChessPolicyDataset(val_samples)
    test_dataset = ChessPolicyDataset(test_samples)

    print(f"Loaded policy games: {len(games)}")
    print(
        "Sample split (game-based) -> "
        f"train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}"
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        shuffle=False,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        shuffle=False,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )

    model = PolicyNetwork(action_space_size=ACTION_SPACE_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    last_train_loss = 0.0
    last_val_loss = 0.0
    epochs_completed = 0

    for epoch in range(EPOCHS):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            label_smoothing=LABEL_SMOOTHING,
            desc=f"Epoch {epoch + 1}/{EPOCHS} train",
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            label_smoothing=LABEL_SMOOTHING,
            desc=f"Epoch {epoch + 1}/{EPOCHS} val",
        )
        last_train_loss = train_loss
        last_val_loss = val_loss
        epochs_completed = epoch + 1
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(
                f"  New best model at epoch {best_epoch} "
                f"with val_loss: {best_val_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping triggered "
                f"after {EARLY_STOPPING_PATIENCE} epochs without improvement."
            )
            break

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_loss = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        label_smoothing=LABEL_SMOOTHING,
        desc="Test",
    )
    summary = TrainingSummary(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        last_train_loss=last_train_loss,
        last_val_loss=last_val_loss,
        test_loss=test_loss,
    )

    torch.save(model.state_dict(), MODEL_PATH)
    print("Training summary")
    print(f"  epochs_completed: {summary.epochs_completed}")
    print(f"  best_epoch: {summary.best_epoch}")
    print(f"  best_val_loss: {summary.best_val_loss:.4f}")
    print(f"  last_train_loss: {summary.last_train_loss:.4f}")
    print(f"  last_val_loss: {summary.last_val_loss:.4f}")
    print(f"  test_loss: {summary.test_loss:.4f}")
    print(f"  best_checkpoint: {BEST_MODEL_PATH}")
    print(f"  final_model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
