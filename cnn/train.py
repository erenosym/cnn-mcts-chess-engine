from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cnn.data.multitask_data import (
    ChessMultiTaskDataset,
    MultiTaskTrainingConfig,
    load_multitask_games_from_pgn,
    split_multitask_games,
)
from cnn.policy_head import supervised_policy_loss
from cnn.policy_value_model import PolicyValueNetwork

PGN_PATH = PROJECT_ROOT / "lichess_elite_2025-01.pgn"
DEFAULT_BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEFAULT_MAX_CPU_THREADS = 4
DEFAULT_NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.08
VALUE_LOSS_WEIGHT = 1.0
POLICY_LOSS_WEIGHT = 1.0
MODEL_PATH = PROJECT_ROOT / "models" / "chess_policy_value_model.pt"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "chess_policy_value_model_best.pt"


@dataclass
class RuntimeSettings:
    batch_size: int
    max_cpu_threads: int
    num_workers: int


@dataclass
class EpochMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float


@dataclass
class TrainingSummary:
    epochs_completed: int
    best_epoch: int
    best_val_loss: float
    final_train_metrics: EpochMetrics
    final_val_metrics: EpochMetrics
    test_metrics: EpochMetrics


def make_loader(
    dataset: ChessMultiTaskDataset,
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
            batch_size=128,
            max_cpu_threads=min(6, cpu_count),
            num_workers=min(4, cpu_count),
        )

    return RuntimeSettings(
        batch_size=DEFAULT_BATCH_SIZE,
        max_cpu_threads=min(DEFAULT_MAX_CPU_THREADS, cpu_count),
        num_workers=min(DEFAULT_NUM_WORKERS, cpu_count),
    )


def run_epoch(
    model: PolicyValueNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    value_criterion: nn.Module,
    desc: str,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )

    for x, policy_targets, value_targets in progress:
        x = x.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)

        policy_logits, value_predictions = model(x)

        policy_loss = supervised_policy_loss(
            logits=policy_logits,
            targets=policy_targets,
            label_smoothing=LABEL_SMOOTHING,
        )
        value_loss = value_criterion(value_predictions, value_targets)
        total_loss = (POLICY_LOSS_WEIGHT * policy_loss) + (VALUE_LOSS_WEIGHT * value_loss)

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss_sum += total_loss.item() * batch_size
        policy_loss_sum += policy_loss.item() * batch_size
        value_loss_sum += value_loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(
            total=f"{total_loss.item():.4f}",
            policy=f"{policy_loss.item():.4f}",
            value=f"{value_loss.item():.4f}",
        )

    progress.close()

    if total_samples == 0:
        return EpochMetrics(total_loss=0.0, policy_loss=0.0, value_loss=0.0)

    return EpochMetrics(
        total_loss=total_loss_sum / total_samples,
        policy_loss=policy_loss_sum / total_samples,
        value_loss=value_loss_sum / total_samples,
    )


def main() -> None:
    if not PGN_PATH.exists():
        raise FileNotFoundError(f"PGN file not found: {PGN_PATH}")

    data_config = MultiTaskTrainingConfig()

    device = select_device()
    settings = select_runtime_settings(device)
    torch.set_num_threads(settings.max_cpu_threads)

    print(f"Using device: {device}")
    if data_config.max_games is None:
        print(f"Reading all filtered games from: {PGN_PATH}")
    else:
        print(f"Reading up to {data_config.max_games} games from: {PGN_PATH}")
    print(
        "Runtime settings -> "
        f"batch_size: {settings.batch_size}, "
        f"cpu_threads: {settings.max_cpu_threads}, "
        f"workers: {settings.num_workers}"
    )
    print(
        "Training filter -> "
        f"min_full_moves: {data_config.min_full_moves}, "
        f"max_full_moves: {data_config.max_full_moves}, "
        f"sample_every_plies: {data_config.sample_every_plies}, "
        f"max_samples_per_game: {data_config.max_samples_per_game}, "
        f"excluded terminations: {sorted(data_config.excluded_terminations)}"
    )
    print(
        "Loss weights -> "
        f"policy: {POLICY_LOSS_WEIGHT}, value: {VALUE_LOSS_WEIGHT}, "
        f"label_smoothing: {LABEL_SMOOTHING}"
    )

    games = load_multitask_games_from_pgn(PGN_PATH, data_config)
    if not games:
        raise ValueError("No valid multitask games were loaded from the PGN file.")

    train_games, val_games, test_games = split_multitask_games(games, data_config)

    train_dataset = ChessMultiTaskDataset(train_games)
    val_dataset = ChessMultiTaskDataset(val_games)
    test_dataset = ChessMultiTaskDataset(test_games)

    print(f"Loaded games: {len(games)}")
    print(
        "Game split -> "
        f"train: {len(train_games)}, val: {len(val_games)}, test: {len(test_games)}"
    )
    print(
        "Position split -> "
        f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
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

    model = PolicyValueNetwork().to(device)
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    last_train_metrics = EpochMetrics(0.0, 0.0, 0.0)
    last_val_metrics = EpochMetrics(0.0, 0.0, 0.0)
    epochs_completed = 0

    for epoch in range(EPOCHS):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            value_criterion=value_criterion,
            desc=f"Epoch {epoch + 1}/{EPOCHS} train",
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            value_criterion=value_criterion,
            desc=f"Epoch {epoch + 1}/{EPOCHS} val",
        )
        last_train_metrics = train_metrics
        last_val_metrics = val_metrics
        epochs_completed = epoch + 1

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"train_total: {train_metrics.total_loss:.4f} - "
            f"train_policy: {train_metrics.policy_loss:.4f} - "
            f"train_value: {train_metrics.value_loss:.4f} - "
            f"val_total: {val_metrics.total_loss:.4f} - "
            f"val_policy: {val_metrics.policy_loss:.4f} - "
            f"val_value: {val_metrics.value_loss:.4f}"
        )

        if val_metrics.total_loss < best_val_loss:
            best_val_loss = val_metrics.total_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(
                f"  New best model at epoch {best_epoch} "
                f"with val_total: {best_val_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping triggered "
                f"after {EARLY_STOPPING_PATIENCE} epochs without improvement."
            )
            break

    print(f"Best validation total loss: {best_val_loss:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        value_criterion=value_criterion,
        desc="Test",
    )
    summary = TrainingSummary(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_train_metrics=last_train_metrics,
        final_val_metrics=last_val_metrics,
        test_metrics=test_metrics,
    )

    torch.save(model.state_dict(), MODEL_PATH)
    print("Training summary")
    print(f"  epochs_completed: {summary.epochs_completed}")
    print(f"  best_epoch: {summary.best_epoch}")
    print(f"  best_val_total: {summary.best_val_loss:.4f}")
    print(
        "  last_train -> "
        f"total: {summary.final_train_metrics.total_loss:.4f}, "
        f"policy: {summary.final_train_metrics.policy_loss:.4f}, "
        f"value: {summary.final_train_metrics.value_loss:.4f}"
    )
    print(
        "  last_val -> "
        f"total: {summary.final_val_metrics.total_loss:.4f}, "
        f"policy: {summary.final_val_metrics.policy_loss:.4f}, "
        f"value: {summary.final_val_metrics.value_loss:.4f}"
    )
    print(
        "  test -> "
        f"total: {summary.test_metrics.total_loss:.4f}, "
        f"policy: {summary.test_metrics.policy_loss:.4f}, "
        f"value: {summary.test_metrics.value_loss:.4f}"
    )
    print(f"  best_checkpoint: {BEST_MODEL_PATH}")
    print(f"  final_model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
