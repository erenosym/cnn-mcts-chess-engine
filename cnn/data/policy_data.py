from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from cnn.encoding import ACTION_SPACE_SIZE, board_to_tensor, move_to_action_index


DEFAULT_EXCLUDED_TERMINATIONS = {"Time forfeit", "Abandoned", "Rules infraction"}


@dataclass
class PolicyTrainingConfig:
    max_games: int | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 42
    excluded_terminations: set[str] | None = None

    def __post_init__(self) -> None:
        if self.excluded_terminations is None:
            self.excluded_terminations = set(DEFAULT_EXCLUDED_TERMINATIONS)


@dataclass
class PolicySample:
    fen: str
    action_index: int


@dataclass
class PolicyGame:
    samples: list[PolicySample]


class ChessPolicyDataset(Dataset):
    def __init__(self, samples: list[PolicySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        board = chess.Board(sample.fen)
        x = board_to_tensor(board)
        y = torch.tensor(sample.action_index, dtype=torch.long)
        return x, y


def extract_policy_samples_from_game(
    game: chess.pgn.Game,
    config: PolicyTrainingConfig,
) -> PolicyGame | None:
    termination = game.headers.get("Termination", "")
    if termination in config.excluded_terminations:
        return None

    board = game.board()
    samples: list[PolicySample] = []

    for move in game.mainline_moves():
        samples.append(
            PolicySample(
                fen=board.fen(),
                action_index=move_to_action_index(move),
            )
        )
        board.push(move)

    if not samples:
        return None

    return PolicyGame(samples=samples)


def load_policy_samples_from_pgn(
    pgn_path: Path,
    config: PolicyTrainingConfig,
) -> list[PolicySample]:
    games = load_policy_games_from_pgn(pgn_path, config)
    samples: list[PolicySample] = []

    for game in games:
        samples.extend(game.samples)

    return samples


def load_policy_games_from_pgn(
    pgn_path: Path,
    config: PolicyTrainingConfig,
) -> list[PolicyGame]:
    games: list[PolicyGame] = []
    games_loaded = 0
    progress = tqdm(
        total=config.max_games,
        desc="Loading policy games",
        unit="game",
        dynamic_ncols=True,
    )

    try:
        with pgn_path.open("r", encoding="utf-8") as pgn_file:
            while config.max_games is None or games_loaded < config.max_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                extracted = extract_policy_samples_from_game(game, config)
                if extracted is not None:
                    games.append(extracted)
                games_loaded += 1
                progress.update(1)
                progress.set_postfix(kept=len(games))
    finally:
        progress.close()

    return games


def split_policy_samples(
    samples: list[PolicySample],
    config: PolicyTrainingConfig,
) -> tuple[list[PolicySample], list[PolicySample], list[PolicySample]]:
    games = [PolicyGame(samples=[sample]) for sample in samples]
    return split_policy_games(games, config)


def split_policy_games(
    games: list[PolicyGame],
    config: PolicyTrainingConfig,
) -> tuple[list[PolicySample], list[PolicySample], list[PolicySample]]:
    shuffled_games = games.copy()
    random.Random(config.random_seed).shuffle(shuffled_games)

    total_games = len(shuffled_games)
    train_end = int(total_games * config.train_ratio)
    val_end = train_end + int(total_games * config.val_ratio)

    train_games = shuffled_games[:train_end]
    val_games = shuffled_games[train_end:val_end]
    test_games = shuffled_games[val_end:]

    def flatten(policy_games: list[PolicyGame]) -> list[PolicySample]:
        samples: list[PolicySample] = []
        for game in policy_games:
            samples.extend(game.samples)
        return samples

    return flatten(train_games), flatten(val_games), flatten(test_games)
