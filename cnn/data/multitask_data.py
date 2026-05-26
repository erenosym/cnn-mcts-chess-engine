from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from cnn.data.value_data import result_to_target
from cnn.encoding import board_to_tensor, move_to_action_index


DEFAULT_EXCLUDED_TERMINATIONS = {"Time forfeit", "Abandoned", "Rules infraction"}


@dataclass
class MultiTaskTrainingConfig:
    max_games: int | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 42
    min_full_moves: int = 0
    max_full_moves: int | None = None
    sample_every_plies: int = 3
    max_samples_per_game: int = 24
    excluded_terminations: set[str] | None = None

    def __post_init__(self) -> None:
        if self.excluded_terminations is None:
            self.excluded_terminations = set(DEFAULT_EXCLUDED_TERMINATIONS)


@dataclass
class MultiTaskSample:
    fen: str
    action_index: int
    value_target: float
    ply_index: int


@dataclass
class MultiTaskGame:
    samples: list[MultiTaskSample]


class ChessMultiTaskDataset(Dataset):
    def __init__(self, games: list[MultiTaskGame]):
        self.samples: list[MultiTaskSample] = []
        for game in games:
            self.samples.extend(game.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        board = chess.Board(sample.fen)
        x = board_to_tensor(board)
        policy_target = torch.tensor(sample.action_index, dtype=torch.long)
        value_target = torch.tensor([sample.value_target], dtype=torch.float32)
        return x, policy_target, value_target


def _evenly_spaced_subset(
    samples: list[MultiTaskSample],
    limit: int,
) -> list[MultiTaskSample]:
    if limit <= 0:
        return []
    if len(samples) <= limit:
        return samples

    if limit == 1:
        return [samples[len(samples) // 2]]

    last_index = len(samples) - 1
    chosen_indices = {
        round(index * last_index / (limit - 1))
        for index in range(limit)
    }
    return [sample for index, sample in enumerate(samples) if index in chosen_indices]


def extract_multitask_game(
    game: chess.pgn.Game,
    config: MultiTaskTrainingConfig,
) -> MultiTaskGame | None:
    result = game.headers.get("Result", "")
    termination = game.headers.get("Termination", "")
    value_target = result_to_target(result)
    if value_target is None:
        return None
    if termination in config.excluded_terminations:
        return None

    moves = list(game.mainline_moves())
    full_moves = (len(moves) + 1) // 2
    if full_moves < config.min_full_moves:
        return None
    if config.max_full_moves is not None and full_moves > config.max_full_moves:
        return None

    board = game.board()
    candidate_samples: list[MultiTaskSample] = []

    for ply_index, move in enumerate(moves):
        if ply_index % config.sample_every_plies == 0:
            candidate_samples.append(
            MultiTaskSample(
                fen=board.fen(),
                action_index=move_to_action_index(move),
                value_target=value_target,
                ply_index=ply_index,
            )
            )
        board.push(move)

    samples = _evenly_spaced_subset(
        candidate_samples,
        limit=config.max_samples_per_game,
    )

    if not samples:
        return None

    return MultiTaskGame(samples=samples)


def load_multitask_games_from_pgn(
    pgn_path: Path,
    config: MultiTaskTrainingConfig,
) -> list[MultiTaskGame]:
    games: list[MultiTaskGame] = []
    progress = tqdm(
        total=config.max_games,
        desc="Loading multitask games",
        unit="game",
        dynamic_ncols=True,
    )

    try:
        with pgn_path.open("r", encoding="utf-8") as pgn_file:
            while config.max_games is None or len(games) < config.max_games:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                extracted = extract_multitask_game(game, config)
                if extracted is not None:
                    games.append(extracted)
                progress.update(1)
                progress.set_postfix(kept=len(games))
    finally:
        progress.close()

    return games


def count_raw_and_sampled_positions(
    pgn_path: Path,
    config: MultiTaskTrainingConfig,
) -> tuple[int, int, int, int]:
    games_seen = 0
    games_kept = 0
    raw_positions = 0
    sampled_positions = 0

    with pgn_path.open("r", encoding="utf-8") as pgn_file:
        while config.max_games is None or games_seen < config.max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            games_seen += 1
            moves = list(game.mainline_moves())
            result = game.headers.get("Result", "")
            termination = game.headers.get("Termination", "")
            value_target = result_to_target(result)
            full_moves = (len(moves) + 1) // 2

            if value_target is None:
                continue
            if termination in config.excluded_terminations:
                continue
            if full_moves < config.min_full_moves:
                continue
            if config.max_full_moves is not None and full_moves > config.max_full_moves:
                continue

            raw_positions += len(moves)
            sampled_positions += min(
                config.max_samples_per_game,
                (len(moves) + config.sample_every_plies - 1) // config.sample_every_plies,
            )
            games_kept += 1

    return games_seen, games_kept, raw_positions, sampled_positions


def split_multitask_games(
    games: list[MultiTaskGame],
    config: MultiTaskTrainingConfig,
) -> tuple[list[MultiTaskGame], list[MultiTaskGame], list[MultiTaskGame]]:
    shuffled_games = games.copy()
    random.Random(config.random_seed).shuffle(shuffled_games)

    total_games = len(shuffled_games)
    train_end = int(total_games * config.train_ratio)
    val_end = train_end + int(total_games * config.val_ratio)

    train_games = shuffled_games[:train_end]
    val_games = shuffled_games[train_end:val_end]
    test_games = shuffled_games[val_end:]
    return train_games, val_games, test_games
