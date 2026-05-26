from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

from cnn.encoding import board_to_tensor


DEFAULT_EXCLUDED_TERMINATIONS = {"Time forfeit", "Abandoned", "Rules infraction"}


@dataclass
class ValueTrainingConfig:
    max_games: int | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 42
    min_full_moves: int = 21
    max_full_moves: int = 60
    skip_plies: int = 16
    sample_every: int = 4
    excluded_terminations: set[str] | None = None

    def __post_init__(self) -> None:
        if self.excluded_terminations is None:
            self.excluded_terminations = set(DEFAULT_EXCLUDED_TERMINATIONS)


@dataclass
class GameSample:
    positions: list[str]
    target: float


class ChessValueDataset(Dataset):
    def __init__(self, games: list[GameSample]):
        self.samples: list[tuple[str, float]] = []

        for game in games:
            for fen in game.positions:
                self.samples.append((fen, game.target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        fen, target = self.samples[index]
        board = chess.Board(fen)
        x = board_to_tensor(board)
        y = torch.tensor([target], dtype=torch.float32)
        return x, y


def result_to_target(result: str) -> float | None:
    mapping = {
        "1-0": 1.0,
        "0-1": -1.0,
        "1/2-1/2": 0.0,
    }
    return mapping.get(result)


def extract_positions_from_game(
    game: chess.pgn.Game,
    config: ValueTrainingConfig,
) -> GameSample | None:
    result = game.headers.get("Result", "")
    termination = game.headers.get("Termination", "")
    target = result_to_target(result)
    if target is None:
        return None
    if termination in config.excluded_terminations:
        return None

    board = game.board()
    positions: list[str] = []
    plies = 0

    for ply_index, move in enumerate(game.mainline_moves()):
        if ply_index >= config.skip_plies and (ply_index - config.skip_plies) % config.sample_every == 0:
            positions.append(board.fen())
        board.push(move)
        plies += 1

    full_moves = (plies + 1) // 2
    if full_moves < config.min_full_moves or full_moves > config.max_full_moves:
        return None
    if not positions:
        return None

    return GameSample(positions=positions, target=target)


def load_games_from_pgn(
    pgn_path: Path,
    config: ValueTrainingConfig,
) -> list[GameSample]:
    games: list[GameSample] = []

    with pgn_path.open("r", encoding="utf-8") as pgn_file:
        while config.max_games is None or len(games) < config.max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            extracted = extract_positions_from_game(game, config)
            if extracted is not None:
                games.append(extracted)

    return games


def split_games(
    games: list[GameSample],
    config: ValueTrainingConfig,
) -> tuple[list[GameSample], list[GameSample], list[GameSample]]:
    shuffled_games = games.copy()
    random.Random(config.random_seed).shuffle(shuffled_games)

    total_games = len(shuffled_games)
    train_end = int(total_games * config.train_ratio)
    val_end = train_end + int(total_games * config.val_ratio)

    train_games = shuffled_games[:train_end]
    val_games = shuffled_games[train_end:val_end]
    test_games = shuffled_games[val_end:]
    return train_games, val_games, test_games
