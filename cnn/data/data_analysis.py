from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PGN_PATH = PROJECT_ROOT / "lichess_elite_2025-01.pgn"
MAX_GAMES: int | None = None
OUTPUT_DIR = PROJECT_ROOT / "cnn" / "analysis_outputs"


@dataclass
class GameInfo:
    result: str
    termination: str
    plies: int
    full_moves: int
    finish_type: str
    draw_type: str | None


def move_bucket(full_moves: int) -> str:
    if full_moves <= 20:
        return "01-20"
    if full_moves <= 40:
        return "21-40"
    if full_moves <= 60:
        return "41-60"
    return "61+"


def classify_finish(
    board: chess.Board,
    result: str,
    termination: str,
) -> tuple[str, str | None]:
    termination = termination or "Unknown"

    if result == "1/2-1/2":
        if board.is_stalemate():
            return "draw", "stalemate"
        if board.is_insufficient_material():
            return "draw", "insufficient_material"
        if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
            return "draw", "repetition"
        if board.is_seventyfive_moves() or board.can_claim_fifty_moves():
            return "draw", "move_rule"
        return "draw", termination.lower().replace(" ", "_")

    if board.is_checkmate():
        return "checkmate", None

    if termination == "Time forfeit":
        return "time_forfeit", None

    if termination == "Abandoned":
        return "abandoned", None

    if termination == "Normal":
        return "resignation_or_normal", None

    return termination.lower().replace(" ", "_"), None


def read_game_info(game: chess.pgn.Game) -> GameInfo:
    board = game.board()
    plies = 0

    for move in game.mainline_moves():
        board.push(move)
        plies += 1

    result = game.headers.get("Result", "Unknown")
    termination = game.headers.get("Termination", "Unknown")
    finish_type, draw_type = classify_finish(board, result, termination)
    full_moves = (plies + 1) // 2

    return GameInfo(
        result=result,
        termination=termination,
        plies=plies,
        full_moves=full_moves,
        finish_type=finish_type,
        draw_type=draw_type,
    )


def load_game_infos(pgn_path: Path, max_games: int | None = MAX_GAMES) -> list[GameInfo]:
    game_infos: list[GameInfo] = []

    with pgn_path.open("r", encoding="utf-8") as pgn_file:
        while max_games is None or len(game_infos) < max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_infos.append(read_game_info(game))

    return game_infos


def print_counter(title: str, counter: Counter[str], total: int) -> None:
    print(title)
    for key, count in counter.most_common():
        percent = (count / total) * 100 if total else 0.0
        print(f"  {key}: {count} ({percent:.2f}%)")
    print()


def print_length_stats(game_infos: list[GameInfo]) -> None:
    total = len(game_infos)
    buckets = Counter(move_bucket(game.full_moves) for game in game_infos)
    print_counter("Game length buckets (full moves)", buckets, total)

    by_bucket = {
        bucket: [game.full_moves for game in game_infos if move_bucket(game.full_moves) == bucket]
        for bucket in ["01-20", "21-40", "41-60", "61+"]
    }

    print("Length details")
    for bucket, moves in by_bucket.items():
        if not moves:
            print(f"  {bucket}: no games")
            continue
        average_moves = sum(moves) / len(moves)
        print(
            f"  {bucket}: games={len(moves)}, min={min(moves)}, "
            f"max={max(moves)}, avg={average_moves:.2f}"
        )
    print()


def print_bucket_breakdown(game_infos: list[GameInfo]) -> None:
    print("Bucket breakdown by finish type")
    for bucket in ["01-20", "21-40", "41-60", "61+"]:
        bucket_games = [game for game in game_infos if move_bucket(game.full_moves) == bucket]
        counter = Counter(game.finish_type for game in bucket_games)
        print_counter(f"  {bucket}", counter, len(bucket_games))


def save_bar_chart(
    counter: Counter[str],
    title: str,
    file_path: Path,
    xlabel: str,
    ylabel: str = "Games",
) -> None:
    labels = list(counter.keys())
    values = list(counter.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="#33658A")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()


def save_length_histogram(game_infos: list[GameInfo], file_path: Path) -> None:
    lengths = [game.full_moves for game in game_infos]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color="#86BBD8", edgecolor="black")
    plt.title("Game Length Distribution")
    plt.xlabel("Full moves")
    plt.ylabel("Games")
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()


def save_bucket_finish_chart(game_infos: list[GameInfo], file_path: Path) -> None:
    buckets = ["01-20", "21-40", "41-60", "61+"]
    finish_types = sorted({game.finish_type for game in game_infos})
    colors = ["#33658A", "#55DDE0", "#2F4858", "#F6AE2D", "#F26419", "#9C89B8"]

    bucket_counters = {
        bucket: Counter(game.finish_type for game in game_infos if move_bucket(game.full_moves) == bucket)
        for bucket in buckets
    }

    x_positions = range(len(buckets))
    bottom = [0] * len(buckets)

    plt.figure(figsize=(11, 6))
    for color_index, finish_type in enumerate(finish_types):
        values = [bucket_counters[bucket].get(finish_type, 0) for bucket in buckets]
        plt.bar(
            x_positions,
            values,
            bottom=bottom,
            label=finish_type,
            color=colors[color_index % len(colors)],
        )
        bottom = [current + value for current, value in zip(bottom, values)]

    plt.title("Finish Types by Game Length Bucket")
    plt.xlabel("Full move bucket")
    plt.ylabel("Games")
    plt.xticks(list(x_positions), buckets)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()


def save_visualizations(game_infos: list[GameInfo]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_counter = Counter(game.result for game in game_infos)
    termination_counter = Counter(game.termination for game in game_infos)
    finish_counter = Counter(game.finish_type for game in game_infos)
    draw_games = [game for game in game_infos if game.finish_type == "draw"]
    draw_counter = Counter(game.draw_type or "unknown" for game in draw_games)
    length_bucket_counter = Counter(move_bucket(game.full_moves) for game in game_infos)

    save_bar_chart(
        result_counter,
        "Result Distribution",
        OUTPUT_DIR / "result_distribution.png",
        xlabel="Result",
    )
    save_bar_chart(
        termination_counter,
        "Termination Header Distribution",
        OUTPUT_DIR / "termination_distribution.png",
        xlabel="Termination",
    )
    save_bar_chart(
        finish_counter,
        "Finish Type Distribution",
        OUTPUT_DIR / "finish_type_distribution.png",
        xlabel="Finish type",
    )
    save_bar_chart(
        draw_counter,
        "Draw Breakdown",
        OUTPUT_DIR / "draw_breakdown.png",
        xlabel="Draw type",
    )
    save_bar_chart(
        length_bucket_counter,
        "Game Length Buckets",
        OUTPUT_DIR / "length_buckets.png",
        xlabel="Full move bucket",
    )
    save_length_histogram(
        game_infos,
        OUTPUT_DIR / "length_histogram.png",
    )
    save_bucket_finish_chart(
        game_infos,
        OUTPUT_DIR / "finish_by_length_bucket.png",
    )


def main() -> None:
    if not PGN_PATH.exists():
        raise FileNotFoundError(f"PGN file not found: {PGN_PATH}")

    game_infos = load_game_infos(PGN_PATH, max_games=MAX_GAMES)
    if not game_infos:
        raise ValueError("No games were read from the PGN file.")

    total_games = len(game_infos)
    print(f"PGN path: {PGN_PATH}")
    if MAX_GAMES is None:
        print(f"Games analyzed: {total_games} (full file)")
    else:
        print(f"Games analyzed: {total_games}")
    print()

    result_counter = Counter(game.result for game in game_infos)
    print_counter("Result distribution", result_counter, total_games)

    termination_counter = Counter(game.termination for game in game_infos)
    print_counter("Termination header distribution", termination_counter, total_games)

    finish_counter = Counter(game.finish_type for game in game_infos)
    print_counter("Finish type distribution", finish_counter, total_games)

    draw_games = [game for game in game_infos if game.finish_type == "draw"]
    draw_counter = Counter(game.draw_type or "unknown" for game in draw_games)
    print_counter("Draw breakdown", draw_counter, len(draw_games))

    print_length_stats(game_infos)
    print_bucket_breakdown(game_infos)
    save_visualizations(game_infos)
    print(f"Charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
