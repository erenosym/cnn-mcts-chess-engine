from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn.data.multitask_data import MultiTaskTrainingConfig, count_raw_and_sampled_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report raw and sampled position counts from a PGN.")
    parser.add_argument(
        "--pgn",
        type=Path,
        default=PROJECT_ROOT / "lichess_elite_2025-01.pgn",
        help="PGN path to analyze.",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Optional cap on games to scan.")
    parser.add_argument("--min-full-moves", type=int, default=0, help="Minimum full moves to keep a game.")
    parser.add_argument("--max-full-moves", type=int, default=None, help="Maximum full moves to keep a game.")
    parser.add_argument("--sample-every-plies", type=int, default=3, help="Keep one candidate every N plies.")
    parser.add_argument("--max-samples-per-game", type=int, default=24, help="Cap sampled positions per game.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultiTaskTrainingConfig(
        max_games=args.max_games,
        min_full_moves=args.min_full_moves,
        max_full_moves=args.max_full_moves,
        sample_every_plies=args.sample_every_plies,
        max_samples_per_game=args.max_samples_per_game,
    )

    start_time = time.time()
    games_seen, games_kept, raw_positions, sampled_positions = count_raw_and_sampled_positions(
        args.pgn,
        config,
    )
    elapsed = time.time() - start_time

    print(f"PGN: {args.pgn}")
    print(f"Games scanned: {games_seen}")
    print(f"Games kept: {games_kept}")
    print(f"Elapsed: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Games/sec: {games_seen / elapsed:.1f}")
    print()
    print("Sampling settings")
    print(f"  sample_every_plies: {config.sample_every_plies}")
    print(f"  max_samples_per_game: {config.max_samples_per_game}")
    print(f"  excluded_terminations: {sorted(config.excluded_terminations)}")
    print()
    print("Position counts")
    print(f"  raw_positions: {raw_positions}")
    print(f"  sampled_positions: {sampled_positions}")
    if raw_positions > 0:
        print(f"  sampled/raw ratio: {sampled_positions / raw_positions:.4f}")
    if games_kept > 0:
        print(f"  avg sampled per kept game: {sampled_positions / games_kept:.2f}")


if __name__ == "__main__":
    main()
