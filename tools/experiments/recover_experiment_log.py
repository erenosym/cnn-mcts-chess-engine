from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


GAME_RE = re.compile(
    r"^\[(?P<experiment>[^\]]+)\]\s+"
    r"(?P<start_position>.+?)\s+\|\s+"
    r"(?P<matchup>.+?)\s+\|\s+"
    r"game\s+(?P<game_index>\d+)/(?P<games>\d+)\s+\|\s+"
    r"white=(?P<white>.+?)\s+black=(?P<black>.+?)\s+\|\s+"
    r"winner=(?P<winner>.+?)\s+\|\s+"
    r"result=(?P<result>\S+)\s+plies=(?P<plies>\d+)\s*$"
)
SUMMARY_RE = re.compile(r"^Summary for (?P<matchup>.+?) \((?P<start_position>.+)\):$")
WINS_RE = re.compile(r"^\s+(?P<bot>.+?) wins: (?P<wins>\d+)$")
DRAWS_RE = re.compile(r"^\s+Draws: (?P<draws>\d+)$")
SCORE_RE = re.compile(r"^\s+(?P<bot>.+?) score rate: (?P<score_rate>\d+(?:\.\d+)?)$")
PLIES_RE = re.compile(r"^\s+Average plies: (?P<average_plies>\d+(?:\.\d+)?)$")


@dataclass
class GameResult:
    experiment: str
    start_position: str
    matchup: str
    game_index: int
    white: str
    black: str
    winner: str
    result: str
    plies: int
    fen: str = ""


@dataclass
class MatchupSummary:
    experiment: str
    start_position: str
    matchup: str
    games: int
    first_bot_wins: int
    second_bot_wins: int
    draws: int
    first_bot_score_rate: float
    average_plies: float


def score_for_first(result: GameResult) -> float:
    first_bot = result.matchup.split(" vs ", 1)[0]
    if result.winner == "Draw":
        return 0.5
    return 1.0 if result.winner == first_bot else 0.0


def recover(log_path: Path, output_dir: Path) -> None:
    game_results: list[GameResult] = []
    printed_summaries: list[MatchupSummary] = []
    active_summary: dict[str, object] | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = GAME_RE.match(line)
        if not match:
            summary_match = SUMMARY_RE.match(line)
            if summary_match:
                active_summary = {
                    "matchup": summary_match.group("matchup"),
                    "start_position": summary_match.group("start_position"),
                    "wins": {},
                }
                continue

            if active_summary is None:
                continue

            wins_match = WINS_RE.match(raw_line)
            if wins_match:
                active_summary["wins"][wins_match.group("bot")] = int(wins_match.group("wins"))
                continue

            draws_match = DRAWS_RE.match(raw_line)
            if draws_match:
                active_summary["draws"] = int(draws_match.group("draws"))
                continue

            score_match = SCORE_RE.match(raw_line)
            if score_match:
                active_summary["first_bot"] = score_match.group("bot")
                active_summary["score_rate"] = float(score_match.group("score_rate"))
                continue

            plies_match = PLIES_RE.match(raw_line)
            if plies_match:
                matchup = str(active_summary["matchup"])
                first_bot, second_bot = matchup.split(" vs ", 1)
                draws = int(active_summary["draws"])
                first_bot_wins = int(active_summary["wins"].get(first_bot, 0))
                second_bot_wins = int(active_summary["wins"].get(second_bot, 0))
                printed_summaries.append(
                    MatchupSummary(
                        experiment="",
                        start_position=str(active_summary["start_position"]),
                        matchup=matchup,
                        games=first_bot_wins + second_bot_wins + draws,
                        first_bot_wins=first_bot_wins,
                        second_bot_wins=second_bot_wins,
                        draws=draws,
                        first_bot_score_rate=float(active_summary["score_rate"]),
                        average_plies=float(plies_match.group("average_plies")),
                    )
                )
                active_summary = None
            continue

        data = match.groupdict()
        game_results.append(
            GameResult(
                experiment=data["experiment"],
                start_position=data["start_position"],
                matchup=data["matchup"],
                game_index=int(data["game_index"]),
                white=data["white"],
                black=data["black"],
                winner=data["winner"],
                result=data["result"],
                plies=int(data["plies"]),
            )
        )

    if not game_results:
        raise SystemExit(f"No game result lines found in {log_path}")

    grouped: dict[tuple[str, str, str], list[GameResult]] = defaultdict(list)
    for result in game_results:
        grouped[(result.experiment, result.start_position, result.matchup)].append(result)

    summaries_by_key: dict[tuple[str, str, str], MatchupSummary] = {}
    for (experiment, start_position, matchup), rows in grouped.items():
        first_bot, second_bot = matchup.split(" vs ", 1)
        first_bot_wins = sum(1 for row in rows if row.winner == first_bot)
        second_bot_wins = sum(1 for row in rows if row.winner == second_bot)
        draws = sum(1 for row in rows if row.winner == "Draw")
        games = len(rows)
        summaries_by_key[(experiment, start_position, matchup)] = MatchupSummary(
                experiment=experiment,
                start_position=start_position,
                matchup=matchup,
                games=games,
                first_bot_wins=first_bot_wins,
                second_bot_wins=second_bot_wins,
                draws=draws,
                first_bot_score_rate=sum(score_for_first(row) for row in rows) / games,
                average_plies=sum(row.plies for row in rows) / games,
        )

    experiments_by_summary_key = {
        (row.start_position, row.matchup): row.experiment for row in game_results
    }
    for printed in printed_summaries:
        experiment = experiments_by_summary_key.get((printed.start_position, printed.matchup), "")
        if not experiment:
            experiment = "standard_start" if printed.start_position == "Initial position" else "opening_positions"
        printed.experiment = experiment
        summaries_by_key[(printed.experiment, printed.start_position, printed.matchup)] = printed

    summaries = list(summaries_by_key.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    game_csv_path = output_dir / "game_results.csv"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    with game_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(game_results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in game_results)

    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in summaries)

    with summary_json_path.open("w", encoding="utf-8") as json_file:
        json.dump([asdict(row) for row in summaries], json_file, indent=2)

    print(f"Recovered games: {len(game_results)}")
    print(f"Recovered summary rows: {len(summaries)}")
    print(f"Wrote game results: {game_csv_path}")
    print(f"Wrote summaries: {summary_csv_path}")
    print(f"Wrote JSON summary: {summary_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover experiment CSV files from terminal logs.")
    parser.add_argument("log_path", type=Path, help="Text file containing copied terminal output.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/recovered_from_log"),
        help="Directory for recovered CSV and JSON files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recover(args.log_path, args.output_dir)
