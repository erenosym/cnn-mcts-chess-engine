from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import chess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.cnn_mcts_bot import choose_cnn_mcts_move
from bots.mcts_bot import choose_mcts_move
from bots.random_bot import choose_random_move
from cnn.inference import PolicyValueEvaluator


OPENING_GAMES_PER_MATCHUP = 20
STANDARD_GAMES_PER_MATCHUP = 100
DEFAULT_SIMULATIONS = 800
DEFAULT_C_PUCT = 1.4


@dataclass(frozen=True)
class OpeningPosition:
    eco: str
    name: str
    moves: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.eco}: {self.name}"

    def board(self) -> chess.Board:
        board = chess.Board()
        for san in self.moves:
            board.push_san(san)
        return board

    def fen(self) -> str:
        return self.board().fen()


OPENING_POSITIONS: tuple[OpeningPosition, ...] = (
    OpeningPosition(
        eco="A10",
        name="English Opening",
        moves=("c4", "e5", "g3", "d5", "cxd5", "Nf6", "Bg2", "Nxd5", "Nf3"),
    ),
    OpeningPosition(
        eco="D06",
        name="Queen's Gambit",
        moves=("d4", "d5", "c4", "c6", "Nc3", "Nf6", "Nf3", "a6", "g3", "dxc4", "a4"),
    ),
    OpeningPosition(
        eco="A46",
        name="Queen's Pawn Game",
        moves=("d4", "Nf6", "Nf3", "d5", "c4", "e6", "Nc3", "Be7", "Bf4", "O-O", "e3"),
    ),
    OpeningPosition(
        eco="E00",
        name="Queen's Pawn Game",
        moves=("d4", "Nf6", "c4", "e6", "Nf3", "d5", "Nc3", "Bb4", "Bg5", "h6", "Qa4+", "Nc6"),
    ),
    OpeningPosition(
        eco="E61",
        name="King's Indian Defence",
        moves=("d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5", "e4", "Nxc3", "bxc3", "Bg7", "Be3"),
    ),
    OpeningPosition(
        eco="C00",
        name="French Defence",
        moves=("e4", "e6", "d4", "d5", "Nc3", "Nf6", "e5", "Nfd7", "f4", "c5", "Nf3", "Be7"),
    ),
    OpeningPosition(
        eco="B50",
        name="Sicilian Defence",
        moves=("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "f3", "e5"),
    ),
    OpeningPosition(
        eco="B30",
        name="Sicilian Defence",
        moves=("e4", "c5", "Nf3", "Nc6", "Bb5", "e6", "O-O", "Nge7", "Re1", "a6", "Bf1", "d5"),
    ),
    OpeningPosition(
        eco="B40",
        name="Sicilian Defence",
        moves=("e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "Nc6", "Nc3", "Qc7", "Be3", "a6"),
    ),
    OpeningPosition(
        eco="C60",
        name="Ruy Lopez",
        moves=("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Be7", "O-O", "Nf6", "Re1", "b5", "Bb3", "O-O"),
    ),
    OpeningPosition(
        eco="B10",
        name="Caro-Kann Defence",
        moves=("e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6", "Be2", "a6"),
    ),
    OpeningPosition(
        eco="A05",
        name="Reti Opening",
        moves=("Nf3", "Nf6", "c4", "e6", "d4", "d5", "Nc3", "Be7", "Bf4", "O-O"),
    ),
)


MATCHUPS: tuple[tuple[str, str], ...] = (
    ("cnn_mcts", "random"),
    ("mcts", "random"),
    ("cnn_mcts", "mcts"),
)

MATCHUP_KEYS = {
    f"{first_bot}-{second_bot}": (first_bot, second_bot)
    for first_bot, second_bot in MATCHUPS
}


BOT_LABELS = {
    "cnn_mcts": "CNN+MCTS",
    "mcts": "Baseline MCTS",
    "random": "Random",
}


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


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
    fen: str


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


def validate_openings() -> None:
    print("Validating opening positions...")
    for opening in OPENING_POSITIONS:
        opening.board()
    print(f"Validated {len(OPENING_POSITIONS)} opening positions.")


def choose_move(
    bot: str,
    board: chess.Board,
    evaluator: PolicyValueEvaluator | None,
    simulations: int,
    c_puct: float,
) -> chess.Move | None:
    if bot == "random":
        return choose_random_move(board)
    if bot == "mcts":
        return choose_mcts_move(board, n_simulations=simulations)
    if bot == "cnn_mcts":
        if evaluator is None:
            raise ValueError("CNN+MCTS requires a PolicyValueEvaluator.")
        return choose_cnn_mcts_move(
            board,
            evaluator=evaluator,
            n_simulations=simulations,
            c_puct=c_puct,
        )
    raise ValueError(f"Unknown bot: {bot}")


def play_game(
    starting_board: chess.Board,
    white_bot: str,
    black_bot: str,
    evaluator: PolicyValueEvaluator | None,
    simulations: int,
    c_puct: float,
) -> chess.Board:
    board = starting_board.copy(stack=False)

    while not board.is_game_over():
        bot = white_bot if board.turn == chess.WHITE else black_bot
        move = choose_move(
            bot=bot,
            board=board,
            evaluator=evaluator,
            simulations=simulations,
            c_puct=c_puct,
        )
        if move is None:
            break
        board.push(move)

    return board


def result_winner(result: str, white_bot: str, black_bot: str) -> str:
    if result == "1-0":
        return white_bot
    if result == "0-1":
        return black_bot
    return "draw"


def run_matchup(
    experiment: str,
    start_position: str,
    starting_board_factory: Callable[[], chess.Board],
    first_bot: str,
    second_bot: str,
    games: int,
    evaluator: PolicyValueEvaluator | None,
    simulations: int,
    c_puct: float,
    game_start_index: int = 1,
) -> tuple[list[GameResult], MatchupSummary]:
    if games % 2 != 0:
        raise ValueError("Game count must be even so colors can be balanced.")

    results: list[GameResult] = []
    matchup = f"{BOT_LABELS[first_bot]} vs {BOT_LABELS[second_bot]}"

    print_section(f"Running matchup: {matchup}")
    print(f"Experiment: {experiment}")
    print(f"Start position: {start_position}")
    print(f"Games: {games}")
    print(f"MCTS simulations per move: {simulations}")
    print(f"CNN+MCTS c_puct: {c_puct}")
    print("Color balance: first bot alternates between White and Black.")

    first_bot_wins = 0
    second_bot_wins = 0
    draws = 0
    total_plies = 0

    game_end_index = game_start_index + games - 1
    for game_offset in range(games):
        game_index = game_start_index + game_offset
        first_bot_is_white = (game_index - 1) % 2 == 0
        white_bot = first_bot if first_bot_is_white else second_bot
        black_bot = second_bot if first_bot_is_white else first_bot

        print(
            f"Starting game {game_index}/{game_end_index}: "
            f"white={BOT_LABELS[white_bot]} black={BOT_LABELS[black_bot]}",
            flush=True,
        )
        board = play_game(
            starting_board=starting_board_factory(),
            white_bot=white_bot,
            black_bot=black_bot,
            evaluator=evaluator,
            simulations=simulations,
            c_puct=c_puct,
        )
        result = board.result()
        winner = result_winner(result, white_bot=white_bot, black_bot=black_bot)
        if winner == first_bot:
            first_bot_wins += 1
        elif winner == second_bot:
            second_bot_wins += 1
        else:
            draws += 1

        total_plies += board.ply()
        results.append(
            GameResult(
                experiment=experiment,
                start_position=start_position,
                matchup=matchup,
                game_index=game_index,
                white=BOT_LABELS[white_bot],
                black=BOT_LABELS[black_bot],
                winner=BOT_LABELS.get(winner, "Draw"),
                result=result,
                plies=board.ply(),
                fen=board.fen(),
            )
        )

        print(
            f"[{experiment}] {start_position} | {matchup} | "
            f"game {game_index}/{game_end_index} | "
            f"white={BOT_LABELS[white_bot]} black={BOT_LABELS[black_bot]} | "
            f"winner={BOT_LABELS.get(winner, 'Draw')} | "
            f"result={result} plies={board.ply()}",
            flush=True,
        )

    score_rate = (first_bot_wins + 0.5 * draws) / games
    summary = MatchupSummary(
        experiment=experiment,
        start_position=start_position,
        matchup=matchup,
        games=games,
        first_bot_wins=first_bot_wins,
        second_bot_wins=second_bot_wins,
        draws=draws,
        first_bot_score_rate=score_rate,
        average_plies=total_plies / games,
    )
    print()
    print(f"Summary for {matchup} ({start_position}):")
    print(f"  {BOT_LABELS[first_bot]} wins: {first_bot_wins}")
    print(f"  {BOT_LABELS[second_bot]} wins: {second_bot_wins}")
    print(f"  Draws: {draws}")
    print(f"  {BOT_LABELS[first_bot]} score rate: {score_rate:.3f}")
    print(f"  Average plies: {summary.average_plies:.1f}")
    return results, summary


def write_outputs(
    output_dir: Path,
    game_results: list[GameResult],
    summaries: list[MatchupSummary],
) -> None:
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

    print()
    print(f"Wrote game results: {game_csv_path}")
    print(f"Wrote summaries: {summary_csv_path}")
    print(f"Wrote JSON summary: {summary_json_path}")


def print_openings() -> None:
    for opening in OPENING_POSITIONS:
        print(f"{opening.label}")
        print(f"  moves: {' '.join(opening.moves)}")
        print(f"  fen:   {opening.fen()}")


def run_experiments(args: argparse.Namespace) -> None:
    print_section("Bot Experiment Suite")
    print(f"Standard games per matchup: {args.standard_games}")
    print(f"Opening games per matchup: {args.opening_games}")
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"CNN+MCTS c_puct: {args.c_puct}")
    print(f"Output directory: {args.output_dir}")
    print(f"Include standard start: {args.include_standard}")
    print(f"Include opening positions: {args.include_openings}")

    validate_openings()
    if args.list_openings:
        print_openings()
        return

    needs_cnn = any("cnn_mcts" in matchup for matchup in args.selected_matchups)
    if needs_cnn:
        print()
        print("Loading CNN policy-value evaluator...")
    evaluator = PolicyValueEvaluator() if needs_cnn else None
    if needs_cnn:
        print("CNN policy-value evaluator loaded.")

    game_results: list[GameResult] = []
    summaries: list[MatchupSummary] = []

    if args.include_standard:
        print_section("Standard Initial Position Experiments")
        for first_bot, second_bot in args.selected_matchups:
            results, summary = run_matchup(
                experiment="standard_start",
                start_position="Initial position",
                starting_board_factory=chess.Board,
                first_bot=first_bot,
                second_bot=second_bot,
                games=args.standard_games,
                evaluator=evaluator,
                simulations=args.simulations,
                c_puct=args.c_puct,
                game_start_index=args.game_start,
            )
            game_results.extend(results)
            summaries.append(summary)

    if args.include_openings:
        print_section("Opening Position Experiments")
        for opening in args.selected_openings:
            print()
            print(f"Opening selected: {opening.label}")
            print(f"Moves: {' '.join(opening.moves)}")
            print(f"FEN: {opening.fen()}")
            for first_bot, second_bot in args.selected_matchups:
                results, summary = run_matchup(
                    experiment="opening_positions",
                    start_position=opening.label,
                    starting_board_factory=opening.board,
                    first_bot=first_bot,
                    second_bot=second_bot,
                    games=args.opening_games,
                    evaluator=evaluator,
                    simulations=args.simulations,
                    c_puct=args.c_puct,
                    game_start_index=args.game_start,
                )
                game_results.extend(results)
                summaries.append(summary)

    if game_results:
        write_outputs(args.output_dir, game_results, summaries)
        print_section("Experiment Completed")
        print(f"Total games played: {len(game_results)}")
        print(f"Summary rows written: {len(summaries)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run balanced bot comparison experiments for the graduation project."
    )
    parser.add_argument(
        "--opening-games",
        type=int,
        default=OPENING_GAMES_PER_MATCHUP,
        help="Games per matchup for each selected opening position. Default: 20.",
    )
    parser.add_argument(
        "--standard-games",
        type=int,
        default=STANDARD_GAMES_PER_MATCHUP,
        help="Games per matchup from the initial chess position. Default: 100.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help="MCTS simulations per move for both Baseline MCTS and CNN+MCTS. Default: 800.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=DEFAULT_C_PUCT,
        help="CNN+MCTS exploration constant. Default: 1.4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_outputs" / "experiments",
        help="Directory for CSV and JSON experiment outputs.",
    )
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Run only the initial-position experiment.",
    )
    parser.add_argument(
        "--openings-only",
        action="store_true",
        help="Run only the opening-position experiment.",
    )
    parser.add_argument(
        "--list-openings",
        action="store_true",
        help="Print opening move lists and generated FENs without running games.",
    )
    parser.add_argument(
        "--matchups",
        default=",".join(MATCHUP_KEYS),
        help=(
            "Comma-separated matchup keys to run. "
            f"Available: {', '.join(MATCHUP_KEYS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--opening-start",
        type=int,
        default=1,
        help="1-based opening index to start from when running opening experiments. Default: 1.",
    )
    parser.add_argument(
        "--opening-end",
        type=int,
        default=len(OPENING_POSITIONS),
        help=f"1-based opening index to end at when running opening experiments. Default: {len(OPENING_POSITIONS)}.",
    )
    parser.add_argument(
        "--game-start",
        type=int,
        default=1,
        help="1-based game index to start from within each selected matchup. Default: 1.",
    )
    args = parser.parse_args()

    if args.opening_games % 2 != 0 or args.standard_games % 2 != 0:
        parser.error("--opening-games and --standard-games must be even for color balance.")

    args.include_standard = not args.openings_only
    args.include_openings = not args.standard_only

    matchup_keys = [key.strip() for key in args.matchups.split(",") if key.strip()]
    invalid_matchups = [key for key in matchup_keys if key not in MATCHUP_KEYS]
    if invalid_matchups:
        parser.error(
            "Unknown --matchups value(s): "
            f"{', '.join(invalid_matchups)}. Available: {', '.join(MATCHUP_KEYS)}."
        )
    args.selected_matchups = tuple(MATCHUP_KEYS[key] for key in matchup_keys)

    if not 1 <= args.opening_start <= len(OPENING_POSITIONS):
        parser.error(f"--opening-start must be between 1 and {len(OPENING_POSITIONS)}.")
    if not 1 <= args.opening_end <= len(OPENING_POSITIONS):
        parser.error(f"--opening-end must be between 1 and {len(OPENING_POSITIONS)}.")
    if args.opening_start > args.opening_end:
        parser.error("--opening-start must be less than or equal to --opening-end.")
    if args.game_start < 1:
        parser.error("--game-start must be at least 1.")
    args.selected_openings = OPENING_POSITIONS[args.opening_start - 1 : args.opening_end]
    return args


if __name__ == "__main__":
    run_experiments(parse_args())
