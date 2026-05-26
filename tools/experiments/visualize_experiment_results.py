from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import chess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matches.experiment_suite import OPENING_POSITIONS

DEFAULT_INPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "final_combined"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "visualizations"

BOT_ORDER = ("CNN+MCTS", "Baseline MCTS", "Random")
TABLE_MATCHUPS = (
    ("CNN+MCTS", "Baseline MCTS"),
    ("CNN+MCTS", "Random"),
    ("Baseline MCTS", "Random"),
)
LABEL_ALIASES = {
    "Vanilla MCTS": "Baseline MCTS",
    "CNN + MCTS": "CNN+MCTS",
}


@dataclass
class SummaryRow:
    experiment: str
    start_position: str
    first_bot: str
    second_bot: str
    games: int
    first_bot_wins: int
    second_bot_wins: int
    draws: int
    average_plies: float


@dataclass
class GameRow:
    experiment: str
    start_position: str
    matchup: str
    game_index: int
    white: str
    black: str
    winner: str
    result: str
    plies: int


@dataclass
class HeadToHead:
    player: str
    opponent: str
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    plies_total: float = 0.0

    @property
    def score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def average_plies(self) -> float:
        if self.games == 0:
            return 0.0
        return self.plies_total / self.games


def clean_label(label: str) -> str:
    return LABEL_ALIASES.get(label.strip(), label.strip())


def read_summary(path: Path, experiment_filter: str) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw in reader:
            if experiment_filter != "all" and raw["experiment"] != experiment_filter:
                continue
            first_bot, second_bot = raw["matchup"].split(" vs ", 1)
            rows.append(
                SummaryRow(
                    experiment=raw["experiment"],
                    start_position=raw["start_position"],
                    first_bot=clean_label(first_bot),
                    second_bot=clean_label(second_bot),
                    games=int(raw["games"]),
                    first_bot_wins=int(raw["first_bot_wins"]),
                    second_bot_wins=int(raw["second_bot_wins"]),
                    draws=int(raw["draws"]),
                    average_plies=float(raw["average_plies"]),
                )
            )
    if not rows:
        raise SystemExit(f"No summary rows found in {path} for experiment={experiment_filter}")
    return rows


def read_game_results(path: Path) -> list[GameRow]:
    rows: list[GameRow] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for raw in reader:
            rows.append(
                GameRow(
                    experiment=raw["experiment"],
                    start_position=raw["start_position"],
                    matchup=" vs ".join(clean_label(part) for part in raw["matchup"].split(" vs ", 1)),
                    game_index=int(raw["game_index"]),
                    white=clean_label(raw["white"]),
                    black=clean_label(raw["black"]),
                    winner=clean_label(raw["winner"]),
                    result=raw["result"],
                    plies=int(raw["plies"]),
                )
            )
    return rows


def aggregate_head_to_head(rows: list[SummaryRow]) -> dict[tuple[str, str], HeadToHead]:
    matchups: dict[tuple[str, str], HeadToHead] = {}

    def get(player: str, opponent: str) -> HeadToHead:
        key = (player, opponent)
        if key not in matchups:
            matchups[key] = HeadToHead(player=player, opponent=opponent)
        return matchups[key]

    for row in rows:
        first = get(row.first_bot, row.second_bot)
        first.games += row.games
        first.wins += row.first_bot_wins
        first.draws += row.draws
        first.losses += row.second_bot_wins
        first.plies_total += row.average_plies * row.games

        second = get(row.second_bot, row.first_bot)
        second.games += row.games
        second.wins += row.second_bot_wins
        second.draws += row.draws
        second.losses += row.first_bot_wins
        second.plies_total += row.average_plies * row.games

    return matchups


def write_table_csv(path: Path, rows: list[HeadToHead]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["player", "opponent", "games", "wins", "draws", "losses", "score_rate", "average_plies"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.player,
                    row.opponent,
                    row.games,
                    row.wins,
                    row.draws,
                    row.losses,
                    f"{row.score_rate:.3f}",
                    f"{row.average_plies:.1f}",
                ]
            )


def plot_alpha_zero_style_table(rows: list[HeadToHead], output_path: Path, title: str) -> None:
    ordered = sorted(rows, key=lambda row: (BOT_ORDER.index(row.player), BOT_ORDER.index(row.opponent)))
    table_rows = [
        [
            row.player,
            row.opponent,
            f"{row.games}",
            f"{row.wins}",
            f"{row.draws}",
            f"{row.losses}",
            f"{100 * row.score_rate:.1f}%",
            f"{row.average_plies:.1f}",
        ]
        for row in ordered
    ]
    headers = ["Player", "Opponent", "Games", "Wins", "Draws", "Losses", "Score", "Avg. plies"]

    fig_height = 1.8 + 0.48 * len(table_rows)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", loc="left", pad=18)

    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.19, 0.19, 0.09, 0.09, 0.09, 0.09, 0.11, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.45)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dde6")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f6f8fb")
        else:
            cell.set_facecolor("white")
        if col_index in (0, 1) and row_index > 0:
            cell.get_text().set_ha("left")

    fig.text(
        0.01,
        0.02,
        "Score = wins + 0.5 x draws. Baseline MCTS is the rollout-based MCTS baseline.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_matrix_csv(path: Path, matchups: dict[tuple[str, str], HeadToHead]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["player"] + list(BOT_ORDER))
        for player in BOT_ORDER:
            values = []
            for opponent in BOT_ORDER:
                if player == opponent:
                    values.append("0.500")
                else:
                    values.append(f"{matchups[(player, opponent)].score_rate:.3f}")
            writer.writerow([player] + values)


def plot_confusion_matrix(
    matchups: dict[tuple[str, str], HeadToHead], output_path: Path, title: str
) -> None:
    matrix = []
    annotations = []
    for player in BOT_ORDER:
        row_values = []
        row_annotations = []
        for opponent in BOT_ORDER:
            if player == opponent:
                row_values.append(0.5)
                row_annotations.append("50.0%\n-")
                continue
            result = matchups[(player, opponent)]
            row_values.append(result.score_rate)
            row_annotations.append(
                f"{100 * result.score_rate:.1f}%\n{result.wins}-{result.draws}-{result.losses}"
            )
        matrix.append(row_values)
        annotations.append(row_annotations)

    cmap = LinearSegmentedColormap.from_list("score_map", ["#c2410c", "#f8fafc", "#0f766e"])
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(BOT_ORDER)), labels=BOT_ORDER)
    ax.set_yticks(range(len(BOT_ORDER)), labels=BOT_ORDER)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Player")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=16)

    for row_index in range(len(BOT_ORDER)):
        for col_index in range(len(BOT_ORDER)):
            value = matrix[row_index][col_index]
            color = "white" if value < 0.22 or value > 0.78 else "#111827"
            ax.text(
                col_index,
                row_index,
                annotations[row_index][col_index],
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks([x - 0.5 for x in range(1, len(BOT_ORDER))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(BOT_ORDER))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Score rate")
    fig.text(
        0.02,
        0.02,
        "Each cell shows score rate and W-D-L from the row player's perspective.",
        ha="left",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_opening_scores_csv(path: Path, rows: list[SummaryRow]) -> None:
    opening_rows = [row for row in rows if row.experiment == "opening_positions"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "opening",
                "matchup",
                "games",
                "first_bot_wins",
                "draws",
                "second_bot_wins",
                "first_bot_score_rate",
                "average_plies",
            ]
        )
        for row in opening_rows:
            score_rate = (row.first_bot_wins + 0.5 * row.draws) / row.games
            writer.writerow(
                [
                    row.start_position,
                    f"{row.first_bot} vs {row.second_bot}",
                    row.games,
                    row.first_bot_wins,
                    row.draws,
                    row.second_bot_wins,
                    f"{score_rate:.3f}",
                    f"{row.average_plies:.1f}",
                ]
            )


def plot_opening_score_heatmap(rows: list[SummaryRow], output_path: Path) -> None:
    opening_rows = [row for row in rows if row.experiment == "opening_positions"]
    if not opening_rows:
        return

    openings = list(dict.fromkeys(row.start_position for row in opening_rows))
    matchups = list(dict.fromkeys(f"{row.first_bot} vs {row.second_bot}" for row in opening_rows))
    values_by_key = {
        (row.start_position, f"{row.first_bot} vs {row.second_bot}"): (
            row.first_bot_wins + 0.5 * row.draws
        )
        / row.games
        for row in opening_rows
    }
    matrix = [
        [values_by_key[(opening, matchup)] for matchup in matchups]
        for opening in openings
    ]

    cmap = LinearSegmentedColormap.from_list("opening_score", ["#c2410c", "#f8fafc", "#0f766e"])
    fig, ax = plt.subplots(figsize=(11.5, 8.8))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(matchups)), labels=matchups, rotation=20, ha="right")
    ax.set_yticks(range(len(openings)), labels=openings)
    ax.set_title("Opening Position Match Scores", fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Matchup")
    ax.set_ylabel("Opening")

    for row_index, opening in enumerate(openings):
        for col_index, matchup in enumerate(matchups):
            value = values_by_key[(opening, matchup)]
            color = "white" if value < 0.22 or value > 0.78 else "#111827"
            ax.text(
                col_index,
                row_index,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks([x - 0.5 for x in range(1, len(matchups))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(openings))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("First bot score rate")
    fig.text(
        0.02,
        0.02,
        "Each opening has 20 games per matchup. Score = wins + 0.5 x draws for the first bot listed.",
        ha="left",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_opening_average_plies(rows: list[SummaryRow], output_path: Path) -> None:
    opening_rows = [row for row in rows if row.experiment == "opening_positions"]
    if not opening_rows:
        return

    openings = list(dict.fromkeys(row.start_position for row in opening_rows))
    matchups = list(dict.fromkeys(f"{row.first_bot} vs {row.second_bot}" for row in opening_rows))
    plies_by_key = {
        (row.start_position, f"{row.first_bot} vs {row.second_bot}"): row.average_plies
        for row in opening_rows
    }

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    x_positions = list(range(len(openings)))
    bar_width = 0.24
    colors = ("#0f766e", "#2563eb", "#9333ea")

    for matchup_index, matchup in enumerate(matchups):
        offset = (matchup_index - (len(matchups) - 1) / 2) * bar_width
        values = [plies_by_key[(opening, matchup)] for opening in openings]
        ax.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width,
            label=matchup,
            color=colors[matchup_index % len(colors)],
        )

    ax.set_title("Average Game Length by Opening", fontsize=18, fontweight="bold", pad=16)
    ax.set_ylabel("Average plies")
    ax.set_xticks(x_positions, labels=openings, rotation=35, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


PIECE_SYMBOLS = {
    chess.PAWN: ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK: ("♖", "♜"),
    chess.QUEEN: ("♕", "♛"),
    chess.KING: ("♔", "♚"),
}


def draw_board(ax: plt.Axes, board: chess.Board) -> None:
    light = "#f6f2e8"
    dark = "#9a9a9a"
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    for rank in range(8):
        for file_index in range(8):
            color = light if (rank + file_index) % 2 == 0 else dark
            ax.add_patch(Rectangle((file_index, rank), 1, 1, facecolor=color, edgecolor="#dddddd", lw=0.2))

    for square, piece in board.piece_map().items():
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        white_symbol, black_symbol = PIECE_SYMBOLS[piece.piece_type]
        ax.text(
            file_index + 0.5,
            rank + 0.5,
            white_symbol if piece.color == chess.WHITE else black_symbol,
            ha="center",
            va="center",
            fontsize=13,
            color="#111111",
            fontfamily="DejaVu Sans",
        )

    for file_index, label in enumerate("abcdefgh"):
        ax.text(file_index + 0.5, -0.25, label, ha="center", va="center", fontsize=6)
    for rank in range(8):
        ax.text(-0.22, rank + 0.5, str(rank + 1), ha="center", va="center", fontsize=6)


def matchup_abbreviation(matchup: str) -> str:
    return {
        "CNN+MCTS vs Random": "C-R",
        "CNN+MCTS vs Baseline MCTS": "C-B",
        "Baseline MCTS vs Random": "B-R",
    }.get(matchup, matchup)


def format_opening_scores(opening_rows: list[SummaryRow]) -> str:
    parts = []
    order = ("CNN+MCTS vs Random", "CNN+MCTS vs Baseline MCTS", "Baseline MCTS vs Random")
    rows_by_matchup = {f"{row.first_bot} vs {row.second_bot}": row for row in opening_rows}
    for matchup in order:
        row = rows_by_matchup.get(matchup)
        if row is None:
            continue
        parts.append(
            f"{matchup_abbreviation(matchup)} {row.first_bot_wins}/{row.draws}/{row.second_bot_wins}"
        )
    return "   ".join(parts)


def moving_average(values: list[int], window: int = 5) -> list[float]:
    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def plot_opening_summary_sheet(
    summary_rows: list[SummaryRow],
    game_rows: list[GameRow],
    output_path: Path,
) -> None:
    opening_summaries = [row for row in summary_rows if row.experiment == "opening_positions"]
    opening_games = [row for row in game_rows if row.experiment == "opening_positions"]
    if not opening_summaries or not opening_games:
        return

    summaries_by_opening: dict[str, list[SummaryRow]] = {}
    games_by_opening: dict[str, list[GameRow]] = {}
    for row in opening_summaries:
        summaries_by_opening.setdefault(row.start_position, []).append(row)
    for row in opening_games:
        games_by_opening.setdefault(row.start_position, []).append(row)

    openings_by_label = {opening.label: opening for opening in OPENING_POSITIONS}
    labels = [opening.label for opening in OPENING_POSITIONS if opening.label in summaries_by_opening]

    fig = plt.figure(figsize=(13.0, 16.8), facecolor="white")
    left_margin = 0.055
    column_gap = 0.045
    top = 0.955
    bottom = 0.075
    row_gap = 0.018
    column_width = (1 - 2 * left_margin - column_gap) / 2
    row_height = (top - bottom - 5 * row_gap) / 6

    fig.lines.append(plt.Line2D([0.04, 0.96], [0.975, 0.975], transform=fig.transFigure, color="black", lw=1.5))
    fig.lines.append(plt.Line2D([0.04, 0.96], [0.035, 0.035], transform=fig.transFigure, color="black", lw=1.5))

    max_plies = max(row.plies for row in opening_games)
    y_max = max(100, int((max_plies + 99) / 100) * 100)
    matchup_sort = {"CNN+MCTS vs Random": 0, "Baseline MCTS vs Random": 1, "CNN+MCTS vs Baseline MCTS": 2}

    for index, label in enumerate(labels):
        column = index % 2
        row_index = index // 2
        x0 = left_margin + column * (column_width + column_gap)
        y0 = top - (row_index + 1) * row_height - row_index * row_gap

        if column == 0:
            line_y = y0 - row_gap * 0.45
            fig.lines.append(
                plt.Line2D([0.04, 0.96], [line_y, line_y], transform=fig.transFigure, color="black", lw=0.8)
            )

        opening = openings_by_label[label]
        title = label
        if label == "C60: Ruy Lopez":
            title = "C60: Ruy Lopez (Spanish Opening)"
        fig.text(x0 + column_width / 2, y0 + row_height - 0.008, title, ha="center", va="top", fontsize=12)

        board_ax = fig.add_axes([x0 + 0.012, y0 + 0.044, column_width * 0.27, row_height * 0.50])
        draw_board(board_ax, opening.board())

        chart_ax = fig.add_axes([x0 + column_width * 0.38, y0 + 0.058, column_width * 0.55, row_height * 0.36])
        games = sorted(
            games_by_opening[label],
            key=lambda item: (matchup_sort.get(item.matchup, 99), item.game_index),
        )
        plies = [game.plies for game in games]
        chart_ax.plot(range(1, len(plies) + 1), moving_average(plies), color="#6b94cf", lw=1.2)
        chart_ax.set_xlim(1, max(1, len(plies)))
        chart_ax.set_ylim(0, y_max)
        chart_ax.set_xticks([1, 20, 40, 60])
        chart_ax.set_xticklabels(["1", "20", "40", "60"], fontsize=7)
        chart_ax.set_yticks([0, y_max // 2, y_max])
        chart_ax.set_yticklabels(["0", str(y_max // 2), str(y_max)], fontsize=7)
        chart_ax.tick_params(length=2, colors="#555555")
        chart_ax.spines["top"].set_visible(False)
        chart_ax.spines["right"].set_visible(False)
        chart_ax.spines["left"].set_color("#777777")
        chart_ax.spines["bottom"].set_color("#777777")
        chart_ax.grid(axis="y", color="#e5e7eb", lw=0.5)

        score_text = format_opening_scores(summaries_by_opening[label])
        fig.text(x0 + 0.012, y0 + 0.028, score_text, ha="left", va="bottom", fontsize=9.5)
        fig.text(
            x0 + 0.012,
            y0 + 0.009,
            " ".join(opening.moves),
            ha="left",
            va="bottom",
            fontsize=8.3,
        )

    total_games = sum(row.games for row in opening_summaries)
    total_wins = sum(row.first_bot_wins for row in opening_summaries)
    total_draws = sum(row.draws for row in opening_summaries)
    total_losses = sum(row.second_bot_wins for row in opening_summaries)
    total_score = (total_wins + 0.5 * total_draws) / total_games
    fig.text(
        0.5,
        0.051,
        (
            f"Opening games: {total_games}     "
            f"First-listed bots total W/D/L: {total_wins}/{total_draws}/{total_losses}     "
            f"Score: {100 * total_score:.1f}%"
        ),
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize chess bot experiment results.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_INPUT_DIR / "summary.csv",
        help="Path to summary.csv generated by the experiment suite.",
    )
    parser.add_argument(
        "--games",
        type=Path,
        default=DEFAULT_INPUT_DIR / "game_results.csv",
        help="Path to game_results.csv generated by the experiment suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated tables and matrices.",
    )
    parser.add_argument(
        "--experiment",
        choices=("all", "standard_start", "opening_positions"),
        default="all",
        help="Subset of results to visualize. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary(args.summary, args.experiment)
    matchups = aggregate_head_to_head(rows)
    table_rows = [matchups[pair] for pair in TABLE_MATCHUPS if pair in matchups]

    suffix = args.experiment
    title_suffix = "All experiments" if suffix == "all" else suffix.replace("_", " ").title()
    table_png = args.output_dir / f"alpha_zero_style_table_{suffix}.png"
    table_csv = args.output_dir / f"alpha_zero_style_table_{suffix}.csv"
    matrix_png = args.output_dir / f"confusion_matrix_{suffix}.png"
    matrix_csv = args.output_dir / f"confusion_matrix_{suffix}.csv"

    write_table_csv(table_csv, table_rows)
    plot_alpha_zero_style_table(
        table_rows,
        table_png,
        f"Chess Bot Match Results - {title_suffix}",
    )
    write_matrix_csv(matrix_csv, matchups)
    plot_confusion_matrix(
        matchups,
        matrix_png,
        f"Head-to-Head Score Matrix - {title_suffix}",
    )

    opening_scores_png = args.output_dir / "opening_matchup_scores.png"
    opening_plies_png = args.output_dir / "opening_average_plies.png"
    opening_sheet_png = args.output_dir / "opening_summary_sheet.png"
    opening_scores_csv = args.output_dir / "opening_matchup_scores.csv"
    if args.experiment in ("all", "opening_positions"):
        game_rows = read_game_results(args.games)
        write_opening_scores_csv(opening_scores_csv, rows)
        plot_opening_score_heatmap(rows, opening_scores_png)
        plot_opening_average_plies(rows, opening_plies_png)
        plot_opening_summary_sheet(rows, game_rows, opening_sheet_png)

    print(f"Wrote table image: {table_png}")
    print(f"Wrote table CSV: {table_csv}")
    print(f"Wrote confusion matrix image: {matrix_png}")
    print(f"Wrote confusion matrix CSV: {matrix_csv}")
    if args.experiment in ("all", "opening_positions"):
        print(f"Wrote opening score heatmap: {opening_scores_png}")
        print(f"Wrote opening plies chart: {opening_plies_png}")
        print(f"Wrote opening summary sheet: {opening_sheet_png}")
        print(f"Wrote opening score CSV: {opening_scores_csv}")


if __name__ == "__main__":
    main()
