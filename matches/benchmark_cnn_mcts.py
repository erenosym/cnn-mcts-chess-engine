from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.cnn_mcts_bot import choose_cnn_mcts_move
from bots.mcts_bot import choose_mcts_move
from bots.random_bot import choose_random_move
from cnn.inference import PolicyValueEvaluator


def play_single_game(
    evaluator: PolicyValueEvaluator,
    opponent: str,
    cnn_plays_white: bool,
    n_simulations: int,
    c_puct: float,
    opponent_simulations: int,
) -> tuple[str, int, str]:
    board = chess.Board()

    while not board.is_game_over():
        is_cnn_turn = board.turn == chess.WHITE if cnn_plays_white else board.turn == chess.BLACK

        if is_cnn_turn:
            move = choose_cnn_mcts_move(
                board,
                evaluator=evaluator,
                n_simulations=n_simulations,
                c_puct=c_puct,
            )
        else:
            if opponent == "random":
                move = choose_random_move(board)
            else:
                move = choose_mcts_move(board, n_simulations=opponent_simulations)

        if move is None:
            break

        board.push(move)

    result = board.result()
    plies = board.ply()

    if result == "1/2-1/2":
        return "draw", plies, result

    cnn_won = (result == "1-0" and cnn_plays_white) or (result == "0-1" and not cnn_plays_white)
    return ("win" if cnn_won else "loss"), plies, result


def run_benchmark(
    opponent: str,
    n_games: int,
    n_simulations: int,
    c_puct: float,
    opponent_simulations: int,
) -> None:
    evaluator = PolicyValueEvaluator()
    stats = {"win": 0, "draw": 0, "loss": 0}
    total_plies = 0

    for game_index in range(n_games):
        cnn_plays_white = game_index % 2 == 0
        outcome, plies, result = play_single_game(
            evaluator=evaluator,
            opponent=opponent,
            cnn_plays_white=cnn_plays_white,
            n_simulations=n_simulations,
            c_puct=c_puct,
            opponent_simulations=opponent_simulations,
        )
        stats[outcome] += 1
        total_plies += plies

        side = "White" if cnn_plays_white else "Black"
        print(
            f"Game {game_index + 1}/{n_games} - "
            f"CNN+MCTS as {side}: {outcome} - "
            f"result: {result} - plies: {plies}"
        )

    total = max(1, n_games)
    win_rate = stats["win"] / total
    draw_rate = stats["draw"] / total
    loss_rate = stats["loss"] / total
    score_rate = (stats["win"] + 0.5 * stats["draw"]) / total
    avg_plies = total_plies / total

    print()
    print("Benchmark summary")
    print(f"Opponent: {opponent}")
    print(f"Wins: {stats['win']}")
    print(f"Draws: {stats['draw']}")
    print(f"Losses: {stats['loss']}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Draw rate: {draw_rate:.3f}")
    print(f"Loss rate: {loss_rate:.3f}")
    print(f"Score rate: {score_rate:.3f}")
    print(f"Average plies: {avg_plies:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CNN+MCTS against Random or MCTS.")
    parser.add_argument(
        "--opponent",
        choices=["random", "mcts"],
        default="random",
        help="Opponent type.",
    )
    parser.add_argument("--games", type=int, default=20, help="Number of games to play.")
    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
        help="Number of CNN+MCTS simulations per move.",
    )
    parser.add_argument(
        "--opponent-simulations",
        type=int,
        default=200,
        help="Number of simulations per move for the opponent when opponent is MCTS.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.4,
        help="Exploration constant for CNN+MCTS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        opponent=args.opponent,
        n_games=args.games,
        n_simulations=args.simulations,
        c_puct=args.c_puct,
        opponent_simulations=args.opponent_simulations,
    )


if __name__ == "__main__":
    main()
