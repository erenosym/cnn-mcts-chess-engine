from __future__ import annotations

import sys
from pathlib import Path

import chess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.cnn_mcts_bot import choose_cnn_mcts_move
from cnn.inference import PolicyValueEvaluator

CNN_MCTS_SIMULATIONS = 1200
CNN_MCTS_C_PUCT = 1.4
CNN_POLICY_TEMPERATURE = 0.50
CNN_MOVE_TEMPERATURE = 0
CNN_CLEAR_BEST_MARGIN = 0.35
CNN_CLEAR_BEST_SCORE = 0.6
CNN_REPETITION_PENALTY = 0.3


def read_color() -> chess.Color:
    while True:
        choice = input("Choose your color [w/b] (default: w): ").strip().lower()
        if choice in {"", "w", "white"}:
            return chess.WHITE
        if choice in {"b", "black"}:
            return chess.BLACK
        print("Please enter w or b.")


def read_human_move(board: chess.Board) -> chess.Move | None:
    while True:
        raw_move = input("Your move (for example e2e4 or Nf3, q to quit): ").strip()
        if raw_move.lower() in {"q", "quit", "exit"}:
            return None

        try:
            move = board.parse_uci(raw_move.lower())
        except ValueError:
            try:
                move = board.parse_san(raw_move)
            except ValueError:
                print("Invalid or illegal move. Try again.")
                continue

        if move in board.legal_moves:
            return move
        print("Illegal move. Try again.")


def print_board(board: chess.Board, human_color: chess.Color) -> None:
    print()
    print(board.unicode(borders=True, orientation=human_color))
    print()


def main() -> None:
    human_color = read_color()
    board = chess.Board()
    evaluator = PolicyValueEvaluator()

    print("CNN+MCTS is ready. Enter q at any move prompt to quit.")
    print_board(board, human_color)

    while not board.is_game_over():
        if board.turn == human_color:
            move = read_human_move(board)
            if move is None:
                print("Game ended by the player.")
                return
            print(f"You played: {board.san(move)}")
        else:
            print("CNN+MCTS is thinking...")
            move = choose_cnn_mcts_move(
                board,
                evaluator=evaluator,
                n_simulations=CNN_MCTS_SIMULATIONS,
                c_puct=CNN_MCTS_C_PUCT,
                policy_temperature=CNN_POLICY_TEMPERATURE,
                move_temperature=CNN_MOVE_TEMPERATURE,
                clear_best_margin=CNN_CLEAR_BEST_MARGIN,
                clear_best_score=CNN_CLEAR_BEST_SCORE,
                repetition_penalty=CNN_REPETITION_PENALTY,
            )
            if move is None:
                break
            print(f"CNN+MCTS played: {board.san(move)} ({move.uci()})")

        board.push(move)
        print_board(board, human_color)

    print(f"Game over: {board.outcome()} | Result: {board.result()}")


if __name__ == "__main__":
    main()
