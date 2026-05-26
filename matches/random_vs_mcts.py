from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from bots.mcts_bot import choose_mcts_move
from bots.random_bot import choose_random_move


def play_random_vs_mcts_game(
    random_plays_white: bool = True,
    n_simulations: int = 200,
) -> chess.Board:
    board = chess.Board()

    while not board.is_game_over():
        is_random_turn = board.turn == chess.WHITE if random_plays_white else board.turn == chess.BLACK

        if is_random_turn:
            move = choose_random_move(board)
            player_name = "Random"
        else:
            move = choose_mcts_move(board, n_simulations=n_simulations)
            player_name = "MCTS"

        if move is None:
            break

        print(f"{player_name} plays: {move}")
        board.push(move)
        print(board)
        print("-----------------")

    print(f"Game over: {board.result()}")
    return board


if __name__ == "__main__":
    play_random_vs_mcts_game()
