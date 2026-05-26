import random
import chess


def choose_random_move(board: chess.Board) -> chess.Move | None:
    moves = list(board.legal_moves)
    if not moves:
        return None
    return random.choice(moves)


def play_random_game() -> chess.Board:
    board = chess.Board()

    while not board.is_game_over():
        move = choose_random_move(board)
        if move is None:
            break

        board.push(move)
        print(board)
        print("-----------------")

    print(board.result())
    return board
