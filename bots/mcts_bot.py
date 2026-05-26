import chess
from core.node import Node


def mcts(root: Node, n_simulations: int = 100) -> Node:
    for _ in range(n_simulations):
        node = root.select()
        child = node.expand()

        if child is None:
            child = node

        reward = child.simulate()
        child.backpropagate(reward)

    return max(root.children, key=lambda x: x.visits)


def choose_mcts_move(
    board: chess.Board, n_simulations: int = 200
) -> chess.Move | None:
    if board.is_game_over():
        return None

    root = Node(board.copy())
    best_node = mcts(root, n_simulations=n_simulations)
    return best_node.move


def play_mcts_game(n_simulations: int = 200) -> chess.Board:
    board = chess.Board()

    while not board.is_game_over():
        move = choose_mcts_move(board, n_simulations=n_simulations)
        if move is None:
            break

        board.push(move)
        print(board)
        print("-------------")

    print(board.result())
    return board
