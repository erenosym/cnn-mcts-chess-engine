from __future__ import annotations

import argparse

import chess

from bots.cnn_mcts_bot import analyze_cnn_mcts_root, choose_cnn_mcts_move
from cnn.encoding import move_to_action_index
from cnn.inference import PolicyValueEvaluator


def inspect_position(
    fen: str,
    top_k: int,
    simulations: int,
    c_puct: float,
) -> None:
    board = chess.Board(fen)
    evaluator = PolicyValueEvaluator()

    logits, raw_value = evaluator.evaluate(board)
    priors = evaluator.policy_priors(board)
    root = analyze_cnn_mcts_root(
        board,
        evaluator=evaluator,
        n_simulations=simulations,
        c_puct=c_puct,
    )
    ranked_children = sorted(root.children, key=lambda child: child.visits, reverse=True)
    chosen_move = ranked_children[0].move if ranked_children else choose_cnn_mcts_move(
        board,
        evaluator=evaluator,
        n_simulations=simulations,
        c_puct=c_puct,
    )

    print("Board")
    print(board)
    print()
    print(f"FEN: {board.fen()}")
    print(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Raw value head output: {raw_value:.4f}")
    print(
        "Turn-adjusted value used by search: "
        f"{evaluator.value(board):.4f}"
    )
    print(f"CNN+MCTS chosen move: {chosen_move}")
    print()

    ranked_moves = sorted(priors.items(), key=lambda item: item[1], reverse=True)
    print(f"Top {min(top_k, len(ranked_moves))} policy moves")
    for index, (move, prob) in enumerate(ranked_moves[:top_k], start=1):
        action_index = move_to_action_index(move)
        logit_value = logits[action_index].item()
        print(
            f"{index}. move={move.uci()} "
            f"prior={prob:.4f} "
            f"logit={logit_value:.4f}"
        )

    print()
    print(f"Top {min(top_k, len(ranked_children))} root children by visits")
    for index, child in enumerate(ranked_children[:top_k], start=1):
        average_q = 0.0 if child.visits == 0 else -(child.reward / child.visits)
        print(
            f"{index}. move={child.move.uci()} "
            f"visits={child.visits} "
            f"avg_q={average_q:.4f} "
            f"prior={child.prior:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CNN+MCTS outputs for a FEN position.")
    parser.add_argument("fen", type=str, help="FEN string to inspect.")
    parser.add_argument("--top-k", type=int, default=10, help="How many policy moves to print.")
    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
        help="Number of CNN+MCTS simulations for the chosen move.",
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
    inspect_position(
        fen=args.fen,
        top_k=args.top_k,
        simulations=args.simulations,
        c_puct=args.c_puct,
    )


if __name__ == "__main__":
    main()
