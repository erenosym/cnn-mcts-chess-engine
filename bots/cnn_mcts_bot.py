from __future__ import annotations

import math
import random
from dataclasses import dataclass

import chess

from cnn.inference import PolicyValueEvaluator
from core.node import PolicyValueNode


DEFAULT_ROOT_POLICY_TEMPERATURE = 1.35
DEFAULT_ROOT_MOVE_TEMPERATURE = 0.8
DEFAULT_CLEAR_BEST_MARGIN = 0.35
DEFAULT_CLEAR_BEST_SCORE = 0.6
DEFAULT_REPETITION_PENALTY = 0.3


@dataclass
class MoveCandidateDiagnostics:
    move: chess.Move
    prior: float
    value: float
    avg_q: float
    visits: int
    is_repetition: bool


@dataclass
class MoveSelectionDiagnostics:
    selected_move: chess.Move
    top_candidates: list[MoveCandidateDiagnostics]
    root_value: float


def _child_score_from_root(child: PolicyValueNode) -> float:
    if child.visits == 0:
        return 0.0
    return -(child.reward / child.visits)


def _would_repeat_position(board: chess.Board, move: chess.Move) -> bool:
    next_board = board.copy(stack=True)
    next_board.push(move)
    return next_board.can_claim_threefold_repetition() or next_board.is_repetition(3)


def _sample_move_from_root(
    root: PolicyValueNode,
    move_temperature: float,
    clear_best_margin: float,
    clear_best_score: float,
    repetition_penalty: float,
) -> PolicyValueNode:
    if not root.children:
        raise ValueError("Cannot choose a move from an empty root.")

    visit_ranked_children = sorted(root.children, key=lambda child: child.visits, reverse=True)
    non_repeating_visit_ranked_children = [
        child
        for child in visit_ranked_children
        if not _would_repeat_position(root.board, child.move)
    ]
    visit_best_child = (
        non_repeating_visit_ranked_children[0]
        if repetition_penalty > 0 and non_repeating_visit_ranked_children
        else visit_ranked_children[0]
    )

    scored_children: list[tuple[float, PolicyValueNode]] = []
    for child in root.children:
        score = _child_score_from_root(child)
        if repetition_penalty > 0 and _would_repeat_position(root.board, child.move):
            score -= repetition_penalty
        scored_children.append((score, child))

    scored_children.sort(key=lambda item: item[0], reverse=True)
    best_score, best_child = scored_children[0]
    runner_up_score = scored_children[1][0] if len(scored_children) > 1 else -float("inf")

    if (
        best_score >= clear_best_score
        and best_score - runner_up_score >= clear_best_margin
    ):
        return best_child

    if move_temperature <= 0:
        return visit_best_child

    normalized_logits = [
        math.log(child.visits + 1) / move_temperature
        for child in visit_ranked_children
    ]

    max_logit = max(normalized_logits)
    weights = [math.exp(logit - max_logit) for logit in normalized_logits]
    return random.choices(
        visit_ranked_children,
        weights=weights,
        k=1,
    )[0]


def _collect_top_candidates(
    root: PolicyValueNode,
    top_k: int,
    selected_child: PolicyValueNode | None = None,
) -> list[MoveCandidateDiagnostics]:
    ranked_children = sorted(root.children, key=lambda child: child.visits, reverse=True)
    selected_is_ranked = selected_child is None or selected_child in ranked_children[:top_k]
    diagnostic_children = ranked_children[:top_k]
    if selected_child is not None and not selected_is_ranked:
        diagnostic_children = [*diagnostic_children, selected_child]

    top_candidates: list[MoveCandidateDiagnostics] = []
    for child in diagnostic_children:
        top_candidates.append(
            MoveCandidateDiagnostics(
                move=child.move,
                prior=child.prior,
                value=-root.evaluator.value(child.board),
                avg_q=_child_score_from_root(child),
                visits=child.visits,
                is_repetition=_would_repeat_position(root.board, child.move),
            )
        )
    return top_candidates


def cnn_mcts(
    root: PolicyValueNode,
    n_simulations: int = 200,
) -> PolicyValueNode:
    for _ in range(n_simulations):
        node = root.select()
        child = node.expand()

        if child is None:
            child = node

        reward = child.simulate()
        child.backpropagate(reward)

    return max(root.children, key=lambda child: child.visits, default=root)


def choose_cnn_mcts_move(
    board: chess.Board,
    evaluator: PolicyValueEvaluator,
    n_simulations: int = 200,
    c_puct: float = 1.4,
    policy_temperature: float = DEFAULT_ROOT_POLICY_TEMPERATURE,
    move_temperature: float = DEFAULT_ROOT_MOVE_TEMPERATURE,
    clear_best_margin: float = DEFAULT_CLEAR_BEST_MARGIN,
    clear_best_score: float = DEFAULT_CLEAR_BEST_SCORE,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
) -> chess.Move | None:
    diagnostics = choose_cnn_mcts_move_with_diagnostics(
        board=board,
        evaluator=evaluator,
        n_simulations=n_simulations,
        c_puct=c_puct,
        policy_temperature=policy_temperature,
        move_temperature=move_temperature,
        clear_best_margin=clear_best_margin,
        clear_best_score=clear_best_score,
        repetition_penalty=repetition_penalty,
    )
    return None if diagnostics is None else diagnostics.selected_move


def choose_cnn_mcts_move_with_diagnostics(
    board: chess.Board,
    evaluator: PolicyValueEvaluator,
    n_simulations: int = 200,
    c_puct: float = 1.4,
    policy_temperature: float = DEFAULT_ROOT_POLICY_TEMPERATURE,
    move_temperature: float = DEFAULT_ROOT_MOVE_TEMPERATURE,
    clear_best_margin: float = DEFAULT_CLEAR_BEST_MARGIN,
    clear_best_score: float = DEFAULT_CLEAR_BEST_SCORE,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    top_k: int = 5,
) -> MoveSelectionDiagnostics | None:
    if board.is_game_over():
        return None

    root = PolicyValueNode(
        board.copy(),
        evaluator=evaluator,
        c_puct=c_puct,
        policy_temperature=policy_temperature,
    )
    cnn_mcts(root, n_simulations=n_simulations)
    selected_child = _sample_move_from_root(
        root,
        move_temperature=move_temperature,
        clear_best_margin=clear_best_margin,
        clear_best_score=clear_best_score,
        repetition_penalty=repetition_penalty,
    )
    return MoveSelectionDiagnostics(
        selected_move=selected_child.move,
        top_candidates=_collect_top_candidates(root, top_k=top_k, selected_child=selected_child),
        root_value=evaluator.value(board),
    )


def analyze_cnn_mcts_root(
    board: chess.Board,
    evaluator: PolicyValueEvaluator,
    n_simulations: int = 200,
    c_puct: float = 1.4,
    policy_temperature: float = DEFAULT_ROOT_POLICY_TEMPERATURE,
) -> PolicyValueNode:
    if board.is_game_over():
        raise ValueError("Cannot analyze a finished game position.")

    root = PolicyValueNode(
        board.copy(),
        evaluator=evaluator,
        c_puct=c_puct,
        policy_temperature=policy_temperature,
    )
    cnn_mcts(root, n_simulations=n_simulations)
    return root


def play_cnn_mcts_game(
    n_simulations: int = 200,
    c_puct: float = 1.4,
    policy_temperature: float = DEFAULT_ROOT_POLICY_TEMPERATURE,
    move_temperature: float = DEFAULT_ROOT_MOVE_TEMPERATURE,
) -> chess.Board:
    board = chess.Board()
    evaluator = PolicyValueEvaluator()

    while not board.is_game_over():
        move = choose_cnn_mcts_move(
            board,
            evaluator=evaluator,
            n_simulations=n_simulations,
            c_puct=c_puct,
            policy_temperature=policy_temperature,
            move_temperature=move_temperature,
        )
        if move is None:
            break

        board.push(move)
        print(board)
        print("-------------")

    print(board.result())
    return board
