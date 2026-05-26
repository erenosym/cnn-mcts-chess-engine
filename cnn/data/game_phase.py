from __future__ import annotations

from collections import Counter

import chess

OPENING = "opening"
MIDDLEGAME = "middlegame"
ENDGAME = "endgame"
PHASE_ORDER = (OPENING, MIDDLEGAME, ENDGAME)

PIECE_PHASE_WEIGHTS = {
    chess.QUEEN: 4,
    chess.ROOK: 2,
    chess.BISHOP: 1,
    chess.KNIGHT: 1,
}

PIECE_MATERIAL_VALUES = {
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def count_non_pawn_material(board: chess.Board) -> int:
    total = 0
    for piece_type, value in PIECE_MATERIAL_VALUES.items():
        total += len(board.pieces(piece_type, chess.WHITE)) * value
        total += len(board.pieces(piece_type, chess.BLACK)) * value
    return total


def count_phase_score(board: chess.Board) -> int:
    total = 0
    for piece_type, weight in PIECE_PHASE_WEIGHTS.items():
        total += len(board.pieces(piece_type, chess.WHITE)) * weight
        total += len(board.pieces(piece_type, chess.BLACK)) * weight
    return total


def classify_phase(board: chess.Board) -> str:
    non_pawn_material = count_non_pawn_material(board)
    phase_score = count_phase_score(board)
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))

    if board.fullmove_number <= 10 and non_pawn_material >= 50:
        return OPENING

    if (
        non_pawn_material <= 20
        or (queens == 0 and non_pawn_material <= 28)
        or (rooks <= 1 and non_pawn_material <= 24)
        or phase_score <= 6
    ):
        return ENDGAME

    return MIDDLEGAME


def empty_phase_counter() -> Counter[str]:
    return Counter({phase: 0 for phase in PHASE_ORDER})
