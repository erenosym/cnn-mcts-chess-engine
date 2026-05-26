from __future__ import annotations

import chess
import torch


PROMOTION_PIECES: tuple[int, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
)
BASE_MOVE_SPACE_SIZE = 64 * 64
ACTION_SPACE_SIZE = BASE_MOVE_SPACE_SIZE * (1 + len(PROMOTION_PIECES))


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((13, 8, 8), dtype=torch.float32)

    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    for square, piece in board.piece_map().items():
        channel = piece_to_channel[(piece.piece_type, piece.color)]
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        tensor[channel, row, col] = 1.0

    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return tensor


def move_to_action_index(move: chess.Move) -> int:
    base_index = move.from_square * 64 + move.to_square

    if move.promotion is None:
        return base_index

    try:
        promotion_offset = PROMOTION_PIECES.index(move.promotion) + 1
    except ValueError as exc:
        raise ValueError(f"Unsupported promotion piece in move: {move.uci()}") from exc

    return promotion_offset * BASE_MOVE_SPACE_SIZE + base_index
