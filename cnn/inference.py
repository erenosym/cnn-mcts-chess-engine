from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import torch

from cnn.encoding import board_to_tensor, move_to_action_index
from cnn.policy_value_model import PolicyValueNetwork

BEST_MODEL_PATH = PROJECT_ROOT / "models" / "chess_policy_value_model_best.pt"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PolicyValueEvaluator:
    def __init__(
        self,
        checkpoint_path: Path = BEST_MODEL_PATH,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or select_device()
        self.model = PolicyValueNetwork().to(self.device)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Policy-value checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[torch.Tensor, float]:
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        return policy_logits.squeeze(0).cpu(), float(value.item())

    @torch.no_grad()
    def policy_priors(
        self,
        board: chess.Board,
        temperature: float = 1.0,
    ) -> dict[chess.Move, float]:
        logits, _ = self.evaluate(board)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        move_indices = [move_to_action_index(move) for move in legal_moves]
        legal_logits = logits[move_indices]
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if temperature != 1.0:
            legal_logits = legal_logits / temperature
        legal_probs = torch.softmax(legal_logits, dim=0)

        return {
            move: float(prob)
            for move, prob in zip(legal_moves, legal_probs.tolist())
        }

    @torch.no_grad()
    def value(self, board: chess.Board) -> float:
        _, value = self.evaluate(board)
        return value if board.turn == chess.WHITE else -value
