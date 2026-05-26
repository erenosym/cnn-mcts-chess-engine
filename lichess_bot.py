from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass

import chess
import requests

from bots.cnn_mcts_bot import MoveSelectionDiagnostics, choose_cnn_mcts_move_with_diagnostics
from cnn.inference import PolicyValueEvaluator

LICHESS_API_BASE = "https://lichess.org/api"
BOT_API_TOKEN = os.getenv("LICHESS_BOT_TOKEN", "")
CNN_MCTS_SIMULATIONS = 1200
CNN_MCTS_C_PUCT = 1.4
CNN_POLICY_TEMPERATURE = 1.10
CNN_MOVE_TEMPERATURE = 0
CNN_CLEAR_BEST_MARGIN = 0.35
CNN_CLEAR_BEST_SCORE = 0.6
CNN_REPETITION_PENALTY = 0.3
MOVE_OVERHEAD_MS = 500


@dataclass
class BotRuntimeConfig:
    token: str = BOT_API_TOKEN
    simulations: int = CNN_MCTS_SIMULATIONS
    c_puct: float = CNN_MCTS_C_PUCT
    policy_temperature: float = CNN_POLICY_TEMPERATURE
    move_temperature: float = CNN_MOVE_TEMPERATURE
    clear_best_margin: float = CNN_CLEAR_BEST_MARGIN
    clear_best_score: float = CNN_CLEAR_BEST_SCORE
    repetition_penalty: float = CNN_REPETITION_PENALTY
    move_overhead_ms: int = MOVE_OVERHEAD_MS


class LichessBotClient:
    def __init__(self, config: BotRuntimeConfig) -> None:
        if not config.token:
            raise ValueError(
                "Set the LICHESS_BOT_TOKEN environment variable before running the bot."
            )

        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Accept": "application/json",
                "User-Agent": "graduation-chess-bot/1.0",
            }
        )
        self.account_username = self._fetch_account_username()
        self.evaluator = PolicyValueEvaluator()
        self.model_lock = threading.Lock()
        self.active_games: dict[str, threading.Thread] = {}
        self.game_colors: dict[str, chess.Color] = {}

    def _fetch_account_username(self) -> str:
        response = self.session.get(f"{LICHESS_API_BASE}/account", timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload["username"]

    def stream_incoming_events(self) -> None:
        while True:
            try:
                with self.session.get(
                    f"{LICHESS_API_BASE}/stream/event",
                    stream=True,
                    timeout=60,
                ) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue

                        event = json.loads(raw_line)
                        self.handle_event(event)
            except requests.RequestException as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                retry_delay = 60 if status_code == 429 else 3
                print(
                    f"Event stream error: {exc}. "
                    f"Reconnecting in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)

    def handle_event(self, event: dict) -> None:
        event_type = event.get("type")

        if event_type == "challenge":
            self.handle_challenge(event["challenge"])
            return

        if event_type == "gameStart":
            game_id = event["game"]["id"]
            if game_id not in self.active_games:
                game_thread = threading.Thread(
                    target=self.play_game_stream,
                    args=(game_id,),
                    daemon=True,
                )
                self.active_games[game_id] = game_thread
                game_thread.start()
            return

        if event_type == "gameFinish":
            game_id = event["game"]["id"]
            self.active_games.pop(game_id, None)
            print(f"Game finished: {game_id}")

    def handle_challenge(self, challenge: dict) -> None:
        challenge_id = challenge["id"]
        variant_key = challenge.get("variant", {}).get("key", "")
        speed = challenge.get("speed", "")
        challenger = challenge.get("challenger", {}).get("name", "unknown")

        if variant_key != "standard":
            print(f"Ignoring unsupported challenge {challenge_id} from {challenger} (variant={variant_key})")
            return

        print(f"Accepting challenge {challenge_id} from {challenger} (speed={speed})")
        response = self.session.post(
            f"{LICHESS_API_BASE}/challenge/{challenge_id}/accept",
            timeout=30,
        )
        response.raise_for_status()

    def play_game_stream(self, game_id: str) -> None:
        print(f"Starting game stream for {game_id}")
        initial_fen = chess.STARTING_FEN
        last_move_string = None

        try:
            with self.session.get(
                f"{LICHESS_API_BASE}/bot/game/stream/{game_id}",
                stream=True,
                timeout=60,
            ) as response:
                response.raise_for_status()

                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    event = json.loads(raw_line)
                    event_type = event.get("type")

                    if event_type == "gameFull":
                        initial_fen = event.get("initialFen", "startpos")
                        if initial_fen == "startpos":
                            initial_fen = chess.STARTING_FEN
                        self.game_colors[game_id] = self.extract_our_color(event)
                        state = event.get("state", {})
                        last_move_string = state.get("moves", "")
                        self.maybe_play_move(game_id, initial_fen, state)
                        continue

                    if event_type == "gameState":
                        moves = event.get("moves", "")
                        if moves == last_move_string:
                            continue
                        last_move_string = moves
                        self.maybe_play_move(game_id, initial_fen, event)
        except requests.RequestException as exc:
            print(f"Game stream error for {game_id}: {exc}")
        finally:
            self.active_games.pop(game_id, None)
            self.game_colors.pop(game_id, None)
            print(f"Stopped game stream for {game_id}")

    def maybe_play_move(self, game_id: str, initial_fen: str, state: dict) -> None:
        board = self.build_board(initial_fen, state.get("moves", ""))

        if board.is_game_over():
            return

        if not self.is_our_turn(game_id, board):
            return

        move = self.select_move(board)
        if move is None:
            return

        self.send_move(game_id, move)

    def build_board(self, initial_fen: str, moves_string: str) -> chess.Board:
        board = chess.Board() if initial_fen == chess.STARTING_FEN else chess.Board(initial_fen)

        if moves_string:
            for uci in moves_string.split():
                board.push_uci(uci)

        return board

    def extract_our_color(self, event: dict) -> chess.Color:
        white_name = event.get("white", {}).get("name", "")
        black_name = event.get("black", {}).get("name", "")
        username = self.account_username.lower()

        if white_name.lower() == username:
            return chess.WHITE
        if black_name.lower() == username:
            return chess.BLACK
        raise ValueError("Could not determine bot color from gameFull event.")

    def is_our_turn(self, game_id: str, board: chess.Board) -> bool:
        our_color = self.game_colors.get(game_id)
        if our_color is None:
            return False
        return board.turn == our_color

    def select_move(self, board: chess.Board) -> chess.Move | None:
        with self.model_lock:
            diagnostics = choose_cnn_mcts_move_with_diagnostics(
                board,
                evaluator=self.evaluator,
                n_simulations=self.config.simulations,
                c_puct=self.config.c_puct,
                policy_temperature=self.config.policy_temperature,
                move_temperature=self.config.move_temperature,
                clear_best_margin=self.config.clear_best_margin,
                clear_best_score=self.config.clear_best_score,
                repetition_penalty=self.config.repetition_penalty,
            )
        if diagnostics is None:
            return None
        self.log_move_diagnostics(board, diagnostics)
        return diagnostics.selected_move

    def log_move_diagnostics(
        self,
        board: chess.Board,
        diagnostics: MoveSelectionDiagnostics,
    ) -> None:
        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        print(
            f"Move diagnostics | side={side_to_move} | "
            f"fen={board.fen()} | root_v={diagnostics.root_value:.4f} | "
            f"selected={diagnostics.selected_move.uci()}"
        )
        for index, candidate in enumerate(diagnostics.top_candidates, start=1):
            print(
                f"  {index}. move={candidate.move.uci()} "
                f"p={candidate.prior:.4f} "
                f"v={candidate.value:.4f} "
                f"q={candidate.avg_q:.4f} "
                f"visits={candidate.visits} "
                f"repeat={candidate.is_repetition}"
            )

    def send_move(self, game_id: str, move: chess.Move) -> None:
        print(f"Playing move in {game_id}: {move.uci()}")
        response = self.session.post(
            f"{LICHESS_API_BASE}/bot/game/{game_id}/move/{move.uci()}",
            data={"offeringDraw": "false"},
            timeout=30,
        )
        response.raise_for_status()


def main() -> None:
    config = BotRuntimeConfig()
    bot = LichessBotClient(config)
    print(f"Logged in as {bot.account_username}")
    print(
        "Bot started with "
        f"{config.simulations} simulations, c_puct={config.c_puct}, "
        f"policy_temperature={config.policy_temperature}, "
        f"move_temperature={config.move_temperature}, "
        f"repetition_penalty={config.repetition_penalty}."
    )
    bot.stream_incoming_events()


if __name__ == "__main__":
    main()
