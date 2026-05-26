import math
import random
import chess


class Node:
    def __init__(
        self,
        board: chess.Board,
        parent=None,
        move: chess.Move | None = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.color = board.turn
        self.prior = prior

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def best_child(self, c_param: float = 1.4):
        best_score = -float("inf")
        best_node = None

        for child in self.children:
            if child.visits == 0:
                return child

            exploit = -(child.reward / child.visits)
            explore = c_param * math.sqrt(
                math.log(self.visits + 1) / child.visits
            )

            score = exploit + explore
            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def select(self):
        node = self

        while not node.is_terminal():
            if len(node.children) < len(list(node.board.legal_moves)):
                return node

            node = node.best_child()

        return node

    def expand(self):
        legal_moves = list(self.board.legal_moves)
        tried_moves = [child.move for child in self.children]
        untried_moves = [move for move in legal_moves if move not in tried_moves]

        if not untried_moves:
            return None

        move = random.choice(untried_moves)
        new_board = self.board.copy()
        new_board.push(move)

        child_node = Node(new_board, parent=self, move=move)
        self.children.append(child_node)

        return child_node

    def simulate(self) -> int:
        board = self.board.copy()

        while not board.is_game_over():
            moves = list(board.legal_moves)
            if not moves:
                break

            board.push(random.choice(moves))

        return {
            "1-0": 1,
            "0-1": -1,
            "1/2-1/2": 0,
        }[board.result()]

    def backpropagate(self, reward: int) -> None:
        self.visits += 1
        self.reward += reward

        if self.parent:
            self.parent.backpropagate(-reward)


class PolicyValueNode(Node):
    def __init__(
        self,
        board: chess.Board,
        evaluator,
        c_puct: float = 1.4,
        policy_temperature: float = 1.0,
        parent=None,
        move: chess.Move | None = None,
        prior: float = 0.0,
    ):
        super().__init__(board=board, parent=parent, move=move, prior=prior)
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.policy_temperature = policy_temperature
        self._policy_priors: dict[chess.Move, float] | None = None

    def _get_policy_priors(self) -> dict[chess.Move, float]:
        if self._policy_priors is None:
            self._policy_priors = self.evaluator.policy_priors(
                self.board,
                temperature=self.policy_temperature,
            )
        return self._policy_priors

    def best_child(self, c_param: float | None = None):
        if c_param is None:
            c_param = self.c_puct
        best_score = -float("inf")
        best_node = None

        for child in self.children:
            exploit = 0.0 if child.visits == 0 else -(child.reward / child.visits)
            explore = c_param * child.prior * math.sqrt(self.visits + 1) / (1 + child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def select(self):
        node = self

        while not node.is_terminal():
            if len(node.children) < len(list(node.board.legal_moves)):
                return node

            next_node = node.best_child()
            if next_node is None:
                return node
            node = next_node

        return node

    def expand(self):
        legal_moves = list(self.board.legal_moves)
        tried_moves = {child.move for child in self.children}
        untried_moves = [move for move in legal_moves if move not in tried_moves]

        if not untried_moves:
            return None

        priors = self._get_policy_priors()
        move = max(untried_moves, key=lambda legal_move: priors.get(legal_move, 0.0))
        new_board = self.board.copy()
        new_board.push(move)

        child_node = PolicyValueNode(
            new_board,
            evaluator=self.evaluator,
            c_puct=self.c_puct,
            policy_temperature=self.policy_temperature,
            parent=self,
            move=move,
            prior=priors.get(move, 0.0),
        )
        self.children.append(child_node)
        return child_node

    def simulate(self) -> float:
        if self.board.is_game_over():
            result = self.board.result()
            terminal_reward = {
                "1-0": 1.0,
                "0-1": -1.0,
                "1/2-1/2": 0.0,
            }[result]
            return terminal_reward if self.board.turn == chess.WHITE else -terminal_reward

        return self.evaluator.value(self.board)
