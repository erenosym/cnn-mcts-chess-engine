from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bots.cnn_mcts_bot import play_cnn_mcts_game
from bots.mcts_bot import play_mcts_game
from bots.random_bot import play_random_game


def main() -> None:
    print("Choose a mode:")
    print("1. Random vs Random")
    print("2. MCTS vs MCTS")
    print("3. CNN+MCTS vs CNN+MCTS")

    choice = input("Enter your choice (1, 2 or 3): ").strip()

    if choice == "1":
        play_random_game()
    elif choice == "2":
        play_mcts_game()
    elif choice == "3":
        play_cnn_mcts_game()
    else:
        print("Invalid choice. Please run the program again and choose 1, 2 or 3.")


if __name__ == "__main__":
    main()
