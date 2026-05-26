# Bot Match Scripts

This folder contains local scripts for bot-vs-bot games and quick benchmarks.

- `play_menu.py`: small menu to launch self-play demos
- `random_vs_mcts.py`: Random vs Baseline MCTS
- `random_vs_cnn_mcts.py`: Random vs CNN+MCTS
- `benchmark_cnn_mcts.py`: repeated benchmark runner for CNN+MCTS
- `experiment_suite.py`: main balanced experiment runner used for the report

Recommended report command:

```bash
python3 matches/experiment_suite.py
```

Defaults:

- Opening positions: 20 games per matchup, 10 games with the first bot as White and 10 as Black.
- Standard initial position: 100 games per matchup, 50 games with the first bot as White and 50 as Black.
- MCTS simulations per move: 800.
