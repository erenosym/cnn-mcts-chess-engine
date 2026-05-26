# Experiment results

- `final/`: complete aggregate CSV/JSON summaries plus 983 recovered per-game
  records. The first 37 standard-start log entries were unavailable; their
  counts remain represented in the aggregate summaries used in the thesis.
- `figures/`: generated comparison tables, score plots and search diagrams.

Regenerate the experiment outputs with `python -m matches.experiment_suite` and
the figures with `python tools/experiments/visualize_experiment_results.py`.
