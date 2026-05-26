# Model checkpoint

`chess_policy_value_model_best.pt` is the best validation checkpoint produced by
`python -m cnn.train`. It is loaded by default by `cnn.inference.PolicyValueEvaluator`.

The intermediate checkpoint `chess_policy_value_model.pt` is intentionally ignored.
Re-running training may overwrite the included best checkpoint.
