# Evaluation results

`stand_in_models_run/` was produced with `--embedder hashing --reranker overlap`: deterministic
**lexical stand-ins**, used because the development sandbox could not download the real models.
It exercises the evaluation pipeline end to end but is **not a measurement of all-MiniLM-L6-v2 or
the Cross-Encoder**, and the 16-query fixture is far too small for statistical conclusions.

To measure the real models (one-time model download), run on your machine:

    python -m trustdoc eval-retrieval --out evaluation_results/real_models_run
