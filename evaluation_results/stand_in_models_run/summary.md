# Retrieval evaluation: taskqueue-sample-v1

> **How to read this.**
> The dataset is an **illustrative fixture** (16 queries over 37 chunks). Differences between pipelines are NOT statistically meaningful.
> This run used **lexical stand-in models** (embedder: `hashing-bow-v1`, reranker: `token-overlap-v1`). The numbers measure the pipeline plumbing with deterministic stubs and are **NOT the performance of the real Sentence-Transformer / Cross-Encoder models**. Re-run with the real models (see README) to measure those.

Generated 2026-10-10T03:44:26+00:00 | candidate_k=20, rrf_k=60 | {'python': '3.12.3', 'platform': 'Linux-6.18.44-fc-v114-x86_64-with-glibc2.39', 'faiss-cpu': '1.15.1', 'rank-bm25': '0.2.2', 'sentence-transformers': 'not installed', 'numpy': '2.4.4', 'torch': 'not installed'}

## Overall

| Pipeline | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Latency ms |
|---|---|---|---|---|---|---|
| A. BM25 only | 0.771 | 0.917 | 0.938 | 0.969 | 0.969 | 2.0 |
| B. FAISS only | 0.708 | 0.854 | 0.969 | 0.969 | 0.891 | 1.7 |
| C. Hybrid (FAISS + BM25 + RRF) | 0.802 | 0.917 | 0.969 | 0.969 | 0.969 | 2.2 |
| D. Hybrid + Cross-Encoder rerank | 0.833 | 0.917 | 0.969 | 0.969 | 1.000 | 3.7 |

## MRR by query type

| Query type | n | A | B | C | D |
|---|---|---|---|---|---|
| class_related | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_function_name | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_parameter_name | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| mixed_code_documentation | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| pdf_reference | 3 | 0.833 | 1.000 | 1.000 | 1.000 |
| readme_documentation | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| semantic_paraphrase | 3 | 1.000 | 0.417 | 0.833 | 1.000 |

## Recall@5 by query type

| Query type | n | A | B | C | D |
|---|---|---|---|---|---|
| class_related | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_function_name | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_parameter_name | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| mixed_code_documentation | 2 | 0.750 | 0.750 | 0.750 | 0.750 |
| pdf_reference | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| readme_documentation | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| semantic_paraphrase | 3 | 0.833 | 1.000 | 1.000 | 1.000 |

## Rank of the first relevant chunk per query (blank = not retrieved)

| Query | Type | A | B | C | D |
|---|---|---|---|---|---|
| `fn-1` compute_backoff | exact_function_name | 1 | 1 | 1 | 1 |
| `fn-2` fetch_pending | exact_function_name | 1 | 1 | 1 | 1 |
| `param-1` delay_seconds | exact_parameter_name | 1 | 1 | 1 | 1 |
| `param-2` visibility_timeout | exact_parameter_name | 1 | 1 | 1 | 1 |
| `sem-1` how do I make jobs that failed run again | semantic_paraphrase | 1 | 2 | 2 | 1 |
| `sem-2` stop the background threads without losing work that is in progress | semantic_paraphrase | 1 | 4 | 1 | 1 |
| `sem-3` reclaim disk space taken by finished jobs | semantic_paraphrase | 1 | 2 | 1 | 1 |
| `class-1` class that stores tasks durably in a SQLite database | class_related | 1 | 1 | 1 | 1 |
| `class-2` WorkerPool class | class_related | 1 | 1 | 1 | 1 |
| `doc-1` how do I install the library | readme_documentation | 1 | 1 | 1 | 1 |
| `doc-2` which environment variables override the configuration file | readme_documentation | 1 | 1 | 1 | 1 |
| `pdf-1` at-least-once delivery guarantee | pdf_reference | 1 | 1 | 1 | 1 |
| `pdf-2` exponential backoff with full jitter and the 300 second cap | pdf_reference | 2 | 1 | 1 | 1 |
| `pdf-3` how long are completed tasks retained before deletion | pdf_reference | 1 | 1 | 1 | 1 |
| `mixed-1` retry delay between attempts | mixed_code_documentation | 1 | 1 | 1 | 1 |
| `mixed-2` what happens when a worker crashes before acknowledging a task | mixed_code_documentation | 1 | 1 | 1 | 1 |

Recall@K counts a relevant item as found if any of its chunks is in the top K; MRR is the mean reciprocal rank of the first relevant chunk (0 if absent). Metrics use the ranking before the relevance threshold.
