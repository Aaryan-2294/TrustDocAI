# TrustDoc AI: Real-Model Retrieval & ASK Pipeline Evaluation Report

**Date**: 2026-10-27
**Corpus**: `retry-task-queue` (Canonical: `workers.py`, `READme.md`, 8 chunks)
**Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (Local on `E:\AI\Cache\HuggingFace\hub`)
**Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Local on `E:\AI\Cache\HuggingFace\hub`)
**LLM**: Ollama `qwen2.5-coder:1.5b` (Local on `E:\Apps\Ollama`, models on `E:\AI\Models\Ollama`)
**Hardware Mode**: CPU inference

---

## 1. Executive Summary

| Pipeline Component | Metric | Result | Target / Baseline |
| :--- | :--- | :--- | :--- |
| **Retrieval Stage** | Recall@1 | **83.3%** | > 70% |
| | Recall@3 | **100.0%** | > 90% |
| | Recall@5 | **100.0%** | 100% |
| | MRR (Mean Reciprocal Rank) | **0.917** | > 0.80 |
| **Score Separation** | Mean Gold Chunk Score | **+5.04** | > 0.0 |
| | Mean Negative Chunk Score | **-4.00** | < 0.0 |
| **Relevance Filtering** | True Abstentions on Unanswerable (at 0.0) | **50.0%** (4/8) | Filter out noise |
| | True Abstentions on Unanswerable (at -15.0) | **0.0%** (0/8) | Passes all noise to LLM |
| **ASK Workflow (End-to-End)** | Answerable Queries Correctly Answered | **0.0%** (0/12) | **Defect** (100% False Abstention) |
| | Unanswerable Queries Correctly Abstained | **100.0%** (8/8) | Correct |
| | Citation ID Validity | **100.0%** | 100% |
| | Average Response Latency | **10.1s** (CPU) | < 30s |

---

## 2. Labeled Evaluation Dataset (20 Queries)

A reproducible 20-query evaluation suite was created under `trustdoc/evaluation/data/retry_task_queue_eval.json`:
- **Factual Explicit Questions** (5 queries): Direct questions whose answers are explicitly stated in prose or docstrings (`READme.md` or `workers.py`).
- **Signature & Defaults Questions** (4 queries): Exact parameter names, default values, and return type annotations.
- **Behavior & Logic Questions** (3 queries): Branch conditions, mathematical calculations, and random jitter generation.
- **Unanswerable Missing Questions** (4 queries): Queries asking about absent systems (Redis, SQLite history tables, thread pools, HTTP webhooks).
- **False-Premise Questions** (4 queries): Queries presupposing nonexistent functions (`retry_all()`), unsupported parameters (`max_delay`), or erroneous assumptions (handling negative attempts without error).

---

## 3. Retrieval & Reranker Performance

### Retrieval Accuracy
For all 12 answerable queries, gold evidence was retrieved in the top candidates:
- **Recall@1**: 10/12 (83.3%)
- **Recall@3**: 12/12 (100.0%)
- **Recall@5**: 12/12 (100.0%)
- **MRR**: 0.917

### Score Distribution
- **Gold Evidence Chunks**:
  - Sample Count: 22 matches across queries
  - Score Range: `[-0.04, +7.79]`
  - Mean Score: `+5.04`
- **Irrelevant Candidates**:
  - Sample Count: 78 non-gold chunks
  - Score Range: `[-11.00, +6.61]`
  - Mean Score: `-4.00`

Score margin between average gold and average irrelevant candidates is **9.04 logit points**.

---

## 4. Relevance Threshold Analysis & Trade-Offs

The cross-encoder `ms-marco-MiniLM-L-6-v2` produces raw, uncalibrated logit scores.

Evaluating candidate threshold values against the labeled set reveals the precision/recall trade-off:

| Threshold | Answerable Gold Retained (%) | Unanswerable Filtered (%) | False Negatives | False Positives |
| :---: | :---: | :---: | :---: | :---: |
| **-20.0** | 100.0% | 0.0% | 0 | 8 |
| **-15.0** (Current `.env`) | 100.0% | 0.0% | 0 | 8 |
| **-12.0** | 100.0% | 0.0% | 0 | 8 |
| **-10.0** | 100.0% | 0.0% | 0 | 8 |
| **-8.0** | 100.0% | 25.0% | 0 | 6 |
| **-5.0** | 100.0% | 37.5% | 0 | 5 |
| **-2.0** | 100.0% | 50.0% | 0 | 4 |
| **0.0** (Built-in default) | 100.0% | 50.0% | 0 | 4 |

### Findings:
1. **The `-15.0` Threshold is Too Permissive**: At `-15.0`, the threshold is effectively disabled for this corpus. Every unanswerable query (e.g. asking about Redis or database tables) retains 5 irrelevant chunks with scores around `-10.5` and forwards them to the LLM.
2. **Threshold `0.0` vs `-1.0` for User Queries**: For specific user questions, all gold chunks scored between `-0.04` and `+7.79`. A threshold of `-1.0` or `0.0` retains **100% of answerable gold chunks** while filtering out **50% of unanswerable noise queries** before the prompt is even constructed.
3. **Difference for Broad Multi-Query README Generation**: In contrast to targeted questions, broad multi-angle queries used by `generate_readme` (e.g. `"project overview purpose architecture"`) produce lower cross-encoder scores (down to `-11.26`). Therefore, a single global threshold is suboptimal: specific QA queries benefit from a threshold near `-1.0` to `0.0`, while broad synthesis queries require a lower threshold (or threshold bypass).

---

## 5. Diagnosis of ASK's Conservative Behavior (100% False Abstentions)

### Root Cause Analysis

In all 12 answerable queries, `qwen2.5-coder:1.5b` outputted `INSUFFICIENT_EVIDENCE` as its opening token, despite internally finding and generating the correct answer immediately after!

#### Examples from Actual Raw Outputs:
- **`q06-backoff-defaults`**:
  > *"INSUFFICIENT_EVIDENCE\n\nThe evidence provided does not explicitly state the default values for base_seconds and jitter in compute_backoff. However, based on the function definition in workers.py, the default values are: base_seconds: 0.5, jitter: 0.1."*
- **`q07-backoff-ret`**:
  > *"INSUFFICIENT_EVIDENCE\n\nThe evidence provided does not explicitly state the return type of compute_backoff. However, based on the function definition in workers.py, the return type is float."*
- **`q08-retry-params`**:
  > *"INSUFFICIENT_EVIDENCE\n\nThe evidence provided does not directly state the parameters of should_retry and their default values. However, based on the function definition in workers.py, the parameters are: attempt (int), max_attempts (int, default=3)."*
- **`q09-retry-ret`**:
  > *"INSUFFICIENT_EVIDENCE\n\nThe evidence provided does not explicitly state the return type of should_retry. However, based on the function definition in workers.py, the return type is bool."*
- **`q10-retry-cond`**:
  > *"INSUFFICIENT_EVIDENCE\n\nThe evidence provided does not directly state the condition under which should_retry returns True. However, based on the code provided in workers.py, should_retry returns True when attempt is between 0 and max_attempts - 1 (inclusive)."*

#### Why does the model do this?
1. **Conflicting Prompt Rules for Code Evidence**:
   - Rule 3 states: *"If the evidence does not support an answer, reply with INSUFFICIENT_EVIDENCE on the first line..."*
   - Rule 6 states: *"Distinguish what the evidence states explicitly from what you infer; start any inference with 'Inference:'."*
   - For a 1.5B parameter coder model, reading Python signatures like `def should_retry(attempt: int, max_attempts: int = 3) -> bool:` is treated as *code data* from which answers must be inferred rather than explicit prose sentences stating *"The return type is bool"*. The model concludes that because the prose does not explicitly state it, Rule 3 mandates prefixing the output with `INSUFFICIENT_EVIDENCE`.
2. **Binary Tripwire in `trustdoc/workflows/ask.py`**:
   - `ask.py` lines 201-206 check `if raw.startswith("INSUFFICIENT_EVIDENCE"):` and immediately discard the entire response, converting it to `AskStatus.INSUFFICIENT_EVIDENCE`.
   - The accurate answer written in the explanatory note is suppressed.

---

## 6. Recommended Smallest Justified Improvements

### Recommendation A: Prompt Framing for Code Evidence (Smallest Change)
In `trustdoc/workflows/prompts.py`, adjust Rule 1 and Rule 3 of `SYSTEM_PROMPT` to clarify that Python code signatures, type annotations, and function definitions constitute explicit evidence:
```text
- 1. Answer ONLY from the evidence supplied in the user message. Do not use outside knowledge about the project.
+ 1. Answer from the evidence supplied (both prose and Python code). Python function signatures, parameter defaults, and type annotations are valid direct evidence.
- 3. If the evidence does not support an answer, reply with INSUFFICIENT_EVIDENCE on the first line, then briefly say what is missing.
+ 3. Only reply with INSUFFICIENT_EVIDENCE if the evidence contains no relevant code or prose to answer the question. If the evidence answers the question, answer directly and cite the evidence ID.
```

### Recommendation B: Dual Thresholding Strategy
1. For targeted user questions in `AskService`, calibrate `relevance_threshold` to `-1.0` (or `0.0`). This safely retains 100% of true answers while filtering 50% of irrelevant noise before LLM prompting.
2. For multi-query synthesis in `generate_readme`, do not enforce the question-level relevance threshold (or use a permissive threshold like `-15.0`) because broad synthesis queries inherently produce lower cross-encoder similarity logits against specific code snippets.
