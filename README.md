# TrustDoc AI

**Hybrid RAG-based software documentation generation, verification and hallucination detection.**

> Every documentation claim should be supported by evidence.

This repository is built in 8 sequential parts. **Current status: Part 7 - Streamlit Web UI**
(on top of Parts 1-6.2).

## Setup

Requires Python 3.10+. The core is standard-library only; PDF ingestion needs PyMuPDF, which is an
**optional extra** (`pdf`), not a mandatory dependency.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,pdf,retrieval]"  # or: pip install -r requirements.txt
                                     # omit ",pdf" / ",retrieval" for a lightweight install; the
                                     # tests that need them are then skipped, not failed

cp .env.example .env                 # optional; defaults work as-is
python -m trustdoc init              # creates data/ and the SQLite database
python -m trustdoc info              # shows settings, schema version, row counts
pytest                               # run the whole test suite
```

Optional extras: `.[pdf]` (PyMuPDF), `.[retrieval]` (sentence-transformers, faiss-cpu, rank_bm25, numpy),
`.[ui]` (Streamlit), `.[all]`. The default install stays lightweight: importing
`trustdoc.retrieval` without the extra still gives you the pure modules (types, tokenizer, RRF,
threshold); anything that needs numpy/FAISS/torch raises a clear `MissingDependencyError`. Without `pdf`, Python and
Markdown ingestion work normally and ingesting a PDF raises a clear `MissingDependencyError`.

## Configuration

`Settings.from_env()` layers: defaults < `.env` < real environment variables. All keys are
prefixed `TRUSTDOC_` (see `.env.example`). Invalid values raise `ConfigError` at startup.
Sections: `storage`, `logging`, `chunking`, `retrieval`, `llm`. Retrieval keys (Part 3): `EMBEDDING_MODEL`, `RERANKER_MODEL`, `CANDIDATE_K`, `FINAL_K`, `RRF_K`, `RELEVANCE_THRESHOLD` (0.0 = uncalibrated placeholder), `DEVICE` (default `cpu`), `EMBEDDING_BATCH_SIZE`. Indexes are stored under `<data_dir>/indexes`. Verification keys (Part 5): `VERIFY_WEIGHT_ACCURACY/COVERAGE/GROUNDING`, `VERIFY_PENALTY_CRITICAL/HIGH/CAP`, `VERIFY_EVIDENCE_TOP_K`, `VERIFY_EVIDENCE_TYPES`, `VERIFY_MAX_SEMANTIC_CLAIMS` (see `.env.example`). The LLM section stores only the
*name* of the environment variable holding an API key, never the key itself.

## Directory layout

```
trustdoc/
  config.py          settings (immutable dataclasses) + validation
  logging_setup.py   console + rotating-file logging under the "trustdoc" namespace
  exceptions.py      TrustDocError hierarchy
  app.py             bootstrap(): dirs -> logging -> database -> AppContext
  cli.py, __main__   `python -m trustdoc init|info`
  models/            enums.py (controlled vocabularies), records.py (dataclasses)
  db/                schema.py (versioned DDL), connection.py (Database, Session),
                     repositories.py (all SQL lives here)
  parsing/           pure parsers, no DB/filesystem: python_analyzer.py (ast), markdown_parser.py,
                     pdf_reader.py (PyMuPDF), textutil.py
  chunking/          text.py (the one reusable windowing algorithm), builders.py (Python /
                     Markdown / PDF chunk construction with provenance)
  ingestion/         storage.py (safe names + upload store), extraction.py (pure dispatch),
                     service.py (IngestionService: orchestration + persistence)
  retrieval/         types.py (Evidence, SearchResult...), text.py (tokenizer), fusion.py (RRF),
                     threshold.py, embedding.py, faiss_index.py, bm25_index.py, reranker.py,
                     index_manager.py (build/load/staleness), retriever.py (the facade), factory.py
  evaluation/        dataset.py, metrics.py, runner.py + data/ (sample project, queries.json)
  llm/               base.py (LLMBackend, GenerationResult), ollama.py, hf_transformers.py,
                     stub.py (extractive stand-in), factory.py (create_llm, register_backend)
  workflows/         ask.py (AskService), generation.py (GenerationService), prompts.py,
                     citations.py, types.py, render.py
  verification/      extraction.py (claims from Markdown + docstrings), resolver.py, static_verifier.py (Tier 1),
                     semantic_verifier.py (Tier 2, optional), scoring.py, findings.py (severity), typenorm.py,
                     engine.py (orchestration + persistence), render.py, types.py
examples/            retrieval_demo.py, ask_demo.py, verify_demo.py, generate_demo.py
evaluation_results/  saved evaluation runs (see its README)
tests/               pytest suite
data/                runtime data (gitignored): trustdoc.db, uploads/, logs/
```

## Architecture (current)

```
cli / (future Streamlit UI)
        |
     app.bootstrap() --> Settings, logging, Database
        |
 IngestionService ---> parsing/ + chunking/ (pure)      UploadStore (data/uploads)
        |                                                   
 Retriever ---> IndexManager (FAISS + BM25 in data/indexes) -> RRF -> Cross-Encoder -> Evidence
        |
 AskService ---> Retriever -> grounded prompt -> LLMBackend (Ollama | Transformers) -> checked citations
        |
 GenerationService ---> Retriever -> grounded prompt -> LLMBackend -> checked citations
        |                                                                      |
        +-----------------------------> VerificationEngine <-------------------+
                                               |
                                claims -> Tier 1 static (AST entities) -> [Tier 2 semantic]
                                               |
                                     findings + Trust Score -> DB
        |
        |
   Database.session()  -- one transaction per `with` block (commit / rollback)
        |
   Session: projects | sources | entities | chunks | claims | runs | findings   (repositories)
        |
   SQLite (foreign keys ON, WAL, schema versioned with PRAGMA user_version)
```

Dependency direction: `ingestion -> chunking -> parsing -> models`, and `ingestion -> db`. Parsers and
chunkers never touch the database or filesystem; repositories hold all SQL.

Principles: models know nothing about SQL; repositories are the only SQL; later parts depend on
`models` and `db` but not on each other's internals. Enum values drive both Python code and the
database CHECK constraints, so the two cannot drift.

## Data model

| Table | Purpose |
|---|---|
| `projects` | A documentation project; owns everything below (cascade delete) |
| `sources` | An uploaded file: `python` / `markdown` / `pdf`, content hash, metadata. Unique name per project |
| `code_entities` | Module/class/function/method with parameters (JSON), return annotation, docstring, decorators, line span, parent link |
| `chunks` | Retrievable text with provenance: source, order, line span, PDF page, Markdown heading, optional entity link |
| `claims` | A checkable documentation statement (`entity_reference`, `parameter`, `return`, `behavioral`), `existing_doc` or `generated` |
| `verification_runs` | One verification execution: status, config snapshot, summary, final Trust Score |
| `trust_score_components` | Ordered score breakdown lines: raw score, weight, signed contribution, explanation |
| `claim_results` | Per-run verdict for a claim: `supported` / `contradicted` / `insufficient_evidence` |
| `evidence_links` | Chunks cited for a claim result, with stance, relevance score and rank |
| `findings` | Detected issues: tier, issue type, severity, message, suggestion, links to source/entity/claim |

Notes:
- Tier 2 has two separate issue types, `contradicted_claim` and `unsupported_claim`, so reports can
  never conflate "contradicted by evidence" with "not supported by available evidence".
- Deleting a source cascades to its entities, chunks and claims; findings from past runs are kept
  with their links set to NULL.
- Generated documentation is meant to be stored as a Markdown source (`metadata.generated = true`),
  so it re-enters the same chunking and claim pipeline.

## Ingestion (Part 2)

### Supported inputs

| Type | Extensions | Notes |
|---|---|---|
| Python | `.py` | Parsed with `ast`; never imported or executed |
| Markdown | `.md`, `.markdown` | UTF-8; ATX and Setext headings |
| PDF | `.pdf` | Text-based only (PyMuPDF). **No OCR**; image-only pages yield no chunks and a warning |

### Using it

```python
from trustdoc.app import bootstrap
from trustdoc.ingestion import IngestionService
from trustdoc.models import Project

ctx = bootstrap()
with ctx.db.session() as s:
    project = s.projects.create(Project(name="my-project"))

svc = IngestionService(ctx.db, ctx.settings)
svc.ingest_file(project.id, "path/to/module.py", name="pkg/module.py")  # name = citation name
svc.ingest_bytes(project.id, "README.md", b"# Title\n...")
batch = svc.ingest_directory(project.id, "path/to/repo")                # recursive, symlinks ignored
print(batch.ok, batch.failures, batch.skipped_unsupported)
```

### Workflow

```
sanitize name -> SHA-256 -> unchanged? stop -> extract (pure; may fail) -> store original
   -> one DB transaction: replace old source / register source / persist entities / persist chunks
```

* **Atomic.** Extraction runs before anything is written, and all database writes share one
  transaction. A malformed file raises `ParsingError` and leaves no source row, entities, chunks or
  stored file; a failed *update* keeps the previous version. If the transaction fails after the file
  was written, a newly created file is removed.
* **Duplicates.** Same name + same content -> `UNCHANGED` (nothing reprocessed; `force=True` overrides).
  Same name + new content -> `UPDATED`: the old source is replaced (its entities, chunks and claims
  cascade away) and gets a new id. Same content under a different name -> ingested as its own source
  with a warning and a shared stored file. Names are unique per project, so projects are isolated.
* **Batch.** `ingest_directory` skips `.git`, `__pycache__`, virtualenvs, `node_modules`, build output and
  hidden directories; a failing file is reported in `failures` and does not abort the batch.
* Files over 25 MB (`max_file_bytes` on the service) are rejected.

### How source files are stored

* `Source.name` is a normalised **project-relative POSIX path** (`pkg/a.py`). It is the only identifier
  used in citations. Backslashes are converted, `./` and duplicate slashes removed; absolute paths,
  drive letters, `~`, `..` segments, control characters and over-long names are rejected.
* Originals are kept byte-for-byte at `data/uploads/project_<id>/<sha256><ext>` (content-addressed; the
  extension is fixed by source type). Filenames never come from user input, so uploads cannot escape the
  uploads directory. `Source.stored_path` is relative to the uploads directory and is never a citation.
* `Source.metadata` records per-type details (see below), `warnings`, and `duplicate_of` when relevant.

### Python AST analysis (`parsing/python_analyzer.py`)

Decoded (PEP 263 / BOM aware), newline-normalised, then `ast.parse`d. Extracted per entity: type
(module/class/function/method), qualified name, parent, parameters (name, kind, annotation, default as
exact source text), return annotation, docstring, decorators (source text), async flag, public/private,
start/end line.

* Qualified names: `pkg.mod.Class.method`; entities inside functions use `<locals>`
  (`pkg.mod.outer.<locals>.inner`). Module name comes from the source name (`pkg/__init__.py` -> `pkg`).
  A `src/` prefix in the name is kept; later parts can match by suffix.
* `start_line` of a decorated entity is its first decorator line; `end_line` comes from the AST.
* Public = name does not start with `_` (dunder methods such as `__init__` count as non-public), not
  inside a function, and every enclosing class/module is public.
* Definitions inside `if`/`try`/`with`/`for`/`while`/`match` blocks are found.
* Syntax errors raise `ParsingError` with file and line number; nothing is imported or run.

### Chunking (`chunking/`)

One algorithm, `chunk_text(text, max_chars, overlap_chars, base_line)`, serves all three source types
(config: `TRUSTDOC_CHUNK_MAX_CHARS`, `TRUSTDOC_CHUNK_OVERLAP_CHARS`). It cuts at paragraph breaks, then
line breaks, sentence ends, spaces, and hard-cuts only as a last resort; overlap starts at a line/word
boundary. Every segment knows its character offsets and exact line range.

* **Python:** one chunk per function/method (whole definition incl. decorators and nested functions);
  a class chunk holds only the class's own lines (header, docstring, attributes) while its methods and
  nested classes get their own chunks; module-level code (imports, constants, scripts) forms module
  chunks. Each links to its `CodeEntity` via `entity_id`. Over-long regions split with `part`/`parts`.
  Conditional definitions can leave small module-level chunks for wrapper lines such as `else:`.
* **Markdown:** one chunk per heading section (any level), `Chunk.heading` = full path such as
  `Project > Installation`, plus level/path/section range in metadata. Headings inside fenced code and YAML
  front matter are ignored; heading-only sections are skipped (the heading survives in descendants' paths).
* **PDF:** text extracted page by page; chunks never span pages and always carry `page_number`;
  `char_start`/`char_end` index into that page's extracted text. Blank/image-only pages are recorded in
  source metadata (`empty_pages`, `image_only_pages`) with a warning.

Chunk text uses `\n` newlines. Python/Markdown chunk text is the literal source lines
`start_line..end_line`.

### Provenance and citations

| Source | Chunk provenance | Citation (`Chunk.location_label(source.name)`) |
|---|---|---|
| Python | source, entity, `start_line`-`end_line`, qualified name | `pkg/a.py:12-30` |
| Markdown | source, heading path, line range | `README.md:6-12 § Project > Installation` |
| PDF | source, `page_number`, char range | `docs/ref.pdf, p.3` |

`metadata["context_header"]` holds a ready-made label (e.g. `pkg/a.py :: pkg.a.Store.put`) that later
retrieval stages can prepend to chunk text.

## Retrieval (Part 3)

### Architecture

```
                         User query
                              |
              +---------------+---------------+
              v                               v
       BM25 keyword search            FAISS semantic search
      (rank_bm25, exact terms)    (Sentence-Transformer embeddings)
              |  ranked list                  |  ranked list
              +---------------+---------------+
                              v
                    Reciprocal Rank Fusion          (ranks only; never mixes scores)
                              v
                top-N fused candidates (N = max(candidate_k, top_k))
                              v
              Cross-Encoder rerank (candidates only, never the whole KB)
                              v
                      ranked Evidence list
                              v
                   relevance threshold (reranker score)
                  /                          \
        score >= threshold            best score < threshold
     return evidence (+ provenance)   abstain: "insufficient evidence"
```

Later parts call one method and never touch FAISS, BM25, RRF, the Cross-Encoder or index files:

```python
from trustdoc.app import bootstrap
from trustdoc.retrieval import create_retriever

ctx = bootstrap()
retriever = create_retriever(ctx.db, ctx.settings)         # models stay unloaded until first use
result = retriever.search(project_id, "how are retries delayed?", top_k=5)

result.abstained            # True -> insufficient evidence
result.evidence             # Evidence list that passed the threshold
result.ranked               # top_k BEFORE the threshold (for evaluation / inspection)
for ev in result.evidence:
    ev.citation.label       # 'taskqueue/workers.py:47-52' | 'README.md:20-23 § TaskQueue > Retry policy' | 'reference.pdf, p.2'
    ev.text, ev.bm25_rank, ev.faiss_score, ev.rrf_score, ev.rerank_score, ev.final_rank
```

`search(..., mode=RetrievalMode.BM25 | FAISS | HYBRID | HYBRID_RERANK)` selects the pipeline (default
`HYBRID_RERANK`); `threshold=` overrides the configured threshold for one call.

### Installing the retrieval dependencies

```bash
pip install -e ".[dev,pdf,retrieval]"
```

`sentence-transformers` pulls in PyTorch. On a CPU-only machine install the small CPU wheel first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. The two models
(`all-MiniLM-L6-v2` embeddings and `cross-encoder/ms-marco-MiniLM-L-6-v2`) download from Hugging Face on
first use and are cached by Hugging Face afterwards. Everything runs on CPU by default
(`TRUSTDOC_DEVICE=cpu`). On macOS, a crash mentioning duplicate OpenMP libraries is a known
FAISS + PyTorch interaction; setting `KMP_DUPLICATE_LIB_OK=TRUE` is the usual workaround.

### How indexes are built and loaded

Indexing is an explicit operation, never part of a query. Each project has its own directory:

```
data/indexes/project_<id>/
    manifest.json     build id/time, embedding model + dimension, chunk/source counts, knowledge hash, metric
    faiss.index       FAISS vectors            faiss_ids.npy   vector position -> chunk id
    bm25.json         tokenised corpus + chunk ids (JSON, no pickle)
```

```python
manager = retriever.index_manager
manager.build(project_id)          # no-op if fresh; embeds every chunk once
manager.status(project_id)         # MISSING | FRESH | STALE | CORRUPT, with human-readable reasons
manager.rebuild(project_id)        # force a full rebuild
manager.ensure(project_id)         # load if fresh, otherwise (re)build then load
manager.metadata(project_id)       # the manifest
```

* **Staleness** is decided from the database: the manifest stores a hash of the project's
  `(chunk id, source id, char count)` rows and its sources' content hashes, plus the embedding model
  name, tokenizer version and index format version. Adding, replacing or removing a source (Part 2
  creates new chunk ids), or changing the embedding model, makes the index stale. An unchanged re-ingest
  does not. The MVP rebuilds the whole project index; there is no incremental update.
* **`search()` auto-indexes** (`auto_index=True`, default): a missing/stale index is rebuilt first. With
  `auto_index=False` it raises `IndexNotBuiltError` / `StaleIndexError` instead.
* **Loading**: indexes load from disk once per process and are cached; models are loaded lazily once and
  shared. Neither is reloaded per query (tests verify the corpus is embedded exactly once).
* **Builds are all-or-nothing**: files are written to a temporary directory and swapped in only when
  complete; a failed rebuild leaves the previous index intact.
* Index and embedding text per chunk is `context_header + chunk text` (e.g. `pkg/a.py :: pkg.a.Store.put`),
  so class and module names match even when a method's source only mentions its own name. It contains
  logical names only, never file paths.

### FAISS (semantic search)

Chunks are embedded with the configured Sentence-Transformer (default
`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) and L2-normalised. They go into a FAISS
`IndexFlatIP`, an exact brute-force index. **Metric: inner product on unit vectors = cosine similarity;
scores are similarities in [-1, 1] and higher is better** (they are not distances). FAISS only knows vector
positions, so the position -> chunk id mapping is persisted next to the index. Note MiniLM truncates
inputs at 256 word-pieces, so very long chunks are only partly embedded.

### BM25 (keyword search)

`rank_bm25.BM25Okapi` (k1=1.5, b=0.75) over a shared tokenizer that keeps identifiers *and* their parts
(`load_config` -> `load_config`, `load`, `config`; `WorkerPool` -> `workerpool`, `worker`, `pool`), so exact
function/class/parameter/config-key queries get exact-match evidence while plain words still match parts.
Chunks with score 0 are never returned. BM25's IDF is ~0 for a term in half the corpus, so tiny corpora
(a handful of chunks) can return few hits; real projects are fine.

### Reciprocal Rank Fusion

`RRF(d) = sum over lists of 1 / (rrf_k + rank(d))` with `rrf_k` from configuration (default 60). Only ranks
are used: BM25 scores and cosine similarities live on different scales and are never normalised or added.
A chunk found by both retrievers collects a contribution from each, e.g. rank 1 in BM25 and rank 2 in FAISS
scores `1/61 + 1/62`. Ties break deterministically (better best-rank, then lower chunk id).

### Cross-Encoder reranking

A bi-encoder embeds query and chunk separately; a Cross-Encoder reads them *together* and outputs a
relevance score, which is more accurate but too slow for the whole knowledge base. So only the fused
candidates (at most `max(candidate_k, top_k)` chunks, default 20) are scored with
`cross-encoder/ms-marco-MiniLM-L-6-v2`. The original BM25/FAISS/RRF ranks and scores stay on each
`Evidence` next to `rerank_score`. The reranker is any object with `score(query, texts)`, so it is
replaceable.

### Relevance threshold and abstention

After reranking, if the best reranker score is below `TRUSTDOC_RELEVANCE_THRESHOLD`, the result is
`abstained=True` ("insufficient evidence") and `evidence` is empty; otherwise `evidence` holds the chunks
scoring `>=` the threshold (the boundary is inclusive). `ranked` always keeps the pre-threshold list.

> **The shipped default of `0.0` is an uncalibrated placeholder, not a meaningful threshold.** Cross-Encoder
> score scales depend on the model and sentence-transformers version (probabilities in [0, 1] or raw
> logits), so the right value must be found experimentally, e.g. by collecting best reranker scores for
> queries known to be answerable and known to be unanswerable and choosing a cut-off that separates them.
> A warning is logged once when the placeholder is in use. The threshold applies only to the reranked
> pipeline; BM25 scores and cosine similarities are not comparable with it. Part 4 turns an abstention into
> a natural-language answer.

### Provenance and citations

Every `Evidence` carries `citation` (logical source name, source type, label, line range / page /
heading, qualified name, chunk id) built from `Source.name` and chunk metadata. The physical `stored_path`
is never read by the retrieval layer; tests assert it (and the uploads directory) never appears in any
serialised result. The retriever also refuses to return a chunk whose `project_id` differs from the one
searched.

### Retrieval evaluation

```bash
python -m trustdoc eval-retrieval --out evaluation_results/real_models_run          # real models
python -m trustdoc eval-retrieval --embedder hashing --reranker overlap --out /tmp/eval   # offline stand-ins
python examples/retrieval_demo.py "retry delay between attempts"                    # inspect one query
```

The command ingests a bundled sample project (`trustdoc/evaluation/data/sample_project`: 4 Python modules,
a README, a 3-page PDF) into an isolated temporary workspace (your database is untouched), builds the
index, and runs four pipelines: **A** BM25 only, **B** FAISS only, **C** FAISS + BM25 + RRF, **D** C +
Cross-Encoder. Method:

* 16 queries in the 7 required types (exact function name, exact parameter name, semantic paraphrase,
  class-related, README, PDF, mixed code + documentation) in `queries.json`.
* Relevance labels name chunks by what they are (source + qualified name / heading path / PDF page), so
  they survive re-ingestion; an unmatched label fails the run loudly.
* **Recall@K** (K = 1, 3, 5, 10): fraction of a query's relevant items with at least one chunk in the top K.
  **MRR**: mean of 1 / rank of the first relevant chunk (0 if none). Computed on the ranking *before* the
  threshold. Mean warm search latency is also reported.
* Outputs: `results.json` (metrics, per-query rankings, environment, caveats), `per_query.csv`,
  `summary.md`. Every report states whether stand-in models were used.

**Read results with care.** The sample project and queries are an *illustrative fixture*, far too small for
statistical conclusions; its "semantic" queries still share some words with the corpus, so it cannot
cleanly separate keyword from semantic retrieval. The run saved in `evaluation_results/stand_in_models_run/`
used lexical stand-ins and is **not** a measurement of the real models. Results for the real models appear
only after you run the first command above on a machine that can download them.

## Ask workflow (Part 4)

`AskService.ask(project_id, question)` answers a question about an ingested project **using only retrieved
evidence**, cites that evidence, and abstains when the evidence is insufficient. It orchestrates; it does
not implement retrieval (Part 3) or text generation (the `llm` package).

### Pipeline

```
question -> validate (non-empty, within length limit, project exists)
         -> Retriever.search(project, question, top_k)            [Part 3: BM25 + FAISS + RRF + Cross-Encoder + threshold]
         -> retrieval abstained / no usable evidence?  --yes-->  "insufficient evidence" (LLM NOT called)
         -> build grounded prompt: system rules + question + fenced evidence blocks [E1]..[En]
         -> LLMBackend.generate(...)                              [Ollama | Transformers | stub]
         -> model replied INSUFFICIENT_EVIDENCE?        --yes-->  "insufficient evidence" (evidence kept)
         -> check citation IDs against the evidence actually shown -> AskResult
```

`AskResult.status` is `answered`, `insufficient_evidence`, `retrieval_error` or `generation_error`. Failures
never produce a made-up answer, and the retrieved evidence is kept on the result either way so Part 5 can
verify claims independently. Unknown projects and empty/over-long questions raise (`NotFoundError`,
`InvalidQuestionError`); questions are never truncated.

### Choosing and configuring a local LLM (no paid API, no key)

Nothing is downloaded at startup. The default is `TRUSTDOC_LLM_PROVIDER=none`: `ask` can still report
"insufficient evidence", but needs a model to generate answers. Embedding models and Cross-Encoders are not
generative and are never used for answers.

| Provider | What it is | Extra Python deps | When to use |
|---|---|---|---|
| `ollama` (recommended) | A local [Ollama](https://ollama.com) server over HTTP | none (stdlib) | Easiest on a laptop CPU; models are managed by Ollama |
| `transformers` | Hugging Face model loaded in-process | `torch`, `transformers` (already pulled in by the `retrieval` extra) | No separate server wanted |
| `stub` | Deterministic extractive stand-in, **not a language model** | none | Smoke tests and demos only |

**Ollama (recommended).** Install Ollama, then:

```bash
ollama pull qwen2.5:1.5b            # example small model; check the Ollama library for current names/sizes
export TRUSTDOC_LLM_PROVIDER=ollama
export TRUSTDOC_LLM_MODEL=qwen2.5:1.5b
# optional: TRUSTDOC_LLM_BASE_URL=http://localhost:11434 (default), TRUSTDOC_LLM_TIMEOUT_SECONDS=300
```

**Transformers (alternative).**

```bash
export TRUSTDOC_LLM_PROVIDER=transformers
export TRUSTDOC_LLM_MODEL=<an instruction-tuned chat model id from Hugging Face>
```

The model downloads on first use if it is not cached (set `HF_HUB_OFFLINE=1` to forbid downloads).

**Resource expectations (rough estimates, not benchmarks).** Memory is roughly *parameters x bytes per weight*:
Ollama models are usually quantised (commonly ~4-bit, about 0.5 byte/parameter, so a 1.5B-parameter model is on
the order of 1 GB on disk and in RAM plus overhead); the in-process Transformers backend loads the library's
default precision (float32 on CPU, ~4 bytes/parameter, so a 1.5B model needs on the order of 6 GB RAM). Disk use
equals the model files. CPU generation is slow: expect seconds to minutes per answer depending on model size,
prompt length (default evidence budget is ~6,000 characters) and your CPU. These figures have not been measured
here; do not assume a large model will run comfortably on an ordinary CPU. `TRUSTDOC_LLM_TIMEOUT_SECONDS`
defaults to 120 and may need raising. The prompt is sized to fit a 4096-token context (`TRUSTDOC_LLM_CONTEXT_TOKENS`,
passed to Ollama as `num_ctx`); a warning is attached when the estimated prompt may not fit.

### Worked example

```python
from trustdoc.app import bootstrap
from trustdoc.ingestion import IngestionService
from trustdoc.models import Project
from trustdoc.workflows import create_ask_service, render_ask_result

ctx = bootstrap()
with ctx.db.session() as s:
    project = s.projects.create(Project(name="my-project"))
IngestionService(ctx.db, ctx.settings).ingest_directory(project.id, "path/to/repo")   # Part 2

service = create_ask_service(ctx.db, ctx.settings)        # real Part 3 retriever + configured LLM (both lazy)
result = service.ask(project.id, "How does the retry mechanism work?")
print(render_ask_result(result))
```

Or from the command line (the project must already be ingested; there is no CLI for creating or ingesting
projects yet):

```bash
python -m trustdoc ask --project-id 1 --question "How does the retry mechanism work?"
python -m trustdoc ask --project my-project -q "..." --show-evidence      # also print short evidence snippets
python -m trustdoc ask --project-id 1 -q "..." --json                     # full structured result
python -m trustdoc ask --help
```

Exit codes: `0` answered or "insufficient evidence"; `1` operational failure (retrieval or generation);
`2` usage/configuration problem (unknown project, invalid question, no model configured, missing dependency).
Errors go to stderr. The CLI never prints the prompt, secrets, or full source text.

Output shape (produced by `examples/ask_demo.py` with the **stand-in** retrieval and stub, to show the format
only; it is not a real answer):

```
Result: answered | model: extractive-stub (stub)  [STAND-IN, not a language model] | citation check: cited

Answer:
Stand-in response (not a language model). The top retrieved evidence [E1] begins: "def compute_backoff(...

Sources cited:
  [E1] taskqueue/workers.py:47-50  (taskqueue.workers.compute_backoff)

Retrieved evidence not cited in the answer:
  [E2] reference.pdf, p.2
  [E5] README.md:20-23 § TaskQueue > Retry policy
```

### How citations map to evidence

Each evidence item shown to the model gets a stable ID (`E1`, `E2`, ... in rank order) and is rendered as a fenced
block carrying its exact retrieval citation label (`pkg/mod.py:10-24`, `reference.pdf, p.3`,
`README.md:4-9 § A > B`) and qualified name. The model is told to cite with those IDs. Afterwards a
deterministic check parses `[E1]`, `[E1, E3]`, `[E1][E2]` and:

* maps valid IDs back to the real evidence (`AskResult.citations`: label, source, lines/page/heading, chunk id);
* never accepts unknown IDs (including IDs of evidence that was dropped by the prompt budget): they are replaced
  in `answer` with `[unknown source E9]`, listed in `invalid_citation_ids`, and never "fixed" by guessing;
  `raw_answer` always holds exactly what the model wrote;
* reports `grounding`: `cited`, `cited_with_invalid_references`, or `uncited` (no valid citation, flagged
  in `warnings` so the answer is not presented as grounded).

This checks that cited IDs exist, **not** that the cited text supports the claim next to it. `claims_verified`
is always `False` in Part 4; claim-level verification is Part 5.

### How abstention works

1. **Retrieval abstention:** if Part 3 reports `abstained` (best reranker score below the threshold) or yields no
   evidence, the result is `insufficient_evidence` and the LLM is not called. The closest sources are listed
   (labels and scores only) for transparency.
2. **Model-declared:** the prompt asks the model to start with `INSUFFICIENT_EVIDENCE` when the evidence does not
   support an answer; that is reported as `insufficient_evidence` with `abstention_reason=model_declared_insufficient`.
   Small models may ignore this instruction.

### Safety measures and limits

* Evidence is fenced, declared untrusted in the system prompt, stripped of control characters, and text that
  imitates our fence markers or citation IDs is neutralised before it reaches the model. This **reduces but does
  not eliminate** prompt-injection risk, and prompting cannot guarantee factual correctness.
* Prompt limits (`TRUSTDOC_ASK_MAX_EVIDENCE_ITEMS`, `..._MAX_EVIDENCE_CHARS`, `..._MAX_QUESTION_CHARS`): extra
  evidence is dropped (and reported), only the top item may be truncated (with an explicit marker), and
  over-long questions are rejected.
* Retrieved evidence from another project is discarded even though retrieval already prevents it.

### Known limitations

* **The retrieval relevance threshold is still the uncalibrated placeholder `0.0`.** A result that did not abstain
  does not show the evidence is sufficient; a warning is attached whenever the placeholder is in effect.
* **The real Sentence-Transformer embedder and Cross-Encoder have not been verified end-to-end** in the original
  development environment (Part 3 was tested with stand-ins), so end-to-end answer quality with real models is
  unknown.
* **The LLM backends have not been run against a real model.** The Ollama backend is tested against a local HTTP
  server that mimics Ollama's documented `/api/chat` contract; the Transformers backend against fake `torch` /
  `transformers` modules. Hallucination rates and answer quality are unmeasured.
* One question at a time (no conversation memory), no streaming, English-centric prompts, and the token estimate
  is a rough characters/3 heuristic.

## Documentation verification (Part 5)

`VerificationEngine` extracts **claims** from Markdown documentation and Python docstrings and checks them against the
project's code and evidence. It reports what is **contradicted** (the evidence says otherwise), what is **supported**,
and what is **unverified** (not enough evidence) as three different things: *missing evidence is never treated as proof
that a claim is false.* It does not eliminate hallucinations; it finds a class of documentation errors that can be decided
from the code, and flags the rest for review.

```
documents (Markdown, docstrings, or supplied text)
   -> claim extraction  (deterministic patterns + conservative rules; no LLM)
   -> Tier 1  static verification against AST entities      (always on, deterministic)
   -> Tier 2  semantic verification, behaviour/config claims only   (optional: --semantic, needs a local LLM)
   -> documentation-coverage findings -> Trust Score -> one database transaction
```

### Two tiers

| | Tier 1 - static | Tier 2 - semantic |
|---|---|---|
| Decides | Whether named entities exist; parameter names, defaults, kinds, order; return types vs annotations | Behaviour and configuration statements ("persists jobs to disk") |
| How | `ast`-derived `CodeEntity` records from Part 2; no model, no retrieval | Part 3 retrieval of evidence from the same project + an optional local LLM |
| Always runs | Yes | No: `--semantic`, a configured LLM and retrieval dependencies are all required |
| Can the LLM decide it? | Never | Only through strict validation (below); never ground truth |

Static claims are never sent to the LLM, and Tier 2 never runs for a claim Tier 1 already decided. Without a usable model the
behaviour/config claims are `NOT_CHECKABLE` and every Tier 1 result is unaffected.

### Claim categories and statuses

Categories keep the existing enum names (`ClaimType`); the spec's names map as shown. Category and status are separate fields.

| Category (`ClaimType`) | Meaning | Spec name | Checked by |
|---|---|---|---|
| `entity_reference` | A function/class/method that should exist (`` `retry_all()` ``, "the `Store` class") | ENTITY_EXISTS | Tier 1 |
| `signature` | A documented signature (`` `def f(a, b=1) -> int` ``, `` `f(a, b=1)` ``) *(new in schema v2)* | SIGNATURE | Tier 1 |
| `parameter` | "`f` accepts `a` and `b`", docstring Args sections, parameter tables under an entity heading | PARAMETER | Tier 1 |
| `return` | "`f` returns a float", docstring Returns/rtype | RETURN_TYPE | Tier 1 |
| `behavioral` | "`TaskQueue` persists jobs to disk" | BEHAVIOR | Tier 2 |
| `config_dependency` | "`TASKQ_WORKER_COUNT` sets the number of workers" *(new)* | CONFIG_DEPENDENCY | Tier 2 |
| `other` | Supplied claims that fit nothing else *(new; the extractors do not produce it)* | OTHER | Tier 2 |

| Status (`ClaimVerdict`) | Meaning |
|---|---|
| `supported` | The evidence directly supports the claim |
| `contradicted` | The evidence directly conflicts with the claim (**false**) |
| `insufficient_evidence` | Checked, but the evidence cannot decide it (**unverified, not false**) |
| `not_checkable` | Not checked in this run: semantic checks off/unavailable, external or standard-library name, example code *(new)* |

### Running it

```bash
python -m trustdoc verify --project-id 1                       # all Markdown docs + docstrings, Tier 1 only
python -m trustdoc verify --project my-project --document docs/API.md
python -m trustdoc verify --project-id 1 --document-file NOTES.md   # ingest this file first, then verify it
python -m trustdoc verify --project-id 1 --semantic             # also LLM-assisted checks (see "Semantic checks")
python -m trustdoc verify --project-id 1 --json                 # full structured report
python -m trustdoc verify --show-run 3                          # print a stored historical run
python -m trustdoc verify --help
python examples/verify_demo.py                                  # offline demo on the bundled sample project
```

Python: `VerificationEngine(db, settings).verify_project(project_id)`; `.verify_text(project_id, text)` verifies supplied text
(for example an Ask answer or other generated documentation, stored as a Markdown source with `origin=generated`);
`.list_runs(project_id)`; `.load_report(run_id)`.

Repeated `--document` / `--document-file` values (or repeated names passed to `verify_project`) are de-duplicated, keeping the order of first appearance; naming a source twice never changes the result.

**Exit codes** (a finding is not a crash): `0` completed, nothing at or above `--fail-on`; `3` completed with documentation
findings at or above `--fail-on` (default `high`; `--fail-on none` never fails; INFO never fails); `2` usage/configuration
problem (unknown project, run or document, non-Markdown `--document`); `1` the run itself failed. Errors go to stderr.

### What is (and is not) reported as wrong

Tier 1 is deliberately conservative:

* A documented **subset** of parameters is fine. Omissions are only reported when the documentation claims completeness
  (a `def`-form signature, or "only"/"exactly").
* `self`/`cls` are never required or reported. Call-like examples (`` `f(x, y)` ``) may use any variable names, so only keyword
  names and the number of positional arguments are checked, and defaults are not compared.
* A name is reported **nonexistent** only if it is attributable to the project (a "function/class/method" word, `()`, a signature, or a
  name rooted in a project package). Otherwise it is *unverified*. Imports, assignments and `__all__` entries (aliases, re-exports),
  ambiguous names, classes without an explicit `__init__` (dataclasses, inherited), and callables with decorators that may change
  their signature (anything outside a small allow-list such as `staticmethod`, `property`, `lru_cache`) are *unverified*,
  never contradicted. Builtin/standard-library/third-party-looking names and names in example code are `not_checkable`.
* A return type is compared with the source annotation after normalisation (`Optional[int]` = `int | None`, `List[x]` = `list[x]`).
  **No annotation means `insufficient_evidence`**; types are never inferred. `bool`/`int`, aliases and user-defined names are not
  declared mismatches.
* Missing documentation is a **coverage finding**, not a hallucination: public classes/functions/methods (the Part 2 rules:
  underscore names, dunders and nested definitions are not public) with neither a docstring nor a mention in the verified documents.

### Severity

| Severity | Used for |
|---|---|
| `critical` | A documented *signature* that would break callers: nonexistent API, a parameter that does not exist, a required parameter omitted |
| `high` | Other wrong API claims: nonexistent entity or wrong parameter in prose, wrong default/annotation, wrong return type, positional/keyword-only conflicts |
| `medium` | A behavioural contradiction found by the semantic tier; missing docs on a public class or a callable with arguments |
| `low` | Optional parameters left out of a "complete" list, parameters missing from a docstring Args section, missing docs on a no-argument callable |
| `info` | Unverified claims (insufficient evidence). Never a proven defect; never raised just because evidence is missing |

A call-like mention (not a real signature) is never `critical`. Suggestions are evidence-based: they quote the actual signature or
annotation, offer a "did you mean" only for a close match, and otherwise recommend review. A model's suggested wording, when any,
is labelled unverified.

### Trust Score (0-100)

```
accuracy  = supported / (supported + contradicted + insufficient_evidence)        over claims that were checked
coverage  = public entities with a docstring or a mention / all public entities
grounding = checked claims that are supported or contradicted AND have >= 1 evidence reference / checked claims

base      = 100 x (w_a*accuracy + w_c*coverage + w_g*grounding) / (sum of the weights of the available components)
penalty   = min(40, 10 x critical_contradictions + 5 x high_contradictions)        # contradictions only
score     = clamp(base - penalty, 0, 100)             defaults: w = 0.50 / 0.25 / 0.25  (TRUSTDOC_VERIFY_*)
```

* **Insufficient evidence is never counted as correct**: it stays in the accuracy denominator and has no evidence, so it also lowers
  grounding. `not_checkable` claims are excluded and reported (see "Skipped checks").
* **A score needs at least one checked claim.** With none, the score is *not measurable* and coverage is shown as an
  informational, unscored figure (coverage alone would say nothing about whether the documentation is correct).
* A component with no denominator (no public entities, or coverage not measured, e.g. when verifying supplied text) is *unavailable*:
  dropped and the other weights renormalised; no value is invented. Contradictions found by the semantic tier (MEDIUM) are not penalised.
* The breakdown (value, effective weight, contribution, denominators) is printed, returned, and stored with the run.

Worked example: 8 supported, 1 contradicted, 1 insufficient of 10 checked; 9 grounded; 7 of 10 entities documented; one HIGH
contradiction: accuracy 0.8, coverage 0.7, grounding 0.9, so base = 100 x (0.5x0.8 + 0.25x0.7 + 0.25x0.9) = 80, penalty 5, score **75**.

### Semantic checks (optional, never ground truth)

`--semantic` retrieves evidence for each behaviour/config claim with the Part 3 retriever (this project only) and asks the configured
local LLM (the Part 4 backends) to classify it using **only** that evidence. Safeguards:

* the claim's own document can never be its evidence; by default only code and PDF material may be (`TRUSTDOC_VERIFY_EVIDENCE_TYPES`);
* no usable evidence means `insufficient_evidence` and the model is not called; silence in the evidence is not a contradiction;
* the model must return one JSON object with a verdict (`supported`, `contradicted` or `insufficient_evidence` only), a short explanation,
  and evidence IDs that were actually supplied; supported/contradicted need at least one. Malformed, ambiguous, conflicting or
  fabricated-ID output is **rejected** and recorded as `insufficient_evidence`; evidence IDs map back to real chunks;
* results are marked model-assisted, their confidence is not stored, and the claim and evidence are fenced as untrusted text;
* at most `TRUSTDOC_VERIFY_MAX_SEMANTIC_CLAIMS` (25) claims per run, because local CPU inference is slow.

An LLM verdict is a lead for a human reviewer, not a fact. Quality of the verdicts is **not evaluated** here.

### Findings and citations

Each finding has a stable `uid`, the claim's `uid`, category, status, severity, a message, the **documentation location**
(`README.md:21`), the **code location** (`pkg/a.py:10`), evidence citations (logical labels such as `pkg/a.py:10-24` or `reference.pdf, p.3`;
never filesystem paths), and a suggestion. Claim `uid`s are derived from the document name, category and normalised text, so they are
reproducible and independent of line numbers. Findings snapshot their claim text and locations, so history stays readable even if a
document is later replaced. Read `INFO` items as "please check", `LOW`/`MEDIUM` as gaps, `HIGH`/`CRITICAL` as probable errors.

### Persistence, history and schema v2

Everything uses the existing tables: `claims`, `verification_runs`, `claim_results`, `evidence_links`, `findings`,
`trust_score_components`. A run is computed fully in memory and written in **one transaction**; a failure leaves no partial data
(the failed run itself is recorded as `failed` with its error). Each `verify` creates a new run; old runs are never modified, identical
claims are reused rather than duplicated, and `load_report(run_id)` rebuilds any past report.

**Schema v2** was unavoidable: the CHECK constraints could not store `signature`/`config_dependency`/`other` claims, the
`not_checkable` status, or `info` severity. `Database.initialize()` upgrades v1 databases automatically with SQLite's table-rebuild
procedure for the three affected tables (`claims`, `claim_results`, `findings`): one transaction, foreign keys verified before commit,
rolled back on any failure; rows, ids, relationships, indexes and id sequences are preserved (tested against genuine v1 databases).
There is no downgrade. Caveat (existing Part 2 behaviour): replacing a document deletes its claims, which cascades to their results in
older runs; older runs' *findings* and scores remain intact and readable.

### Known limitations and false-positive risks

* Claim extraction is pattern/rule based and misses many claims; behaviour extraction can include sentences that are not really claims and
  skips imperative, hedged, conditional and second-person text. There is no LLM-assisted extraction. Parameter tables/lists are only read under
  a "Parameters"/"Arguments" heading that names an entity.
* Static checks see only indexed Python source: dynamic definitions (`setattr`, metaprogramming), inherited or generated signatures,
  dataclasses, and third-party APIs are unverified or not checkable. Call-like examples can still be mis-read.
* A nonexistent-entity verdict rests on absence from the index; incomplete ingestion can cause one. Add missing sources and re-run.
* Documentation coverage counts a mention in the verified documents as documentation, which is coarse.
* The retrieval threshold is still the uncalibrated placeholder `0.0` and the real embedder/Cross-Encoder and LLM backends have not been
  validated end to end, so semantic results are unevaluated. The Trust Score is a transparent heuristic, not a calibrated probability.

### Running the tests without a model

All verification tests use constructed code entities, a fake retriever or the Part 3 stand-ins, and scripted mock LLMs; nothing is downloaded:

```bash
pytest tests/test_verify_*.py tests/test_schema_v2_migration.py        # Part 5
pytest                                                                  # everything
```

## Documentation generation (Part 6)

Part 6 introduces the grounded **GENERATE** workflow with automated verification.

### Capabilities

1. **README generation**: Creates a structured project README with overview, key modules/components, and setup/usage details backed by ingested evidence. If setup or installation details are absent from the evidence, the generator explicitly notes their absence rather than inventing third-party packages or shell commands.
2. **Python API reference**: Creates reference documentation for any indexed function, method, or class resolved from the AST entity repository. The generator is provided the exact AST signature, parameter kinds, default expressions, annotations, and retrieved contextual evidence.

### Grounding & Safety Guarantees

* **Untrusted data fencing**: All project code and documentation chunks in the prompt are enclosed in strict delimiters (`=== BEGIN EVIDENCE E1 ===`) and declared as data, not instructions.
* **Anti-hallucination rules**: System prompts strictly forbid inventing non-existent parameters, default values, return types, exceptions, or behaviors.
* **Citation validation**: Output text is analyzed for citation IDs (`[E1]`, `[E2]`). Valid IDs map back to real source chunks and line spans. Fabricated IDs (e.g. `[E99]`) are neutralized to `[unknown source E99]` and logged as warnings.
* **Generate → Verify integration**: Generated Markdown is ingested as a Markdown source (`origin=GENERATED`) and automatically verified by the existing Part 5 `VerificationEngine`. The result bundles the generated document together with its independent verification report, defect findings, and measurable Trust Score.

### CLI usage

```bash
# Generate README for project 1
python -m trustdoc generate --project-id 1 --kind readme

# Generate API documentation for a specific function or class
python -m trustdoc generate --project-id 1 --kind api --entity taskqueue.workers.compute_backoff

# Safely write to a file (fails if destination exists without --force; will not write on verification failure unless --allow-unverified is given)
python -m trustdoc generate --project-id 1 --kind readme --output generated_README.md
python -m trustdoc generate --project-id 1 --kind readme --output generated_README.md --force
python -m trustdoc generate --project-id 1 --kind readme --output generated_README.md --allow-unverified

# Also run semantic (LLM-assisted) verification on generated behavioral claims
python -m trustdoc generate --project-id 1 --kind api --entity compute_backoff --semantic

# Output structured JSON
python -m trustdoc generate --project-id 1 --kind readme --json

# Set failure threshold (exits with status 3 on defects at or above severity; default: high)
python -m trustdoc generate --project-id 1 --kind api --entity compute_backoff --fail-on high
```

Exit codes:
* `0`: Generation and verification succeeded with no findings at or above `--fail-on` (or insufficient evidence abstention).
* `1`: Operational failure (retrieval error, LLM failure, or verification error).
* `2`: Usage or configuration error (unknown project/entity, missing flags, existing file without `--force`).
* `3`: Verification detected documentation defects at or above `--fail-on` (e.g. hallucinated parameter or contradicted return type).

### Python API

```python
from trustdoc.app import bootstrap
from trustdoc.workflows import create_generation_service

ctx = bootstrap()
service = create_generation_service(ctx.db, ctx.settings)

# Generate README
res_readme = service.generate_readme(project_id=1)
print(res_readme.markdown)
print(f"Trust Score: {res_readme.trust_score}")

# Generate API documentation
res_api = service.generate_api(project_id=1, entity_name="compute_backoff")
print(res_api.markdown)

# Verification can be skipped explicitly (status becomes 'verification_skipped', is_verified=False):
res_unverified = service.generate_readme(project_id=1, run_verify=False)
assert res_unverified.status == "verification_skipped"
assert res_unverified.is_verified is False
```

### Runnable demo

Run the offline Generate → Verify showcase comparing accurate vs hallucinated output:

```bash
# Full showcase (accurate vs flawed comparison):
python examples/generate_demo.py

# Good scenario (accurate signature, 100/100 Trust Score):
python examples/generate_demo.py --good

# Flawed scenario (hallucinated parameter, verifier flags contradictions):
python examples/generate_demo.py --flawed

# Grounded README generation:
python examples/generate_demo.py --readme
```

## Web Interface (Part 7)

TrustDoc AI includes a clean, polished Streamlit UI exposing all core workflows:
* **📁 Project & Source Management**: Create projects, upload `.py`, `.md`, and `.pdf` files, inspect indexed code entities and chunks, and manage project data safely.
* **💬 Ask**: Interactive question answering strictly grounded in retrieved project evidence with validated citation anchors, evidence inspection drawers, and honest abstention when evidence is insufficient.
* **📝 Generate**: Grounded README and Python API documentation generation with AST entity selection, Markdown preview, one-click file download (`st.download_button`), and automated AST verification integration.
* **🔍 Verify**: Verify existing indexed documentation or custom Markdown against code AST, displaying the transparent Trust Score breakdown, 4-quadrant claim outcomes, severity-grouped remediation suggestions, and prominent heuristic disclaimers.

### Installation & Launch

Install with the `ui` extra:
```bash
pip install -e ".[ui]"
# or install everything:
pip install -e ".[all]"
```

Launch the interface via the CLI:
```bash
python -m trustdoc ui
# or with custom port/host:
python -m trustdoc ui --port 8501 --host 127.0.0.1
```

Or run Streamlit directly:
```bash
streamlit run trustdoc/ui/app.py
```

## Tests

```bash
pip install -e ".[dev,pdf,retrieval]"
pytest                                    # everything
pytest tests/test_retrieval_fusion.py tests/test_retrieval_threshold.py tests/test_faiss_index.py \
       tests/test_bm25_index.py tests/test_index_manager.py tests/test_retriever.py \
       tests/test_retrieval_integration.py tests/test_evaluation.py     # Part 3 core
pytest tests/test_ask_prompts.py tests/test_ask_citations.py tests/test_ask_workflow.py \
       tests/test_llm_factory_config.py tests/test_ollama_backend.py tests/test_transformers_backend.py \
       tests/test_stub_llm.py tests/test_ask_cli.py tests/test_ask_integration.py    # Part 4
pytest tests/test_verify_*.py tests/test_schema_v2_migration.py                  # Part 5
pytest tests/test_generation_*.py                                               # Part 6
pytest tests/test_ui.py                                                         # Part 7 (Streamlit UI)
pytest --cov=trustdoc                     # with coverage
```

Tests never download models: embeddings and reranking use deterministic stand-ins (`HashingEmbedder`,
`OverlapReranker`) or scripted rerankers, and the Sentence-Transformers wrapper code is tested against a
fake `sentence_transformers` module. Part 4 and Part 6 tests use mocked retrievers and LLMs, a local HTTP server mimicking Ollama, and fake `torch`/`transformers` modules, so no LLM is ever downloaded. With a lightweight install the tests that need numpy / FAISS /
rank_bm25 / PyMuPDF are skipped.

## Roadmap

1. Foundation & data model - done
2. Source & document ingestion - done
3. Knowledge base & hybrid retrieval - done
4. RAG Ask + grounded generation - done
5. Documentation verification + Trust Score - done
6. Grounded Documentation Generation (GENERATE -> VERIFY) - done
7. Streamlit Web UI - done (this part)
8. Final evaluation & packaging - next
