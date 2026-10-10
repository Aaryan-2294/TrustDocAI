# TrustDoc AI

**Hybrid RAG-based software documentation generation, verification and hallucination detection.**

> Every documentation claim should be supported by evidence.

This repository is built in 8 sequential parts. **Current status: Part 3 - Knowledge Base & Hybrid Retrieval**
(on top of Part 1 foundation and Part 2 ingestion). No LLM calls, Ask/Generate/Verify workflows, Trust Score or UI exist yet.

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
`.[ui]` (Streamlit - later part), `.[all]`. The default install stays lightweight: importing
`trustdoc.retrieval` without the extra still gives you the pure modules (types, tokenizer, RRF,
threshold); anything that needs numpy/FAISS/torch raises a clear `MissingDependencyError`. Without `pdf`, Python and
Markdown ingestion work normally and ingesting a PDF raises a clear `MissingDependencyError`.

## Configuration

`Settings.from_env()` layers: defaults < `.env` < real environment variables. All keys are
prefixed `TRUSTDOC_` (see `.env.example`). Invalid values raise `ConfigError` at startup.
Sections: `storage`, `logging`, `chunking`, `retrieval`, `llm`. Retrieval keys (Part 3): `EMBEDDING_MODEL`, `RERANKER_MODEL`, `CANDIDATE_K`, `FINAL_K`, `RRF_K`, `RELEVANCE_THRESHOLD` (0.0 = uncalibrated placeholder), `DEVICE` (default `cpu`), `EMBEDDING_BATCH_SIZE`. Indexes are stored under `<data_dir>/indexes`. The LLM section stores only the
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
examples/            retrieval_demo.py
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

## Tests

```bash
pip install -e ".[dev,pdf,retrieval]"
pytest                                    # everything (239 tests with all extras)
pytest tests/test_retrieval_fusion.py tests/test_retrieval_threshold.py tests/test_faiss_index.py \
       tests/test_bm25_index.py tests/test_index_manager.py tests/test_retriever.py \
       tests/test_retrieval_integration.py tests/test_evaluation.py     # Part 3 core
pytest --cov=trustdoc                     # with coverage
```

Tests never download models: embeddings and reranking use deterministic stand-ins (`HashingEmbedder`,
`OverlapReranker`) or scripted rerankers, and the Sentence-Transformers wrapper code is tested against a
fake `sentence_transformers` module. With a lightweight install the tests that need numpy / FAISS /
rank_bm25 / PyMuPDF are skipped.

## Roadmap

1. Foundation & data model - done
2. Source & document ingestion - done
3. Knowledge base & hybrid retrieval - done (this part)
4-8. ASK, GENERATE, VERIFY (Tier 1 / Tier 2, Trust Score), UI, final evaluation - one part at a time.
