# TrustDoc AI

**Hybrid RAG-based software documentation generation, verification and hallucination detection.**

> Every documentation claim should be supported by evidence.

This repository is built in 8 sequential parts. **Current status: Part 2 - Source & Document Ingestion**
(on top of the Part 1 foundation). No retrieval, embeddings, LLM calls, verification logic or UI exist yet.

## Setup

Requires Python 3.10+. The core is standard-library only; PDF ingestion needs PyMuPDF, which is an
**optional extra** (`pdf`), not a mandatory dependency.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,pdf]"          # or: pip install -r requirements.txt
                                     # (omit ",pdf" to skip PyMuPDF; PDF tests are then skipped)

cp .env.example .env                 # optional; defaults work as-is
python -m trustdoc init              # creates data/ and the SQLite database
python -m trustdoc info              # shows settings, schema version, row counts
pytest                               # run the whole test suite
```

Optional extras: `.[pdf]` (PyMuPDF, used now), `.[retrieval]` (sentence-transformers, faiss-cpu,
rank_bm25 - later parts), `.[ui]` (Streamlit - later part), `.[all]`. Without `pdf`, Python and
Markdown ingestion work normally and ingesting a PDF raises a clear `MissingDependencyError`.

## Configuration

`Settings.from_env()` layers: defaults < `.env` < real environment variables. All keys are
prefixed `TRUSTDOC_` (see `.env.example`). Invalid values raise `ConfigError` at startup.
Sections: `storage`, `logging`, `chunking`, `retrieval`, `llm`. The LLM section stores only the
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

## Tests

```bash
pip install -e ".[dev,pdf]"
pytest                                   # everything (138 tests)
pytest tests/test_python_analyzer.py tests/test_chunk_builders.py tests/test_chunking_text.py \
       tests/test_pdf.py tests/test_storage.py tests/test_ingestion_service.py   # Part 2 only
pytest --cov=trustdoc                    # with coverage
```

PDF tests generate their PDFs with PyMuPDF and are skipped if it is not installed.

## Roadmap

1. Foundation & data model - done
2. Source & document ingestion - done (this part)
3-8. Retrieval, ASK, GENERATE, VERIFY (Tier 1 / Tier 2, Trust Score), UI, evaluation - one part at a time.
