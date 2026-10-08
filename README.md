# TrustDoc AI

**Hybrid RAG-based software documentation generation, verification and hallucination detection.**

> Every documentation claim should be supported by evidence.

This repository is built in 8 sequential parts. **Current status: Part 1 - Project Foundation & Data Model.**
No retrieval, embeddings, LLM calls, verification logic or UI exist yet; this part provides the
shared structure everything else plugs into.

## Setup

Requires Python 3.10+. Part 1 uses only the standard library (plus pytest for tests).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # or: pip install -r requirements.txt

cp .env.example .env                 # optional; defaults work as-is
python -m trustdoc init              # creates data/ and the SQLite database
python -m trustdoc info              # shows settings, schema version, row counts
pytest                               # run the foundation tests
```

Optional extras are installed in later parts: `.[pdf]` (PyMuPDF), `.[retrieval]`
(sentence-transformers, faiss-cpu, rank_bm25), `.[ui]` (Streamlit), or `.[all]`.

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
tests/               pytest suite (config, logging, models, schema, repositories, CLI)
data/                runtime data (gitignored): trustdoc.db, uploads/, logs/
```

## Architecture (current)

```
cli / (future Streamlit UI)
        |
     app.bootstrap() --> Settings, logging, Database
        |
   Database.session()  -- one transaction per `with` block (commit / rollback)
        |
   Session: projects | sources | entities | chunks | claims | runs | findings   (repositories)
        |
   SQLite (foreign keys ON, WAL, schema versioned with PRAGMA user_version)
```

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

## Tests

`pytest` runs 37 tests (unit + SQLite integration). Add `--cov=trustdoc` for coverage.

## Roadmap

1. Foundation & data model (this part)
2-8. Ingestion, retrieval, ASK, GENERATE, VERIFY (Tier 1 / Tier 2, Trust Score), UI, evaluation - each delivered one part at a time.
