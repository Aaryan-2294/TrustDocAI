"""Generate grounded documentation and verify it against the bundled sample project.

Works completely offline in a temporary workspace with zero model downloads.

Usage:
    # Run the full Generate -> Verify showcase (good vs flawed comparison):
    python examples/generate_demo.py

    # Show only the accurate generation scenario (high Trust Score):
    python examples/generate_demo.py --good

    # Show only the hallucinated/flawed scenario (verifier catches contradictions):
    python examples/generate_demo.py --flawed

    # Generate a grounded README using the extractive stand-in:
    python examples/generate_demo.py --readme

    # Test with your locally configured Ollama model:
    python examples/generate_demo.py --llm configured --entity compute_backoff
"""

from __future__ import annotations

import argparse
import os
import tempfile

from trustdoc.config import Settings
from trustdoc.db import Database
from trustdoc.evaluation.dataset import DEFAULT_QUERIES, load_dataset
from trustdoc.ingestion import IngestionService
from trustdoc.llm.base import GenerationResult, LLMBackend
from trustdoc.llm.stub import ExtractiveStubLLM
from trustdoc.models import Project
from trustdoc.retrieval import HashingEmbedder, OverlapReranker, create_retriever
from trustdoc.workflows import (
    GenerationService,
    render_generation_result,
)

ACCURATE_API_DOC = """### `compute_backoff`
`def compute_backoff(attempt: int, base_seconds: float = 2.0, jitter: bool = True) -> float:`

#### Description
The `compute_backoff` function accepts `attempt`, `base_seconds` and `jitter`, and returns a float. [E1]

#### Parameters
- `attempt` (int): Current retry attempt [E1]
- `base_seconds` (float, default: 2.0): Base retry delay [E1]
- `jitter` (bool, default: True): Whether to randomize delay with jitter [E1]

#### Returns
(float): Calculated backoff delay in seconds [E1]
"""

FLAWED_API_DOC = """### `compute_backoff`
`def compute_backoff(attempt: int, base_delay: int = 5) -> int:`

#### Description
The `compute_backoff` function accepts `attempt` and `base_delay`, and returns an integer. [E1]

Call `retry_all()` to requeue everything. [E1]

#### Parameters
- `attempt`: Current attempt [E1]
- `base_delay`: Fabricated delay parameter [E1]

#### Returns
(int): Returns an integer delay [E1]
"""


class ScriptedDemoLLM(LLMBackend):
    name = "scripted-demo"
    is_stub = False

    def __init__(self, canned_text: str) -> None:
        super().__init__("scripted-demo-model")
        self._canned = canned_text

    def generate(self, prompt, system_prompt=None, *, max_tokens=None, temperature=None):
        return GenerationResult(
            text=self._canned,
            model=self.model_name,
            backend=self.name,
            finish_reason="stop",
        )


def run_scenario(title: str, service: GenerationService, project_id: int, kind: str, entity: str | None = None) -> None:
    print(f"\n{'=' * 78}")
    print(f"SCENARIO: {title}")
    print(f"{'=' * 78}")
    res = service.generate(project_id, kind, entity=entity)
    print(render_generation_result(res, show_evidence=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="TrustDoc AI Part 6 Generate -> Verify Demo")
    ap.add_argument("--good", action="store_true", help="run accurate documentation generation only")
    ap.add_argument("--flawed", action="store_true", help="run flawed/hallucinated generation only")
    ap.add_argument("--readme", action="store_true", help="run README generation only")
    ap.add_argument("--entity", default="compute_backoff", help="entity to document (default: compute_backoff)")
    ap.add_argument("--llm", choices=["stub", "configured"], default="stub",
                    help="use stand-in/scripted LLM or your configured local model (e.g. Ollama)")
    ap.add_argument("--real-retrieval", action="store_true", help="use real Sentence-Transformers/Cross-Encoder")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        env = {
            "TRUSTDOC_DATA_DIR": tmp,
            "TRUSTDOC_LOG_TO_FILE": "false",
            "TRUSTDOC_LOG_LEVEL": "WARNING",
        }
        if args.llm == "configured":
            for k in os.environ:
                if k.startswith(("TRUSTDOC_LLM_", "TRUSTDOC_GENERATE_")):
                    env[k] = os.environ[k]

        settings = Settings.from_env(env)
        settings.ensure_directories()
        db = Database(settings.storage.db_path)
        db.initialize()

        with db.session() as s:
            project_id = s.projects.create(Project(name="sample")).id

        print("Ingesting sample codebase into temporary project...")
        batch = IngestionService(db, settings).ingest_directory(
            project_id, load_dataset(DEFAULT_QUERIES).project_dir
        )
        print(f"Ingested {len(batch.results)} files into project #{project_id}.\n")

        if args.real_retrieval:
            retriever = create_retriever(db, settings)
        else:
            retriever = create_retriever(db, settings, embedder=HashingEmbedder(), reranker=OverlapReranker())

        if args.llm == "configured":
            service = GenerationService(db, settings, retriever, llm=None)
            kind = "readme" if args.readme else "api"
            run_scenario("Configured Local LLM Generation", service, project_id, kind, args.entity)
            return

        if args.good:
            good_service = GenerationService(db, settings, retriever, llm=ScriptedDemoLLM(ACCURATE_API_DOC))
            run_scenario("Accurate Grounded API Generation (High Trust Score)", good_service, project_id, "api", args.entity)
            return

        if args.flawed:
            flawed_service = GenerationService(db, settings, retriever, llm=ScriptedDemoLLM(FLAWED_API_DOC))
            run_scenario("Hallucinated / Flawed API Generation (Verifier Flags Contradictions)", flawed_service, project_id, "api", args.entity)
            return

        if args.readme:
            readme_service = GenerationService(db, settings, retriever, llm=ExtractiveStubLLM())
            run_scenario("Grounded Project README Generation", readme_service, project_id, "readme")
            return

        # Default: Full showcase showing both accurate and flawed scenarios
        print("Running full Generate -> Verify showcase:")
        good_service = GenerationService(db, settings, retriever, llm=ScriptedDemoLLM(ACCURATE_API_DOC))
        run_scenario("Case 1: Grounded & Accurate API Generation (Expect Trust Score: 100/100)",
                     good_service, project_id, "api", args.entity)

        flawed_service = GenerationService(db, settings, retriever, llm=ScriptedDemoLLM(FLAWED_API_DOC))
        run_scenario("Case 2: Flawed / Hallucinated API Generation (Verifier detects contradictions & drops score)",
                     flawed_service, project_id, "api", args.entity)

        stub_service = GenerationService(db, settings, retriever, llm=ExtractiveStubLLM())
        run_scenario("Case 3: Grounded README Generation (Using extractive stand-in)",
                     stub_service, project_id, "readme")


if __name__ == "__main__":
    main()
