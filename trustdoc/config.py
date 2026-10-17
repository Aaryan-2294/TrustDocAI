"""Configuration management.

Settings are immutable dataclasses built from (in increasing priority):
built-in defaults < ``.env`` file < real environment variables.

Sections for later parts (chunking, retrieval, LLM) are declared here so that
every module reads configuration from one place. Part 1 only *consumes* the
storage and logging sections.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from trustdoc.exceptions import ConfigError

ENV_PREFIX = "TRUSTDOC_"
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path
    db_path: Path
    uploads_dir: Path
    log_dir: Path
    indexes_dir: Path


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    to_file: bool = True
    file_name: str = "trustdoc.log"
    max_bytes: int = 1_000_000
    backup_count: int = 3


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1200
    overlap_chars: int = 150


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 20          # candidates taken from each retriever
    final_k: int = 5               # evidence items kept after reranking
    rrf_k: int = 60                # RRF constant
    relevance_threshold: float = 0.0  # UNCALIBRATED placeholder; calibrate experimentally
    device: str = "cpu"            # "cpu" (default), "cuda", "mps" ... passed to sentence-transformers
    embedding_batch_size: int = 32


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "none"
    model: str = ""
    base_url: str | None = None
    api_key_env: str = "TRUSTDOC_LLM_API_KEY"  # NAME of the env var, never the key
    temperature: float = 0.0
    timeout_seconds: int = 120     # CPU inference is slow; raised from 60 in Part 4
    max_tokens: int = 512          # maximum generated tokens per answer
    context_tokens: int = 4096     # context window requested from / assumed for the model


@dataclass(frozen=True)
class AskConfig:
    """Limits for the Ask workflow (Part 4). Sized to fit a ~4k-token context."""

    max_evidence_items: int = 5       # evidence chunks placed in the prompt (also the retrieval top_k)
    max_evidence_chars: int = 6000    # total characters of evidence text in the prompt
    max_question_chars: int = 2000    # longer questions are rejected, never truncated


@dataclass(frozen=True)
class VerificationConfig:
    """Documentation verification and Trust Score parameters (Part 5)."""

    weight_accuracy: float = 0.50       # relative weights; unavailable components are dropped and the
    weight_coverage: float = 0.25       # remaining weights renormalised (see verification/scoring.py)
    weight_grounding: float = 0.25
    penalty_critical: float = 10.0      # points subtracted per CRITICAL contradiction finding
    penalty_high: float = 5.0           # points subtracted per HIGH contradiction finding
    penalty_cap: float = 40.0           # maximum total severity penalty
    evidence_top_k: int = 5             # evidence chunks shown to the semantic verifier per claim
    evidence_source_types: tuple[str, ...] = ("python", "pdf")   # what may serve as evidence
    max_semantic_claims: int = 25       # cap on LLM-checked claims per run (CPU cost)


@dataclass(frozen=True)
class Settings:
    storage: StorageConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ask: AskConfig = field(default_factory=AskConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        dotenv_path: str | Path | None = None,
    ) -> "Settings":
        """Build settings.

        ``env=None`` uses ``os.environ`` layered over ``./.env`` (if present).
        An explicit ``env`` mapping is used as-is (handy for tests); a
        ``.env`` file is only read when ``dotenv_path`` is given.
        """
        if env is None:
            merged = dict(_read_dotenv(Path(dotenv_path or ".env")))
            merged.update(os.environ)
        else:
            merged = dict(_read_dotenv(Path(dotenv_path))) if dotenv_path else {}
            merged.update(env)
        return cls._from_mapping(merged)

    @classmethod
    def _from_mapping(cls, env: Mapping[str, str]) -> "Settings":
        def get(name: str, default: Any = None) -> Any:
            raw = env.get(ENV_PREFIX + name)
            return default if raw is None or raw == "" else raw

        data_dir = Path(get("DATA_DIR", "data")).expanduser().resolve()
        db_path = Path(get("DB_PATH", data_dir / "trustdoc.db")).expanduser().resolve()
        storage = StorageConfig(
            data_dir=data_dir,
            db_path=db_path,
            uploads_dir=data_dir / "uploads",
            log_dir=data_dir / "logs",
            indexes_dir=data_dir / "indexes",
        )

        log_defaults = LoggingConfig()
        logging_cfg = LoggingConfig(
            level=str(get("LOG_LEVEL", log_defaults.level)).upper(),
            to_file=_as_bool(get("LOG_TO_FILE", log_defaults.to_file), "LOG_TO_FILE"),
        )

        c = ChunkingConfig()
        chunking = ChunkingConfig(
            max_chars=_as_int(get("CHUNK_MAX_CHARS", c.max_chars), "CHUNK_MAX_CHARS"),
            overlap_chars=_as_int(
                get("CHUNK_OVERLAP_CHARS", c.overlap_chars), "CHUNK_OVERLAP_CHARS"
            ),
        )

        r = RetrievalConfig()
        retrieval = RetrievalConfig(
            embedding_model=str(get("EMBEDDING_MODEL", r.embedding_model)),
            reranker_model=str(get("RERANKER_MODEL", r.reranker_model)),
            candidate_k=_as_int(get("CANDIDATE_K", r.candidate_k), "CANDIDATE_K"),
            final_k=_as_int(get("FINAL_K", r.final_k), "FINAL_K"),
            rrf_k=_as_int(get("RRF_K", r.rrf_k), "RRF_K"),
            relevance_threshold=_as_float(
                get("RELEVANCE_THRESHOLD", r.relevance_threshold), "RELEVANCE_THRESHOLD"
            ),
            device=str(get("DEVICE", r.device)).lower(),
            embedding_batch_size=_as_int(
                get("EMBEDDING_BATCH_SIZE", r.embedding_batch_size), "EMBEDDING_BATCH_SIZE"
            ),
        )

        l = LLMConfig()
        llm = LLMConfig(
            provider=str(get("LLM_PROVIDER", l.provider)).lower(),
            model=str(get("LLM_MODEL", l.model)),
            base_url=get("LLM_BASE_URL", l.base_url),
            api_key_env=str(get("LLM_API_KEY_ENV", l.api_key_env)),
            temperature=_as_float(get("LLM_TEMPERATURE", l.temperature), "LLM_TEMPERATURE"),
            timeout_seconds=_as_int(
                get("LLM_TIMEOUT_SECONDS", l.timeout_seconds), "LLM_TIMEOUT_SECONDS"
            ),
            max_tokens=_as_int(get("LLM_MAX_TOKENS", l.max_tokens), "LLM_MAX_TOKENS"),
            context_tokens=_as_int(get("LLM_CONTEXT_TOKENS", l.context_tokens), "LLM_CONTEXT_TOKENS"),
        )

        a = AskConfig()
        ask = AskConfig(
            max_evidence_items=_as_int(
                get("ASK_MAX_EVIDENCE_ITEMS", a.max_evidence_items), "ASK_MAX_EVIDENCE_ITEMS"),
            max_evidence_chars=_as_int(
                get("ASK_MAX_EVIDENCE_CHARS", a.max_evidence_chars), "ASK_MAX_EVIDENCE_CHARS"),
            max_question_chars=_as_int(
                get("ASK_MAX_QUESTION_CHARS", a.max_question_chars), "ASK_MAX_QUESTION_CHARS"),
        )

        v = VerificationConfig()
        types = tuple(
            x.strip().lower() for x in str(get("VERIFY_EVIDENCE_TYPES", ",".join(v.evidence_source_types))).split(",")
            if x.strip()
        )
        verification = VerificationConfig(
            weight_accuracy=_as_float(get("VERIFY_WEIGHT_ACCURACY", v.weight_accuracy), "VERIFY_WEIGHT_ACCURACY"),
            weight_coverage=_as_float(get("VERIFY_WEIGHT_COVERAGE", v.weight_coverage), "VERIFY_WEIGHT_COVERAGE"),
            weight_grounding=_as_float(get("VERIFY_WEIGHT_GROUNDING", v.weight_grounding), "VERIFY_WEIGHT_GROUNDING"),
            penalty_critical=_as_float(get("VERIFY_PENALTY_CRITICAL", v.penalty_critical), "VERIFY_PENALTY_CRITICAL"),
            penalty_high=_as_float(get("VERIFY_PENALTY_HIGH", v.penalty_high), "VERIFY_PENALTY_HIGH"),
            penalty_cap=_as_float(get("VERIFY_PENALTY_CAP", v.penalty_cap), "VERIFY_PENALTY_CAP"),
            evidence_top_k=_as_int(get("VERIFY_EVIDENCE_TOP_K", v.evidence_top_k), "VERIFY_EVIDENCE_TOP_K"),
            evidence_source_types=types,
            max_semantic_claims=_as_int(get("VERIFY_MAX_SEMANTIC_CLAIMS", v.max_semantic_claims), "VERIFY_MAX_SEMANTIC_CLAIMS"),
        )

        settings = cls(storage, logging_cfg, chunking, retrieval, llm, ask, verification)
        settings.validate()
        return settings

    # --------------------------------------------------------------- behaviour
    def validate(self) -> None:
        if self.logging.level not in _VALID_LOG_LEVELS:
            raise ConfigError(
                f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}, got {self.logging.level!r}"
            )
        if self.chunking.max_chars <= 0:
            raise ConfigError("CHUNK_MAX_CHARS must be positive")
        if not 0 <= self.chunking.overlap_chars < self.chunking.max_chars:
            raise ConfigError("CHUNK_OVERLAP_CHARS must be >= 0 and < CHUNK_MAX_CHARS")
        rt = self.retrieval
        if rt.candidate_k <= 0 or rt.final_k <= 0 or rt.rrf_k <= 0:
            raise ConfigError("CANDIDATE_K, FINAL_K and RRF_K must be positive")
        if rt.final_k > rt.candidate_k:
            raise ConfigError("FINAL_K must not exceed CANDIDATE_K")
        if rt.embedding_batch_size <= 0:
            raise ConfigError("EMBEDDING_BATCH_SIZE must be positive")
        if not rt.device:
            raise ConfigError("DEVICE must not be empty")
        if not 0.0 <= self.llm.temperature <= 2.0:
            raise ConfigError("LLM_TEMPERATURE must be between 0 and 2")
        if self.llm.timeout_seconds <= 0:
            raise ConfigError("LLM_TIMEOUT_SECONDS must be positive")
        if self.llm.max_tokens <= 0:
            raise ConfigError("LLM_MAX_TOKENS must be positive")
        if self.llm.context_tokens <= self.llm.max_tokens:
            raise ConfigError("LLM_CONTEXT_TOKENS must be larger than LLM_MAX_TOKENS")
        ask = self.ask
        if ask.max_evidence_items <= 0:
            raise ConfigError("ASK_MAX_EVIDENCE_ITEMS must be positive")
        if ask.max_evidence_chars < 200:
            raise ConfigError("ASK_MAX_EVIDENCE_CHARS must be at least 200")
        if ask.max_question_chars <= 0:
            raise ConfigError("ASK_MAX_QUESTION_CHARS must be positive")
        v = self.verification
        weights = (v.weight_accuracy, v.weight_coverage, v.weight_grounding)
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ConfigError("VERIFY_WEIGHT_* must be non-negative and not all zero")
        if min(v.penalty_critical, v.penalty_high, v.penalty_cap) < 0:
            raise ConfigError("VERIFY_PENALTY_* must be non-negative")
        if v.evidence_top_k <= 0:
            raise ConfigError("VERIFY_EVIDENCE_TOP_K must be positive")
        if v.max_semantic_claims < 0:
            raise ConfigError("VERIFY_MAX_SEMANTIC_CLAIMS must not be negative")
        bad = [x for x in v.evidence_source_types if x not in ("python", "markdown", "pdf")]
        if bad or not v.evidence_source_types:
            raise ConfigError("VERIFY_EVIDENCE_TYPES must list one or more of: python, markdown, pdf")

    @property
    def log_level_int(self) -> int:
        return getattr(logging, self.logging.level)

    def ensure_directories(self) -> None:
        for path in (
            self.storage.data_dir,
            self.storage.uploads_dir,
            self.storage.log_dir,
            self.storage.indexes_dir,
            self.storage.db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (paths become strings). Contains no secrets."""

        def conv(v: Any) -> Any:
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v

        return conv(asdict(self))


# ------------------------------------------------------------------- helpers
def _read_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser (comments, blank lines, optional quotes)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{ENV_PREFIX}{name} must be a boolean, got {value!r}")


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {value!r}") from None


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{ENV_PREFIX}{name} must be a number, got {value!r}") from None
