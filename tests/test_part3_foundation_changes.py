import pytest

from trustdoc.config import Settings
from trustdoc.exceptions import ConfigError
from trustdoc.models import Chunk, ChunkType, Project, Source, SourceType


def test_retrieval_settings_defaults_and_overrides(tmp_path):
    base = {"TRUSTDOC_DATA_DIR": str(tmp_path)}
    s = Settings.from_env(base)
    assert s.retrieval.device == "cpu" and s.retrieval.embedding_batch_size == 32
    assert s.retrieval.relevance_threshold == 0.0           # documented placeholder
    assert s.retrieval.embedding_model.endswith("all-MiniLM-L6-v2")
    assert s.retrieval.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    s2 = Settings.from_env({**base, "TRUSTDOC_DEVICE": "CUDA", "TRUSTDOC_EMBEDDING_BATCH_SIZE": "8",
                            "TRUSTDOC_RELEVANCE_THRESHOLD": "0.35"})
    assert (s2.retrieval.device, s2.retrieval.embedding_batch_size, s2.retrieval.relevance_threshold) == ("cuda", 8, 0.35)
    with pytest.raises(ConfigError):
        Settings.from_env({**base, "TRUSTDOC_EMBEDDING_BATCH_SIZE": "0"})


def test_indexes_dir_is_derived_and_created(tmp_path):
    s = Settings.from_env({"TRUSTDOC_DATA_DIR": str(tmp_path / "d")})
    assert s.storage.indexes_dir == (tmp_path / "d" / "indexes").resolve()
    s.ensure_directories()
    assert s.storage.indexes_dir.is_dir()


def test_chunk_signatures_are_cheap_and_project_scoped(db):
    with db.session() as s:
        p1, p2 = s.projects.create(Project(name="a")), s.projects.create(Project(name="b"))
        s1 = s.sources.add(Source(p1.id, SourceType.PYTHON, "a.py", "h1"))
        s2 = s.sources.add(Source(p2.id, SourceType.PYTHON, "a.py", "h2"))
        c1 = s.chunks.add(Chunk(p1.id, s1.id, 0, ChunkType.CODE, "abcd"))
        s.chunks.add(Chunk(p2.id, s2.id, 0, ChunkType.CODE, "x"))
        assert s.chunks.list_signatures(p1.id) == [(c1.id, s1.id, 4)]
        assert s.chunks.list_signatures(99999) == []
