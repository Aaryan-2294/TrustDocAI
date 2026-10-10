from trustdoc.models import Chunk, ChunkType
from trustdoc.retrieval import chunk_index_text, tokenize


def test_identifiers_keep_whole_and_part_tokens():
    assert tokenize("load_config") == ["load_config", "load", "config"]
    assert tokenize("WorkerPool") == ["workerpool", "worker", "pool"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]


def test_dotted_names_and_punctuation_split():
    assert tokenize("pkg.mod.Store.put(key)") == ["pkg", "mod", "store", "put", "key"]
    assert tokenize("  ") == [] and tokenize("___") == []


def test_query_and_document_tokenisation_agree():
    assert tokenize("Visibility_Timeout") == tokenize("visibility_timeout")
    assert set(tokenize("load config")) <= set(tokenize("def load_config(path):"))


def test_chunk_index_text_prepends_logical_header_only():
    c = Chunk(1, 1, 0, ChunkType.CODE, "def f(): ...", metadata={"context_header": "pkg/a.py :: pkg.a.f"})
    assert chunk_index_text(c) == "pkg/a.py :: pkg.a.f\ndef f(): ..."
    assert chunk_index_text(Chunk(1, 1, 0, ChunkType.CODE, "x = 1")) == "x = 1"
