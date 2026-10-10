"""A core install (no numpy/faiss/rank_bm25/torch) must still import the pure retrieval modules."""

import subprocess
import sys
import textwrap

CODE = textwrap.dedent("""
    import sys
    for heavy in ("numpy", "faiss", "rank_bm25", "sentence_transformers", "torch"):
        sys.modules[heavy] = None            # makes `import <heavy>` raise ImportError
    import trustdoc.retrieval as r
    assert r.reciprocal_rank_fusion([]) == []
    assert r.tokenize("load_config")[0] == "load_config"
    assert r.apply_relevance_threshold([1.0], 0.5, lambda x: x).abstain is False
    try:
        r.Retriever
    except ImportError:
        print("lazy-heavy-ok")
""")


def test_pure_modules_import_without_heavy_dependencies():
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "lazy-heavy-ok" in out.stdout
