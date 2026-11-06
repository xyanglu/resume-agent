"""Regression guards for the Nov 5 break.

The app lost its bundled ONNX embedder, so the first call went straight to
HuggingFaceEmbeddings, which needs sentence-transformers. The page rendered its
header and then died before the chat input, so every link looked broken. A guarded
fallback is fine; an unguarded one is the bug.
"""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text()


def test_bundled_onnx_embedder_exists():
    assert "_OnnxMiniLM" in SRC, "the self-contained embedder is gone"


def test_onnx_embedder_is_tried_first():
    onnx_at = SRC.index("return _OnnxMiniLM()")
    hf_at = SRC.index("HuggingFaceEmbeddings(")
    assert onnx_at < hf_at, "HuggingFaceEmbeddings is attempted before the ONNX embedder"


def test_huggingface_fallback_is_guarded():
    hf_at = SRC.index("HuggingFaceEmbeddings(model_name")
    guard_at = SRC.rindex("_EMBEDDINGS_AVAILABLE", 0, hf_at)
    assert hf_at - guard_at < 400, "the sentence-transformers fallback is not behind a guard"


def test_availability_flag_is_computed_defensively():
    flag_at = SRC.index("_EMBEDDINGS_AVAILABLE = True")
    try_at = SRC.rindex("try:", 0, flag_at)
    except_at = SRC.index("except ImportError:", flag_at)
    assert try_at < flag_at < except_at, "the import is not wrapped in try/except"
