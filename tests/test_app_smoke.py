"""App-level smoke eval for the Streamlit resume agent.

Locks in the current access behaviour and the properties we broke once already:
- the landing page is an access request form, not a chat box
- a valid email unlocks the chat
- recruiter tokens and owner links skip the form
- no email address or domain leaks into rendered output
- MAX_QUERIES defaults to 5

Run: cd ~/projects/resume-agent && .venv/bin/python -m pytest tests/test_app_smoke.py -q
"""
import os
import re
import sys
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run(query_params=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in (query_params or {}).items():
        at.query_params[k] = v
    at.run()
    return at


def _rendered_text(at):
    blobs = []
    for kind in ("markdown", "caption", "text", "title", "header", "subheader"):
        for el in getattr(at, kind, []) or []:
            try:
                blobs.append(str(el.value))
            except Exception:
                pass
    return " ".join(blobs)


def test_landing_shows_request_form_not_chat():
    at = _run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.chat_input) == 0
    assert len(at.text_input) >= 1
    assert len(at.text_area) >= 1


def test_invalid_email_is_rejected():
    at = _run()
    at.text_input[0].set_value("not-an-email")
    at.button[0].click().run()
    assert len(at.chat_input) == 0
    assert any("valid email" in e.value.lower() for e in at.error)


def test_valid_email_unlocks_chat():
    at = _run()
    at.text_input[0].set_value("someone@example.com")
    at.text_area[0].set_value("Senior AI Engineer. Python, RAG, evaluation.")
    at.button[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.chat_input) == 1


def test_recruiter_token_skips_form():
    at = _run({"j": "7f3a2b"})
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.chat_input) == 1
    assert len(at.text_input) == 0


def test_owner_link_skips_form():
    at = _run({"owner": "true"})
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.chat_input) == 1


def test_no_contact_details_in_output():
    for qp in ({}, {"owner": "true"}):
        at = _run(qp)
        rendered = _rendered_text(at)
        assert "@" not in rendered, "rendered output contains an email address"
        assert "gmail" not in rendered.lower()


def test_query_cap_defaults_to_five():
    src = Path(APP).read_text()
    m = re.search(r'MAX_QUERIES = int\(os\.environ\.get\("MAX_QUERIES", "(\d+)"\)\)', src)
    assert m, "MAX_QUERIES default not found"
    assert m.group(1) == "5"

def test_query_cap_is_per_account():
    """A used-up account stays locked even in a brand new browser session."""
    import json
    from pathlib import Path as _P
    store = _P.home() / "Library" / "Logs" / "resume-agent" / "query-counts.json"
    backup = store.read_text() if store.exists() else None
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps({"email:used@example.com": 5}))
        at = _run()
        at.text_input[0].set_value("used@example.com")
        at.button[0].click().run()
        assert len(at.chat_input) == 1
        assert getattr(at.chat_input[0], "disabled", False) is True
        assert any("used all" in i.value.lower() for i in at.info)
    finally:
        store.write_text(backup if backup is not None else "{}")
