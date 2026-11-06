import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_location_states_relocation_target():
    data = json.loads((REPO / "resume.json").read_text())
    loc = data["contact"]["location"]
    assert "Toronto" in loc
    assert "New York" in loc
    assert "Greater Toronto Area" not in loc


def test_local_resume_wins_over_remote():
    src = (REPO / "streamlit_app.py").read_text()
    local_at = src.index("for p in [RESUME_JSON_PATH, LOCAL_RESUME_JSON]")
    remote_at = src.index("urlopen(GITHUB_RESUME_URL")
    assert local_at < remote_at, "remote resume is loaded before the local file"
