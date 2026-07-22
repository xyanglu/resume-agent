"""
Resume Agent Analytics Tracker
Logs events to a Google Sheet you own (shared with service account for writing).
Fallback: logs to local JSONL file if Sheets API unavailable.
"""

import json
import os
import uuid
import hashlib
import threading
from datetime import datetime, timezone

# --- Config ---
SHEET_ID = os.getenv("ANALYTICS_SHEET_ID", "")
SHEET_NAME = "Events"
TRACKING_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").lower() == "true"
LOCAL_LOG = os.getenv("ANALYTICS_LOCAL_LOG", "")


def get_session_id(st):
    sid = st.session_state.get("analytics_session_id")
    if not sid:
        sid = str(uuid.uuid4())[:8]
        st.session_state["analytics_session_id"] = sid
        st.session_state["analytics_session_start"] = datetime.now(timezone.utc).isoformat()
    return sid


def _get_user_info(st):
    user_agent = ""
    ip_hash = ""
    try:
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if ctx and hasattr(ctx, "request"):
            headers = ctx.request.headers or {}
            user_agent = headers.get("user-agent", "")
            ip = headers.get("x-forwarded-for", headers.get("x-real-ip", ""))
            if ip:
                ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:12]
    except Exception:
        pass
    return user_agent[:200], ip_hash


def _get_referrer(st):
    """Resolve the obscured ref code to a human-readable label via REFERRAL_CODES."""
    try:
        code = st.query_params.get("ref", st.query_params.get("utm_source", ""))
        # Try to resolve via the mapping in app.py
        try:
            from app import REFERRAL_CODES
            return REFERRAL_CODES.get(code, code or "direct")
        except Exception:
            return code or "direct"
    except Exception:
        return "direct"


def _get_duration(st):
    try:
        started = datetime.fromisoformat(st.session_state["analytics_session_start"])
        return f"{(datetime.now(timezone.utc) - started).total_seconds():.0f}s"
    except Exception:
        return ""


def _write_local(row):
    log_path = LOCAL_LOG or "/tmp/resume_agent_analytics.jsonl"
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass


def _write_sheet(rows, st):
    if not SHEET_ID:
        return False
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        import tempfile

        sa_json = None
        sa_path = os.getenv("service_account_path")
        try:
            sa_json = st.secrets.get("service_account_json")
        except Exception:
            pass

        if sa_path:
            creds = service_account.Credentials.from_service_account_file(
                sa_path, scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
        elif sa_json:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(sa_json if isinstance(sa_json, str) else json.dumps(sa_json))
                tmp = f.name
            creds = service_account.Credentials.from_service_account_file(
                tmp, scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            os.unlink(tmp)
        else:
            return False

        service = build("sheets", "v4", credentials=creds)
        service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range=f"{SHEET_NAME}!A:A",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": rows},
        ).execute()
        return True
    except Exception:
        return False


def track(st, event_type, event_data=""):
    if not TRACKING_ENABLED:
        return
    ua, ip = _get_user_info(st)
    dur = _get_duration(st)
    row = [
        datetime.now(timezone.utc).isoformat(),
        get_session_id(st),
        f"{event_type} ({dur})" if dur else event_type,
        str(event_data)[:500] if event_data else "",
        ua,
        ip,
        "",
        _get_referrer(st),
    ]
    def _do():
        if not _write_sheet([row], st):
            _write_local(row)
    threading.Thread(target=_do, daemon=True).start()


# Convenience functions
def track_page_view(st):
    track(st, "page_view")

def track_chat_query(st, text):
    """Log the full question text — this is the key data Yang wants to see."""
    track(st, "chat_query", text.replace("\n", " ") if text else "")


def track_chat_response(st, length, verification_status=""):
    """Log response length and verification outcome."""
    detail = f"len={length}"
    if verification_status:
        detail += f"|verified={verification_status}"
    track(st, "chat_response", detail)

def track_pdf_generate(st, doc_type, company=""):
    track(st, "pdf_generate", f"{doc_type}|company={company}")

def track_vision_review(st, doc_type, rating=""):
    track(st, "vision_review", f"{doc_type}|rating={rating}")

def track_resume_load(st, chars=0):
    track(st, "resume_load", f"chars={chars}")

def track_error(st, etype, msg=""):
    track(st, "error", f"{etype}|{str(msg)[:100]}")

def track_button(st, name):
    track(st, "button_click", name)

def track_init(st, ok=True, msg=""):
    track(st, "init_success" if ok else "init_failure", str(msg)[:200])
