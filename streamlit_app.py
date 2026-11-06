"""
Resume Agent - RAG Chatbot with Custom Job Links
LangChain + Chroma + OpenRouter | Dark Themed UI | 5-Query Session Cap
"""

import streamlit as st
import os
import re
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="Yang's Resume Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stChatInput textarea { background-color: #1a1f2b !important; color: #fafafa !important; }
    .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
    .card h3 { margin-top: 0; }
    .welcome-banner { background: linear-gradient(135deg, #1a1f2b 0%, #1f2937 100%); border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
    .match-bar-bg { background-color: #21262d; border-radius: 8px; height: 14px; overflow: hidden; }
    .match-bar-fill { height: 14px; border-radius: 8px; background: linear-gradient(90deg, #238636, #3fb950); }
    .skill-chip { display: inline-block; background-color: #1f2937; border: 1px solid #30363d; border-radius: 999px; padding: 4px 12px; margin: 3px; font-size: 13px; color: #c9d1d9; }
    .skill-chip.matched { border-color: #238636; color: #3fb950; }
    .skill-chip.gap { border-color: #da3633; color: #f85149; }
    .footer { text-align: center; color: #484f58; font-size: 12px; padding: 24px 0 12px 0; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

MAX_QUERIES = int(os.environ.get("MAX_QUERIES", "5"))
RESUME_SOURCE_URL_ENV = "RESUME_URL"
OWNER_HASH = "d4c5627c"

from jd_skills import extract_jd_skills, skill_in_text


def parse_jd_text(raw_text):
    return {
        "title": "this role",
        "company": "your company",
        "description": raw_text,
        "skills": extract_jd_skills(raw_text),
    }

JOBS_JSON_PATH = Path(__file__).parent / "jobs.json"
RESUME_JSON_PATH = Path.home() / "Documents" / "interview-prep" / "resume.json"
LOCAL_RESUME_JSON = Path(__file__).parent / "resume.json"
GITHUB_RESUME_URL = "https://raw.githubusercontent.com/xyanglu/resume-agent/master/resume.json"

def load_resume_from_json(path=None):
    if path is not None:
        try:
            with open(path) as f:
                return _format_resume_text(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    # Local file first: the running app must reflect what is on disk. The GitHub copy is a
    # fallback for a fresh deploy and had been masking local edits (edits to resume.json did
    # not show up because the remote copy was served instead).
    for p in [RESUME_JSON_PATH, LOCAL_RESUME_JSON]:
        try:
            with open(p) as f:
                return _format_resume_text(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    try:
        import urllib.request
        resp = urllib.request.urlopen(GITHUB_RESUME_URL, timeout=10)
        data = json.loads(resp.read().decode())
        if data and data.get("name"):
            return _format_resume_text(data)
    except Exception:
        pass
    return None

def _format_resume_text(data):
    lines = []
    lines.append(f"# {data['name']}")
    c = data.get("contact", {})
    parts = [v for v in [c.get("location",""), c.get("linkedin",""), c.get("website",""), c.get("huggingface","")] if v]
    lines.append(" | ".join(parts))
    lines.append("")
    summary = data.get("summary","")
    if summary:
        lines.append("## Summary")
        lines.append(summary)
        lines.append("")
    lines.append("## Experience")
    for exp in data.get("experience",[]):
        lines.append(f"### {exp.get('title','')}")
        lines.append(f"**{exp.get('company','')}** | {exp.get('location','')} | {exp.get('dates','')}")
        for b in exp.get("bullets",[]):
            lines.append(f"- {b}")
        lines.append("")
    lines.append("## Projects")
    for proj in data.get("projects",[]):
        lines.append(f"### {proj.get('title','')}")
        if proj.get("subtitle"):
            lines.append(f"*{proj['subtitle']}*")
        for b in proj.get("bullets",[]):
            lines.append(f"- {b}")
        lines.append("")
    lines.append("## Education")
    for edu in data.get("education",[]):
        lines.append(f"- {edu.get('degree','')} - {edu.get('school','')} ({edu.get('dates','')})")
    lines.append("")
    lines.append("## Skills")
    for sg in data.get("skills",[]):
        items = sg.get("items",[])
        if items:
            lines.append(f"- **{sg.get('category','')}**: {', '.join(items)}")
    return "\n".join(lines)

def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, AttributeError):
        return os.getenv(key, default)

def get_llm(temperature=0.7):
    """OpenRouter first (verified working), Z.ai as fallback.

    Two fixes worth knowing: the old Z.ai base URL (api.zai.chat) does not
    resolve, and "openrouter/free" is not a real model id.
    """
    oc_key = get_secret("OPENCODE_GO_API_KEY") or os.getenv("OPENCODE_GO_API_KEY")
    if oc_key:
        # opencode subscription: needs an x-opencode-session header or it 400s.
        return ChatOpenAI(
            model=get_secret("OPENCODE_MODEL") or "deepseek-v4.1-flash",
            api_key=oc_key,
            base_url="https://opencode.ai/zen/go/v1",
            temperature=temperature,
            max_tokens=2500,
            default_headers={"x-opencode-session": str(uuid.uuid4()), "User-Agent": "opencode/1.0"},
        )
    or_key = get_secret("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if or_key:
        model = get_secret("MODEL_NAME") or os.getenv("RESUME_AGENT_MODEL") or "deepseek/deepseek-chat"
        return ChatOpenAI(model=model, api_key=or_key, base_url="https://openrouter.ai/api/v1", temperature=temperature)
    zai_key = get_secret("ZAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if zai_key:
        return ChatOpenAI(model="glm-4.6", api_key=zai_key, base_url="https://api.z.ai/api/paas/v4", temperature=temperature)
    raise RuntimeError("No LLM credentials: set OPENROUTER_API_KEY")

def extract_text_from_doc(doc):
    text_parts = []
    for element in doc.get("body", {}).get("content", []):
        if "paragraph" in element:
            for elem in element["paragraph"].get("elements", []):
                if "textRun" in elem:
                    text_parts.append(elem["textRun"]["content"])
        elif "table" in element:
            for row in element["table"].get("tableRows", []):
                for cell in row.get("tableCells", []):
                    for celem in cell.get("content", []):
                        if "paragraph" in celem:
                            for pelem in celem["paragraph"].get("elements", []):
                                if "textRun" in pelem:
                                    text_parts.append(pelem["textRun"]["content"])
    return "".join(text_parts)

def fetch_google_doc(url):
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError("Invalid Google Doc URL")
    doc_id = match.group(1)
    sa_path = get_secret("service_account_path")
    if sa_path:
        creds = service_account.Credentials.from_service_account_file(sa_path, scopes=["https://www.googleapis.com/auth/documents.readonly"])
    else:
        creds_dict = dict(st.secrets["service_account_json"])
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/documents.readonly"])
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()
    return extract_text_from_doc(doc)

class _OnnxMiniLM:
    """LangChain-compatible wrapper around Chroma's bundled ONNX MiniLM.

    Avoids sentence-transformers/torch entirely (no Intel-macOS wheel risk) while
    keeping real semantic retrieval instead of keyword fallback.
    """
    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return [list(map(float, v)) for v in self._ef(list(texts))]

    def embed_query(self, text):
        return list(map(float, self._ef([text])[0]))


@st.cache_resource
def get_embeddings():
    try:
        return _OnnxMiniLM()
    except Exception:
        if _EMBEDDINGS_AVAILABLE:
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        return None

@st.cache_resource
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "])
    chunks = splitter.create_documents([text])
    embeddings = get_embeddings()
    if embeddings is None:
        return None, chunks
    vs = Chroma.from_documents(chunks, embeddings, collection_name="resume")
    return vs, chunks

def load_jobs_db():
    """jobs.json is stored base64-encoded (obfuscated links). Plain json.load
    returns {} on it, which locked out every ?j= invite link."""
    try:
        with open(JOBS_JSON_PATH, "r") as f:
            raw = f.read().strip()
    except FileNotFoundError:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        import base64
        return json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception:
        return {}

def retrieve_context(vectorstore, chunks, query, k=4):
    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(d.page_content for d in docs)
    return "\n\n".join(c.page_content for c in chunks[:3])

SKILL_ALIASES = {
    "agents": ["agent", "agentic", "multi-agent", "langgraph", "mcp", "tool-calling", "orchestrat", "autonomous"],
    "vector-databases": ["vector", "chroma", "embedding", "faiss", "pinecone", "pgvector", "retrieval"],
    "rag": ["rag", "retrieval-augmented", "retrieval augmented", "retrieval"],
    "mlops": ["mlops", "mlflow", "deployment", "monitoring", "serving", "vllm", "observability", "pipeline"],
    "cloud": ["aws", "azure", "gcp", "cloud", "hugging face", "huggingface"],
    "ml-engineering": ["machine learning", "model", "training", "inference", "pytorch", "sklearn"],
    "data-engineering": ["etl", "pipeline", "sql", "postgres", "mysql", "data"],
    "model-monitoring": ["monitoring", "observability", "eval", "drift", "logging"],
    "prompt-engineering": ["prompt", "system prompt", "few-shot", "instructions"],
    "llm": ["llm", "gpt", "glm", "deepseek", "gemini", "language model"],
    "cicd": ["ci/cd", "continuous", "github actions", "pipeline", "deploy"],
}


def _skill_present(skill, haystack):
    """Skill match that survives wording differences.

    The old version looked for the literal string in the top 2 chunks, so a
    resume that says "agent infrastructure" and "LangGraph" scored a gap on
    "agents". That is a measurement artifact, not a missing skill.
    """
    if skill_in_text(skill, haystack):
        return True
    s = str(skill).lower().strip()
    variants = {s, s.replace("-", " "), s.replace("-", ""), s.rstrip("s")}
    variants |= set(SKILL_ALIASES.get(s, []))
    return any(v and v in haystack for v in variants)


def compute_match_score(vectorstore, chunks, jd_skills, full_text=None):
    if not jd_skills:
        return 0, [], []
    haystack = (full_text or " ".join(c.page_content for c in chunks)).lower()
    matched = [s for s in jd_skills if _skill_present(s, haystack)]
    gaps = [s for s in jd_skills if s not in matched]
    score = int((len(matched) / max(len(jd_skills), 1)) * 100)
    return score, matched, gaps


SKILL_DISPLAY = {
    "python": "Python", "llm": "LLMs", "pytorch": "PyTorch", "rag": "RAG",
    "langchain": "LangChain", "langgraph": "LangGraph", "vector-databases": "vector databases",
    "prompt-engineering": "prompt engineering", "mlops": "MLOps", "cloud": "cloud platforms",
    "agents": "agents", "docker": "Docker", "kubernetes": "Kubernetes", "cicd": "CI/CD",
    "model-monitoring": "model monitoring", "data-engineering": "data engineering",
    "ml-engineering": "ML engineering", "fastapi": "FastAPI", "react": "React",
    "typescript": "TypeScript", "azure": "Azure", "aws": "AWS", "sql": "SQL", "git": "Git",
}


def _disp(skill):
    return SKILL_DISPLAY.get(str(skill).lower(), str(skill))


def generate_suggested_questions(jd_entry):
    """Presets written from the job's own requirements, phrased for a recruiter."""
    skills = [_disp(s) for s in jd_entry.get("skills", [])]
    title = jd_entry["title"]
    company = jd_entry["company"]
    qs = []
    if skills:
        qs.append(f"Where has Yang shipped {skills[0]} in production?")
        qs.append(f"How does his background map to the {title} role at {company}?")
        if len(skills) > 2:
            qs.append(f"Walk me through his most relevant project using {skills[1]} and {skills[2]}")
    qs.append(f"What makes him a strong fit for this {title} role?")
    qs.append("Where would he need to ramp up?")
    return qs[:5]


SYSTEM_GENERIC = """You are Yang's AI resume assistant. You have access to Yang's resume content retrieved via semantic search. Your job is to answer questions about Yang's background, skills, experience, and projects accurately and honestly based ONLY on the provided resume context. If something is not in the resume, say so. Keep responses concise and professional. Do not use em dashes; use commas, colons, or full stops instead.

RISK NOTES: {risk_notes}

RESUME CONTEXT: {context}"""

SYSTEM_WITH_JD = """You are Yang's AI resume assistant. You have the full resume below and a specific job description. Compare Yang's qualifications against the JD and answer questions about fit, strengths and gaps.

Rules:
- Ground every claim in the resume. You are given the whole resume, so a real gap is a real gap. Do not infer one from a section that simply does not mention it.
- When a gap is real, say it once, then name the closest adjacent evidence in the resume (transferable work, same class of problem).
- Do not describe shipped or publicly deployed projects as academic, personal or side projects. If the resume says a project is deployed and publicly accessible, treat it as shipped work.
- Do not invent requirements that are not in the JD, and do not downgrade a skill the resume lists.
- Be concise. No em dashes: use commas, colons or full stops.

RISK NOTES: {risk_notes}
ROLE CONTEXT: {role_context}
JOB DESCRIPTION ({title} at {company}): {jd}
RESUME (full): {context}"""

ACCESS_LOCKED_CSS = """
<style>
    .stApp { background-color: #0d1117; }
    .lock-container { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 80vh; text-align: center; padding: 40px; }
    .lock-icon { font-size: 64px; margin-bottom: 16px; }
    .lock-title { color: #c9d1d9; font-size: 24px; font-weight: 600; margin-bottom: 8px; }
    .lock-sub { color: #8b949e; font-size: 14px; }
</style>
"""

ACCESS_LOG = Path.home() / "Library" / "Logs" / "resume-agent" / "access-requests.jsonl"
COUNT_STORE = Path.home() / "Library" / "Logs" / "resume-agent" / "query-counts.json"

def load_query_counts():
    try:
        return json.loads(COUNT_STORE.read_text())
    except Exception:
        return {}

def save_query_counts(counts):
    try:
        COUNT_STORE.parent.mkdir(parents=True, exist_ok=True)
        COUNT_STORE.write_text(json.dumps(counts, indent=1, sort_keys=True))
    except Exception:
        pass

def account_key(email, token, is_owner):
    """Identity used for the question cap. Owner is uncapped."""
    if is_owner:
        return "OWNER"
    if email:
        return "email:" + email.strip().lower()
    if token:
        return "token:" + token
    return "anon"


def log_access_request(email, jd_text="", token=""):
    """Append a voluntary access request. No IPs, no device data."""
    try:
        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "email": email,
            "jd_provided": bool(jd_text.strip()),
            "jd_chars": len(jd_text.strip()),
            "jd_excerpt": jd_text.strip()[:400],
            "token": token,
        }
        with open(ACCESS_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
        return True
    except Exception:
        return False

def main():
    # Access gate: require valid ?j= or ?owner=true or ?override=hash
    query_params = st.query_params
    j_param = query_params.get("j")
    is_owner = query_params.get("owner") == "true"
    is_overridden = query_params.get("override") == OWNER_HASH
    jobs_db = load_jobs_db()
    has_valid_job = j_param in jobs_db if j_param else False

    # No token match: fall through to the generic public view (box always available).
    # Access request gate: recruiter links and owner skip it.
    if not (has_valid_job or is_owner or is_overridden):
        if not st.session_state.get("unlocked"):
            st.markdown('<div class="welcome-banner"><h3>Yang Lu, Resume Assistant</h3><p>Request access to ask questions about Yang background, or paste a job description for a fit review.</p></div>', unsafe_allow_html=True)
            with st.form("access_request_form"):
                req_email = st.text_input("Your email")
                req_jd = st.text_area("Job description (optional)", height=180)
                req_submit = st.form_submit_button("Request access", use_container_width=True)
            if req_submit:
                em = (req_email or "").strip()
                if "@" not in em or "." not in em.split("@")[-1]:
                    st.error("Please enter a valid email address.")
                else:
                    log_access_request(em, req_jd or "", j_param or "")
                    st.session_state.unlocked = True
                    st.session_state.requester_email = em
                    st.session_state.acct_key = account_key(em, j_param or "", False)
                    if (req_jd or "").strip():
                        st.session_state.jd_entry = parse_jd_text(req_jd.strip())
                        st.session_state.jd_prompt_dismissed = True
                    st.rerun()
            st.stop()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "acct_key" not in st.session_state:
        st.session_state.acct_key = account_key(
            st.session_state.get("requester_email", ""),
            j_param or "",
            bool(is_owner or is_overridden),
        )
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "resume_loaded" not in st.session_state:
        st.session_state.resume_loaded = False

    # Load resume
    if not st.session_state.resume_loaded:
        resume_text = load_resume_from_json()
        if resume_text:
            st.session_state.resume_text = resume_text
            st.session_state.vs, st.session_state.chunks = build_vector_store(resume_text)
            st.session_state.resume_loaded = True
            st.session_state.resume_source = "resume.json"
            with st.spinner("Analyzing resume..."):
                from resume_audit import audit_recruiter_risks
                st.session_state.risk_flags = audit_recruiter_risks(resume_text)
        else:
            resume_url = get_secret(RESUME_SOURCE_URL_ENV, "")
            if resume_url:
                try:
                    with st.spinner("Loading resume from Google Docs..."):
                        resume_text = fetch_google_doc(resume_url)
                    st.session_state.resume_text = resume_text
                    st.session_state.vs, st.session_state.chunks = build_vector_store(resume_text)
                    st.session_state.resume_loaded = True
                    st.session_state.resume_source = "google_doc"
                except Exception as e:
                    st.error(f"Could not load resume: {e}")
                    st.session_state.resume_text = ""
                    st.session_state.vs = None
                    st.session_state.chunks = []
                    st.session_state.resume_loaded = True
            else:
                st.session_state.resume_text = ""
                st.session_state.vs = None
                st.session_state.chunks = []
                st.session_state.resume_loaded = True

    # Detect ?j= param
    jd_entry = None
    if j_param and j_param in jobs_db:
        jd_entry = jobs_db[j_param]
        st.session_state.jd_entry = jd_entry
    if jd_entry is None and "jd_entry" in st.session_state:
        jd_entry = st.session_state.jd_entry

    if is_owner or is_overridden:
        st.session_state.bypass_prompt = True

    # JD prompt for owners
    if jd_entry is None and not st.session_state.get("bypass_prompt"):
        if "jd_prompt_dismissed" not in st.session_state:
            with st.container():
                st.markdown('<div style="background-color:#1a1f2b;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px;"><p style="margin:0;color:#c9d1d9;">Have a specific role in mind? Paste a job description and I will analyze Yang fit.</p></div>', unsafe_allow_html=True)
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    jd_quick = st.text_input("Paste job description", placeholder="e.g. Senior Backend Engineer, Python...", label_visibility="collapsed")
                with col_b:
                    if st.button("Analyze", use_container_width=True):
                        if jd_quick.strip():
                            from resume_audit import analyze_role_context
                            jd_entry = parse_jd_text(jd_quick.strip())
                            st.session_state.jd_entry = jd_entry
                            st.session_state.jd_prompt_dismissed = True
                            st.rerun()
                if st.button("Skip", key="skip_jd"):
                    st.session_state.jd_prompt_dismissed = True
                    st.rerun()

    if jd_entry and st.session_state.resume_text and "role_context" not in st.session_state:
        with st.spinner("Analyzing role context..."):
            from resume_audit import analyze_role_context
            ctx = analyze_role_context(st.session_state.resume_text, jd_entry["title"])
            if ctx:
                st.session_state.role_context = ctx

    # Sidebar
    with st.sidebar:
        st.markdown("### Yang's Resume Assistant")
        if st.session_state.resume_text:
            src_label = {"resume.json": "resume.json", "google_doc": "Google Docs"}
            src = src_label.get(st.session_state.get("resume_source", ""), "")
            st.success(f"{src} Resume loaded ({len(st.session_state.resume_text):,} chars)")
        else:
            st.warning("No resume loaded")
        st.divider()
        acct = st.session_state.get("acct_key", "anon")
        counts = load_query_counts()
        used = 0 if acct == "OWNER" else counts.get(acct, 0)
        remaining = 999 if acct == "OWNER" else max(0, MAX_QUERIES - used)
        if acct == "OWNER":
            st.caption("Owner session, no question limit")
        elif 0 < remaining <= 3:
            st.caption(f"{remaining} questions left on this account")
        if remaining <= 0:
            st.info(f"You have used all {MAX_QUERIES} questions on this account. Reply to Yang's email if you need more.")
        if acct == "OWNER" and st.button("Reset all question counts"):
            save_query_counts({})
            st.rerun()
        st.divider()
        if jd_entry:
            st.markdown(f"**{jd_entry['title']}**")
            st.markdown(f"{jd_entry['company']}")
            if st.button("Clear Job Context"):
                del st.session_state.jd_entry
                st.rerun()
        st.divider()
        if st.button("New Session", type="primary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Welcome card
    if jd_entry:
        title = jd_entry["title"]
        company = jd_entry["company"]
        skills = jd_entry.get("skills", [])
        st.markdown(f'<div class="welcome-banner"><h3>Thanks for taking a look.</h3><p>I am the resume assistant for Yang Lu. Ask me anything about his background, or how it lines up with the <strong>{title}</strong> role at <strong>{company}</strong>. Every answer comes from his actual resume, so if something is not in there, I will say so rather than guess.</p></div>', unsafe_allow_html=True)
        vs = getattr(st.session_state, "vs", None)
        chunks = getattr(st.session_state, "chunks", [])
        if chunks:
            score, matched, gaps = compute_match_score(vs, chunks, skills, st.session_state.get("resume_text"))
            st.markdown(f'<div class="card"><h3>Match Score: {score}%</h3><div class="match-bar-bg"><div class="match-bar-fill" style="width:{min(score,100)}%;"></div></div></div>', unsafe_allow_html=True)
            chips = ""
            for s in skills:
                cls = "matched" if s in matched else "gap"
                icon = "&#10003;" if s in matched else "&#10007;"
                chips += f'<span class="skill-chip {cls}">{icon} {s}</span> '
            st.markdown(f'<div class="card"><h4>Skills Assessment</h4>{chips}</div>', unsafe_allow_html=True)
        suggestions = generate_suggested_questions(jd_entry)
        st.markdown("#### Suggested Questions")
        cols = st.columns(min(len(suggestions), 3))
        for idx, q in enumerate(suggestions):
            with cols[idx % len(cols)]:
                if st.button(q, key=f"sug_{idx}"):
                    st.session_state.suggested_query = q
    else:
        st.markdown('<div class="welcome-banner"><h3>Thanks for stopping by.</h3><p>I am the resume assistant for Yang Lu. Ask me anything about his background, projects, or the kind of role he is looking for. Answers come from his actual resume, so if something is not covered, I will tell you straight rather than make it up.</p></div>', unsafe_allow_html=True)

    # Chat
    for msg in st.session_state.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        avatar = "&#x1F916;" if role == "assistant" else "&#x1F464;"
        with st.chat_message(role):
            st.markdown(msg.content)

    acct = st.session_state.get("acct_key", "anon")
    _counts = load_query_counts()
    remaining = 999 if acct == "OWNER" else max(0, MAX_QUERIES - _counts.get(acct, 0))
    disabled = remaining <= 0
    if "suggested_query" in st.session_state:
        prefill = st.session_state.pop("suggested_query")
    else:
        prefill = None
    user_input = st.chat_input("Ask anything about Yang's background..." if not disabled else "Session complete. Reload for a fresh start.", disabled=disabled)
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        if remaining <= 0:
            st.error(f"{MAX_QUERIES}-question limit reached for this account.")
            st.stop()
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        vs = getattr(st.session_state, "vs", None)
        chunks = getattr(st.session_state, "chunks", [])
        context = retrieve_context(vs, chunks, user_input)
        # A JD comparison is an absence question ("what is he missing?"). Four retrieved
        # chunks are about a third of the resume, so the model answered "not listed" about
        # skills that were sitting in chunks it never saw. The whole resume is ~6k chars,
        # cheap enough to send in full whenever a JD is loaded.
        if jd_entry and st.session_state.get("resume_text"):
            context = st.session_state["resume_text"]
        flags = st.session_state.get("risk_flags", [])
        if flags:
            risk_notes = "Items flagged:"
            for f in flags:
                risk_notes += f"\n- [{f.get('severity',5)}/10] {f.get('issue','')}"
        else:
            risk_notes = "None."

        role_context = ""
        if jd_entry:
            ctx = st.session_state.get("role_context", {})
            if ctx:
                from resume_audit import format_context_guide
                role_context = format_context_guide(ctx)

        if jd_entry:
            system_text = SYSTEM_WITH_JD.format(context=context, jd=jd_entry["description"], title=jd_entry["title"], company=jd_entry["company"], risk_notes=risk_notes, role_context=role_context)
        else:
            system_text = SYSTEM_GENERIC.format(context=context, risk_notes=risk_notes)

        llm = get_llm(temperature=0.5)
        prompt_messages = [HumanMessage(content=system_text)]
        history = st.session_state.messages[-8:]
        prompt_messages.extend(history)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = llm.invoke(prompt_messages)
                    answer = response.content
                    ctx = st.session_state.get("role_context", {})
                    if ctx:
                        from resume_audit import filter_response
                        filtered = filter_response(answer, ctx)
                        if filtered and filtered != answer:
                            answer = filtered
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.query_count += 1
        if acct != "OWNER":
            _c = load_query_counts()
            _c[acct] = _c.get(acct, 0) + 1
            save_query_counts(_c)
        st.rerun()

    st.markdown('<div class="footer">Powered by LangChain + Chroma + DeepSeek</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
