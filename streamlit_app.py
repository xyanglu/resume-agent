"""
Resume Agent - RAG Chatbot with Custom Job Links
LangChain + Chroma + OpenRouter | Dark Themed UI | 5-Query Session Cap
"""

import streamlit as st
import os
import re
import json
import hashlib
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

MAX_QUERIES = 5
RESUME_SOURCE_URL_ENV = "RESUME_URL"
OWNER_HASH = "d4c5627c"

def parse_jd_text(raw_text):
    return {
        "title": "this role",
        "company": "your company",
        "description": raw_text,
        "skills": [s.strip().lower() for s in raw_text.replace(",", " ").split() if len(s.strip()) > 2 and s.strip().lower() not in ["the", "and", "for", "with", "from", "that", "this", "are", "have", "will"]],
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
    try:
        import urllib.request
        resp = urllib.request.urlopen(GITHUB_RESUME_URL, timeout=10)
        data = json.loads(resp.read().decode())
        if data and data.get("name"):
            return _format_resume_text(data)
    except Exception:
        pass
    for p in [RESUME_JSON_PATH, LOCAL_RESUME_JSON]:
        try:
            with open(p) as f:
                return _format_resume_text(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return None

def _format_resume_text(data):
    lines = []
    lines.append(f"# {data['name']}")
    c = data.get("contact", {})
    parts = [v for v in [c.get("location",""), c.get("email",""), c.get("linkedin",""), c.get("website",""), c.get("huggingface","")] if v]
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
    zai_key = get_secret("ZAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if zai_key:
        return ChatOpenAI(model="glm-4.7-flash", api_key=zai_key, base_url="https://api.zai.chat/v1", temperature=temperature)
    return ChatOpenAI(model="openrouter/free", api_key=get_secret("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1", temperature=temperature)

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

@st.cache_resource
def get_embeddings():
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
    try:
        with open(JOBS_JSON_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def retrieve_context(vectorstore, chunks, query, k=4):
    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(d.page_content for d in docs)
    return "\n\n".join(c.page_content for c in chunks[:3])

def compute_match_score(vectorstore, chunks, jd_skills):
    if not jd_skills or vectorstore is None:
        all_text = " ".join(c.page_content for c in chunks).lower()
        matched = [s for s in jd_skills if s.lower() in all_text]
        gaps = [s for s in jd_skills if s.lower() not in all_text]
        score = int((len(matched) / max(len(jd_skills), 1)) * 100)
        return score, matched, gaps
    matched = []
    for skill in jd_skills:
        results = vectorstore.similarity_search(skill, k=2)
        combined = " ".join(d.page_content.lower() for d in results)
        skill_lower = skill.lower()
        variants = [skill_lower, skill_lower.replace("-", " "), skill_lower.replace("-", "")]
        if any(v in combined for v in variants):
            matched.append(skill)
    gaps = [s for s in jd_skills if s not in matched]
    score = int((len(matched) / max(len(jd_skills), 1)) * 100)
    return score, matched, gaps

def generate_suggested_questions(jd_entry):
    skills = jd_entry.get("skills", [])
    title = jd_entry["title"]
    company = jd_entry["company"]
    questions = []
    if skills:
        top_skills = skills[:3]
        questions.append(f"What experience does Yang have with {top_skills[0]}?")
        if len(skills) > 1:
            questions.append(f"How does Yang's background align with the {title} role requirements at {company}?")
        if len(skills) > 2:
            questions.append(f"Can you walk through Yang's projects involving {top_skills[1]} and {top_skills[2]}?")
    questions.append(f"What are Yang's strongest qualifications for this {title} position?")
    questions.append(f"Are there any gaps between Yang's experience and the {title} role at {company}?")
    return questions[:5]

SYSTEM_GENERIC = """You are Yang's AI resume assistant. You have access to Yang's resume content retrieved via semantic search. Your job is to answer questions about Yang's background, skills, experience, and projects accurately and honestly based ONLY on the provided resume context. If something is not in the resume, say so. Keep responses concise and professional.

RISK NOTES: {risk_notes}

RESUME CONTEXT: {context}"""

SYSTEM_WITH_JD = """You are Yang's AI resume assistant. You have access to Yang's resume content and a specific job description. Your job is to compare Yang's qualifications against the job requirements and answer questions about fit, alignment, strengths, and gaps. Be specific and reference actual experience that matches the JD requirements. If something is not in the resume, say so. Keep responses concise and professional.

RISK NOTES: {risk_notes}
ROLE CONTEXT: {role_context}
JOB DESCRIPTION ({title} at {company}): {jd}
RESUME CONTEXT: {context}"""

ACCESS_LOCKED_CSS = """
<style>
    .stApp { background-color: #0d1117; }
    .lock-container { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 80vh; text-align: center; padding: 40px; }
    .lock-icon { font-size: 64px; margin-bottom: 16px; }
    .lock-title { color: #c9d1d9; font-size: 24px; font-weight: 600; margin-bottom: 8px; }
    .lock-sub { color: #8b949e; font-size: 14px; }
</style>
"""

def main():
    # Access gate: require valid ?j= or ?owner=true or ?override=hash
    query_params = st.query_params
    j_param = query_params.get("j")
    is_owner = query_params.get("owner") == "true"
    is_overridden = query_params.get("override") == OWNER_HASH
    jobs_db = load_jobs_db()
    has_valid_job = j_param in jobs_db if j_param else False

    if not has_valid_job and not is_owner and not is_overridden:
        st.markdown(ACCESS_LOCKED_CSS, unsafe_allow_html=True)
        st.markdown(
            '<div class="lock-container">'
            '<div class="lock-icon">🔒</div>'
            '<div class="lock-title">This link is for invited recruiters only</div>'
            '<div class="lock-sub">If you are a recruiter reviewing Yang application please use the link provided in your email invitation.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="footer" style="position:fixed;bottom:0;width:100%;text-align:center;color:#484f58;font-size:12px;padding:12px 0;">Yang Lu &middot; AI Engineer</div>', unsafe_allow_html=True)
        return

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
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
        st.markdown("### Resume Agent")
        if st.session_state.resume_text:
            src_label = {"resume.json": "resume.json", "google_doc": "Google Docs"}
            src = src_label.get(st.session_state.get("resume_source", ""), "")
            st.success(f"{src} Resume loaded ({len(st.session_state.resume_text):,} chars)")
        else:
            st.warning("No resume loaded")
        st.divider()
        remaining = MAX_QUERIES - st.session_state.query_count
        st.metric("Queries remaining", remaining)
        if remaining <= 0:
            st.error("Session limit reached")
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
        st.markdown(f'<div class="welcome-banner"><h3>Hi! I am Yang resume assistant.</h3><p>Ask me how Yang fits the <strong>{title}</strong> role at <strong>{company}</strong>.</p></div>', unsafe_allow_html=True)
        vs = getattr(st.session_state, "vs", None)
        chunks = getattr(st.session_state, "chunks", [])
        if chunks:
            score, matched, gaps = compute_match_score(vs, chunks, skills)
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
        st.markdown('<div class="welcome-banner"><h3>Hi! I am Yang resume assistant.</h3><p>Ask me anything about Yang background skills and experience.</p></div>', unsafe_allow_html=True)

    # Chat
    for msg in st.session_state.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        avatar = "&#x1F916;" if role == "assistant" else "&#x1F464;"
        with st.chat_message(role):
            st.markdown(msg.content)

    remaining = MAX_QUERIES - st.session_state.query_count
    disabled = remaining <= 0
    if "suggested_query" in st.session_state:
        prefill = st.session_state.pop("suggested_query")
    else:
        prefill = None
    user_input = st.chat_input("Ask about Yang resume..." if not disabled else "Session limit reached", disabled=disabled)
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        if st.session_state.query_count >= MAX_QUERIES:
            st.error("5-query limit reached. Start a new session.")
            st.stop()
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        vs = getattr(st.session_state, "vs", None)
        chunks = getattr(st.session_state, "chunks", [])
        context = retrieve_context(vs, chunks, user_input)
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
        st.rerun()

    st.markdown('<div class="footer">Powered by LangChain + Chroma + OpenRouter</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
