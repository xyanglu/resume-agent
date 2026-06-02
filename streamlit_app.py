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

# ---------------------------------------------------------------------------
# Try to load community embeddings; fall back to a stub if unavailable
# (HF Spaces may not have sentence-transformers installed)
# ---------------------------------------------------------------------------
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page Config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Yang's Resume Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark Theme CSS
# ---------------------------------------------------------------------------
DARK_CSS = """
<style>
    /* Global dark background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    /* Chat input */
    .stChatInput textarea {
        background-color: #1a1f2b !important;
        color: #fafafa !important;
    }
    /* Cards / containers */
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card h3 { margin-top: 0; }
    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, #1a1f2b 0%, #1f2937 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    /* Match score bar */
    .match-bar-bg {
        background-color: #21262d;
        border-radius: 8px;
        height: 14px;
        overflow: hidden;
    }
    .match-bar-fill {
        height: 14px;
        border-radius: 8px;
        background: linear-gradient(90deg, #238636, #3fb950);
    }
    /* Skill chips */
    .skill-chip {
        display: inline-block;
        background-color: #1f2937;
        border: 1px solid #30363d;
        border-radius: 999px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 13px;
        color: #c9d1d9;
    }
    .skill-chip.matched {
        border-color: #238636;
        color: #3fb950;
    }
    .skill-chip.gap {
        border-color: #da3633;
        color: #f85149;
    }
    /* Suggested question buttons */
    .suggested-btn {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 4px;
        color: #58a6ff;
        cursor: pointer;
        text-align: left;
        width: 100%;
    }
    .suggested-btn:hover {
        background-color: #1f2937;
        border-color: #58a6ff;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #484f58;
        font-size: 12px;
        padding: 24px 0 12px 0;
    }
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_QUERIES = 5
RESUME_SOURCE_URL_ENV = "RESUME_URL"
JOBS_JSON_PATH = Path(__file__).parent / "jobs.json"

# ---------------------------------------------------------------------------
# Helper: secrets
# ---------------------------------------------------------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, AttributeError):
        return os.getenv(key, default)

# ---------------------------------------------------------------------------
# Helper: LLM
# ---------------------------------------------------------------------------
def get_llm(temperature=0.7):
    zai_key = get_secret("ZAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if zai_key:
        return ChatOpenAI(
            model="glm-5-turbo",
            api_key=zai_key,
            base_url="https://api.zai.chat/v1",
            temperature=temperature,
        )
    return ChatOpenAI(
        model="openrouter/free",
        api_key=get_secret("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )

# ---------------------------------------------------------------------------
# Helper: Google Docs extraction
# ---------------------------------------------------------------------------
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
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/documents.readonly"]
        )
    else:
        creds_dict = dict(st.secrets["service_account_json"])
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/documents.readonly"]
        )
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()
    return extract_text_from_doc(doc)

# ---------------------------------------------------------------------------
# Helper: Embeddings (lazy init)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    if _EMBEDDINGS_AVAILABLE:
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return None

# ---------------------------------------------------------------------------
# Helper: Build / load vector store
# ---------------------------------------------------------------------------
@st.cache_resource
def build_vector_store(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.create_documents([text])
    embeddings = get_embeddings()
    if embeddings is None:
        return None, chunks
    vs = Chroma.from_documents(chunks, embeddings, collection_name="resume")
    return vs, chunks

# ---------------------------------------------------------------------------
# Helper: Load jobs.json
# ---------------------------------------------------------------------------
def load_jobs_db():
    try:
        with open(JOBS_JSON_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# ---------------------------------------------------------------------------
# Helper: Retrieve relevant resume chunks for a query
# ---------------------------------------------------------------------------
def retrieve_context(vectorstore, chunks, query, k=4):
    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(d.page_content for d in docs)
    # Fallback: return first 3 chunks
    return "\n\n".join(c.page_content for c in chunks[:3])

# ---------------------------------------------------------------------------
# Helper: Compute client-side match score
# ---------------------------------------------------------------------------
def compute_match_score(vectorstore, chunks, jd_skills):
    """Compare JD skills against vector-store similarity search results.
    Returns (score_pct, matched, gaps)."""
    if not jd_skills or vectorstore is None:
        # Fallback: keyword search against raw chunks text
        all_text = " ".join(c.page_content for c in chunks).lower()
        matched = [s for s in jd_skills if s.lower() in all_text]
        gaps = [s for s in jd_skills if s.lower() not in all_text]
        score = int((len(matched) / max(len(jd_skills), 1)) * 100)
        return score, matched, gaps

    matched = []
    for skill in jd_skills:
        results = vectorstore.similarity_search(skill, k=2)
        combined = " ".join(d.page_content.lower() for d in results)
        # Check skill and common variants
        skill_lower = skill.lower()
        variants = [skill_lower, skill_lower.replace("-", " "), skill_lower.replace("-", "")]
        if any(v in combined for v in variants):
            matched.append(skill)

    gaps = [s for s in jd_skills if s not in matched]
    score = int((len(matched) / max(len(jd_skills), 1)) * 100)
    return score, matched, gaps

# ---------------------------------------------------------------------------
# Helper: Generate suggested questions from JD
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_GENERIC = """You are Yang's AI resume assistant. You have access to Yang's resume content \
retrieved via semantic search. Your job is to answer questions about Yang's background, skills, \
experience, and projects accurately and honestly based ONLY on the provided resume context.

If something is not in the resume, say so — do not fabricate information.
Keep responses concise, professional, and actionable.

RESUME CONTEXT:
{context}"""

SYSTEM_WITH_JD = """You are Yang's AI resume assistant. You have access to Yang's resume content \
and a specific job description. Your job is to compare Yang's qualifications against the job \
requirements and answer questions about fit, alignment, strengths, and gaps.

Be specific — reference actual experience and projects from the resume that match the JD requirements.
If something is not in the resume, say so — do not fabricate information.
Keep responses concise, professional, and actionable.

JOB DESCRIPTION — {title} at {company}:
{jd}

RESUME CONTEXT:
{context}"""

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Session state initialization
    # ------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "resume_loaded" not in st.session_state:
        st.session_state.resume_loaded = False

    # ------------------------------------------------------------------
    # Load resume (once per session)
    # ------------------------------------------------------------------
    if not st.session_state.resume_loaded:
        resume_url = get_secret(RESUME_SOURCE_URL_ENV, "")
        if resume_url:
            try:
                with st.spinner("Loading resume..."):
                    resume_text = fetch_google_doc(resume_url)
                st.session_state.resume_text = resume_text
                st.session_state.vs, st.session_state.chunks = build_vector_store(resume_text)
                st.session_state.resume_loaded = True
            except Exception as e:
                st.error(f"Could not load resume: {e}")
                st.session_state.resume_text = ""
                st.session_state.vs = None
                st.session_state.chunks = []
                st.session_state.resume_loaded = True  # Don't retry endlessly
        else:
            st.session_state.resume_text = ""
            st.session_state.vs = None
            st.session_state.chunks = []
            st.session_state.resume_loaded = True

    # ------------------------------------------------------------------
    # Load jobs DB & detect ?j= param
    # ------------------------------------------------------------------
    jobs_db = load_jobs_db()
    jd_entry = None
    query_params = st.query_params
    j_param = query_params.get("j")
    if j_param and j_param in jobs_db:
        jd_entry = jobs_db[j_param]
        st.session_state.jd_entry = jd_entry

    # Use the JD from session state (survives reruns when user types in chat)
    if jd_entry is None and "jd_entry" in st.session_state:
        jd_entry = st.session_state.jd_entry

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📄 Yang's Resume Agent")

        if st.session_state.resume_text:
            st.success(f"Resume loaded ({len(st.session_state.resume_text):,} chars)")
        else:
            st.warning("No resume loaded")

        st.divider()

        # Query cap indicator
        remaining = MAX_QUERIES - st.session_state.query_count
        st.metric("Queries remaining", remaining)
        if remaining <= 2 and remaining > 0:
            st.warning(f"Only {remaining} queries left this session")
        elif remaining <= 0:
            st.error("Session limit reached")

        st.divider()

        if jd_entry:
            st.markdown("#### 🎯 Target Role")
            st.write(f"**{jd_entry['title']}**")
            st.write(f"{jd_entry['company']}")
            st.write(f"📍 {jd_entry['location']}")

            if st.button("Clear Job Context"):
                del st.session_state.jd_entry
                st.rerun()

        st.divider()
        if st.button("🔄 New Session", type="primary"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ------------------------------------------------------------------
    # JD-specific: Welcome card + Match Score
    # ------------------------------------------------------------------
    if jd_entry:
        title = jd_entry["title"]
        company = jd_entry["company"]
        skills = jd_entry.get("skills", [])

        # Welcome banner
        st.markdown(
            f"""
            <div class="welcome-banner">
                <h3>👋 Hi! I'm Yang's resume assistant.</h3>
                <p style="font-size:16px; color:#c9d1d9;">
                    Ask me how Yang fits the <strong style="color:#58a6ff;">{title}</strong> 
                    role at <strong style="color:#58a6ff;">{company}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Match Score card
        vs = getattr(st.session_state, "vs", None)
        chunks = getattr(st.session_state, "chunks", [])
        if chunks:
            score, matched, gaps = compute_match_score(vs, chunks, skills)
            bar_width = min(score, 100)
            st.markdown(
                f"""
                <div class="card">
                    <h3>🎯 Match Score: {score}%</h3>
                    <div class="match-bar-bg">
                        <div class="match-bar-fill" style="width: {bar_width}%;"></div>
                    </div>
                    <p style="margin-top:12px; font-size:13px; color:#8b949e;">
                        Based on semantic similarity between Yang's resume and the JD requirements.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Skill chips
            chips_html = ""
            for s in skills:
                cls = "matched" if s in matched else "gap"
                icon = "✓" if s in matched else "✗"
                chips_html += f'<span class="skill-chip {cls}">{icon} {s}</span> '
            st.markdown(
                f'<div class="card"><h4>Skills Assessment</h4>{chips_html}</div>',
                unsafe_allow_html=True,
            )

        # Suggested questions
        suggestions = generate_suggested_questions(jd_entry)
        st.markdown("#### 💡 Suggested Questions")
        cols = st.columns(min(len(suggestions), 3))
        for idx, q in enumerate(suggestions):
            with cols[idx % len(cols)]:
                if st.button(q, key=f"sug_{idx}"):
                    st.session_state.suggested_query = q

    else:
        # Generic welcome
        st.markdown(
            """
            <div class="welcome-banner">
                <h3>👋 Hi! I'm Yang's resume assistant.</h3>
                <p style="font-size:16px; color:#c9d1d9;">
                    Ask me anything about Yang's background, skills, and experience.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Chat display — show history
    # ------------------------------------------------------------------
    for msg in st.session_state.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        avatar = "🤖" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)

    # ------------------------------------------------------------------
    # Chat input
    # ------------------------------------------------------------------
    remaining = MAX_QUERIES - st.session_state.query_count
    disabled = remaining <= 0

    # Handle suggested question click
    if "suggested_query" in st.session_state:
        prefill = st.session_state.pop("suggested_query")
    else:
        prefill = None

    user_input = st.chat_input(
        "Ask about Yang's resume..." if not disabled else "Session limit reached. Start a new session.",
        disabled=disabled,
    )

    # If we have a prefilled suggestion but no manual input, use the suggestion
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # Check cap
        if st.session_state.query_count >= MAX_QUERIES:
            st.error("You've reached the 5-query limit. Please start a new session.")
            st.stop()

        # Append user message
        st.session_state.messages.append(HumanMessage(content=user_input))

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Retrieve context
        vs = getattr(st.session_state, "vs", None)
        chunks = getattr(st.session_state, "chunks", [])
        context = retrieve_context(vs, chunks, user_input)

        # Build system prompt
        if jd_entry:
            system_text = SYSTEM_WITH_JD.format(
                context=context,
                jd=jd_entry["description"],
                title=jd_entry["title"],
                company=jd_entry["company"],
            )
        else:
            system_text = SYSTEM_GENERIC.format(context=context)

        # Build message list for the LLM
        llm = get_llm(temperature=0.5)
        prompt_messages = [HumanMessage(content=system_text)]
        # Include recent conversation history (last 4 exchanges)
        history = st.session_state.messages[-8:]
        prompt_messages.extend(history)

        # Call LLM
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response = llm.invoke(prompt_messages)
                    answer = response.content
                except Exception as e:
                    answer = f"Sorry, I encountered an error: {e}"
            st.markdown(answer)

        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.query_count += 1

        # Rerun to refresh the sidebar counter
        st.rerun()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown(
        """<div class="footer">
            Powered by LangChain + Chroma + OpenRouter &middot; Built with ❤️ by Yang
        </div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
