import streamlit as st
import os
import re
import markdown
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- NEW IMPORTS ---
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# Helper to get secrets
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, AttributeError):
        return os.getenv(key, default)


# Check for WeasyPrint
try:
    from weasyprint import HTML

    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False

# Page Config — must be the very first Streamlit call
st.set_page_config(
    page_title="Resume AI Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — Modern Dark Theme
# =============================================================================
CUSTOM_CSS = """
<style>
/* ---------- Base / Reset ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---------- Root Variables ---------- */
:root {
    --bg-primary:    #0f0f13;
    --bg-secondary:  #16161d;
    --bg-card:       #1c1c27;
    --bg-input:      #22222f;
    --border:        #2a2a3a;
    --text-primary:  #e8e8ef;
    --text-secondary:#9b9bb0;
    --accent:        #6c63ff;
    --accent-light:  #8b83ff;
    --accent-dim:    rgba(108,99,255,0.12);
    --success:       #34d399;
    --danger:        #f87171;
    --warning:       #fbbf24;
}

/* ---------- Global Backgrounds ---------- */
.stApp {
    background-color: var(--bg-primary) !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: var(--bg-secondary) !important;
}
section[data-testid="stSidebar"] .sidebar-title {
    color: var(--text-primary);
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
section[data-testid="stSidebar"] .sidebar-subtitle {
    color: var(--text-secondary);
    font-size: 0.82rem;
    line-height: 1.45;
}

/* ---------- Sidebar divider ---------- */
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}

/* ---------- Chat Messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
    transition: border-color 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    border-color: var(--accent) !important;
}
/* User bubble accent */
[data-testid="stChatMessage"][data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid var(--accent) !important;
}
/* Assistant bubble */
[data-testid="stChatMessage"] p {
    color: var(--text-primary) !important;
    font-size: 0.95rem;
    line-height: 1.7;
}
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] b {
    color: var(--accent-light) !important;
}
/* Avatar styling */
[data-testid="stChatMessageAvatar"] {
    width: 36px !important;
    height: 36px !important;
}
[data-testid="stChatMessageAvatarUser"] svg {
    fill: var(--accent) !important;
}
[data-testid="stChatMessageAvatarAssistant"] svg {
    fill: var(--success) !important;
}

/* ---------- Chat Input ---------- */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"] > div > div {
    background: var(--bg-input) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-secondary) !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--accent-light) !important;
    box-shadow: 0 4px 14px rgba(108,99,255,0.35) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ---------- Status boxes ---------- */
.element-container .stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
}

/* ---------- Spinners ---------- */
.stSpinner > div {
    border-color: var(--accent) transparent transparent transparent !important;
}

/* ---------- Expander ---------- */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
details[open] .streamlit-expanderHeader {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ---------- Welcome Card ---------- */
.welcome-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-secondary));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 32px 36px;
    margin: 24px auto;
    max-width: 640px;
    text-align: center;
}
.welcome-card h2 {
    color: var(--accent-light);
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 0 12px 0;
    letter-spacing: -0.03em;
}
.welcome-card p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.65;
    margin: 6px 0;
}
.welcome-card .chip {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent-light);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 20px;
    padding: 5px 14px;
    margin: 4px;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: default;
    transition: background 0.2s;
}
.welcome-card .chip:hover {
    background: rgba(108,99,255,0.22);
}

/* ---------- Footer ---------- */
.footer-bar {
    text-align: center;
    padding: 18px 0 10px 0;
    color: var(--text-secondary);
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    opacity: 0.75;
}
.footer-bar a {
    color: var(--accent-light);
    text-decoration: none;
}
.footer-bar a:hover {
    text-decoration: underline;
}

/* ---------- Init Progress Steps ---------- */
.init-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    color: var(--text-secondary);
    font-size: 0.9rem;
}
.init-step .icon {
    font-size: 1.1rem;
}
.init-step.done .icon { color: var(--success); }
.init-step.active .icon { color: var(--accent-light); }

/* ---------- Status badges ---------- */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,211,153,0.12);
    color: var(--success);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-badge .dot {
    width: 7px; height: 7px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---------- Mobile Responsive ---------- */
@media (max-width: 768px) {
    .welcome-card {
        padding: 22px 18px;
        margin: 16px 8px;
    }
    .welcome-card h2 {
        font-size: 1.25rem;
    }
    [data-testid="stChatMessage"] {
        padding: 12px 14px !important;
        border-radius: 10px !important;
    }
}

/* ---------- Typing indicator ---------- */
.typing-indicator {
    display: inline-flex;
    gap: 4px;
    padding: 4px 0;
}
.typing-indicator span {
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: typingBounce 1.2s infinite;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# CORE FUNCTIONS — Backend (unchanged)
# =============================================================================

def get_llm(temperature=0.1):
    return ChatOpenAI(
        model="openrouter/free",
        api_key=get_secret("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


def extract_text_from_doc(doc):
    """Extract plain text from Google Docs API response"""
    text = []
    content = doc.get("body").get("content")
    if not content:
        return ""

    for element in content:
        if "paragraph" in element:
            paragraph = element["paragraph"]
            for elem in paragraph.get("elements", []):
                if "textRun" in elem:
                    text.append(elem["textRun"]["content"])
        elif "table" in element:
            table = element["table"]
            for row in table.get("tableRows", []):
                for cell in row.get("tableCells", []):
                    for elem in cell.get("content", []):
                        if "paragraph" in elem:
                            for p_elem in elem["paragraph"].get("elements", []):
                                if "textRun" in p_elem:
                                    text.append(p_elem["textRun"]["content"])
    return "".join(text)


def get_google_docs_content(url):
    """Fetch content from Google Doc using Service Account"""
    # 1. Extract Document ID
    match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError("Invalid Google Doc URL format")
    doc_id = match.group(1)

    # 2. Load Credentials from file path in .env
    service_account_path = get_secret("service_account_path")
    if service_account_path:
        creds = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=["https://www.googleapis.com/auth/documents.readonly"],
        )
    else:
        # Fallback to service_account_json from secrets
        creds_dict = dict(st.secrets["service_account_json"])
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/documents.readonly"]
        )

    # 3. Build Service and Fetch
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()

    return extract_text_from_doc(doc)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Draw the sidebar UI."""
    with st.sidebar:
        # --- Logo / Title ---
        st.markdown(
            """
            <div style="text-align:center; padding: 10px 0 6px 0;">
                <div style="font-size:2.4rem; margin-bottom:2px;">📄</div>
                <div class="sidebar-title">Resume AI Chat</div>
                <div class="sidebar-subtitle">
                    Ask questions about a resume stored in Google Docs.<br>
                    Powered by RAG + semantic search.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Status badge ---
        if st.session_state.get("qa_chain") is not None:
            st.markdown(
                '<div style="text-align:center; margin:10px 0;">'
                '<span class="status-badge"><span class="dot"></span> Online</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center; margin:10px 0;">'
                '<span class="status-badge" style="background:rgba(251,191,36,0.12);color:var(--warning);">'
                '<span class="dot" style="background:var(--warning);"></span> Initializing</span>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Clear Chat ---
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗑  Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄  Reload", use_container_width=True):
                st.session_state.qa_chain = None
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")

        # --- Model Info ---
        st.markdown(
            f"""
            <div style="padding:4px 0;">
                <div style="color:var(--text-secondary); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Model</div>
                <div style="color:var(--text-primary); font-size:0.88rem; font-weight:600;">OpenRouter Free</div>
                <div style="color:var(--text-secondary); font-size:0.78rem; margin-top:2px;">Embedding: MiniLM-L6-v2</div>
                <div style="color:var(--text-secondary); font-size:0.78rem;">Vector Store: ChromaDB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # --- Source Document Info ---
        doc_url = get_secret("RESUME_URL")
        if doc_url:
            doc_id_match = re.search(r"/document/d/([a-zA-Z0-9-_]+)", doc_url)
            display_id = doc_id_match.group(1)[:12] + "..." if doc_id_match else "—"
            st.markdown(
                f"""
                <div style="padding:4px 0;">
                    <div style="color:var(--text-secondary); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Source Document</div>
                    <div style="color:var(--text-primary); font-size:0.82rem;">Google Doc <code style="background:var(--bg-input);padding:2px 6px;border-radius:4px;font-size:0.76rem;">{display_id}</code></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- Settings Expander ---
        with st.expander("⚙️  Settings"):
            temp = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Lower = more factual. Higher = more creative.",
            )
            st.session_state["temperature"] = temp

            chunk_size = st.number_input(
                "Chunk Size",
                min_value=500,
                max_value=8000,
                value=4000,
                step=500,
                help="Size of text chunks for vector store. Larger = more context per chunk.",
            )
            st.session_state["chunk_size"] = chunk_size

        # --- PDF Section (optional) ---
        if WEASYPRINT_AVAILABLE:
            with st.expander("📋  PDF Generation"):
                st.info("PDF export is available.")
        else:
            st.caption("⚠️ PDF export disabled (WeasyPrint not installed)")

        # --- Conversation Stats ---
        msg_count = len(st.session_state.get("messages", []))
        user_msgs = sum(1 for m in st.session_state.get("messages", []) if m["role"] == "user")
        st.markdown("---")
        st.markdown(
            f"""
            <div style="padding:4px 0;">
                <div style="color:var(--text-secondary); font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Session</div>
                <div style="color:var(--text-primary); font-size:0.82rem;">{user_msgs} question{'s' if user_msgs != 1 else ''} asked</div>
                <div style="color:var(--text-secondary); font-size:0.78rem;">{msg_count} messages total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# WELCOME CARD
# =============================================================================

def render_welcome():
    """Show a friendly welcome card when the chat is empty."""
    if len(st.session_state.get("messages", [])) > 0:
        return

    st.markdown(
        """
        <div class="welcome-card">
            <h2>👋  Welcome to Resume AI Chat</h2>
            <p>
                I'm an AI assistant that can answer questions about a resume
                stored in a Google Doc. I use <strong>retrieval-augmented generation (RAG)</strong>
                to find the most relevant sections and give you accurate answers.
            </p>
            <p style="margin-top:14px; color:var(--text-secondary); font-size:0.85rem;">
                Try asking something like:
            </p>
            <p style="margin-top:18px; font-size:0.82rem; color:var(--text-secondary);">
                Click a question below or type your own! 🚀
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

def render_footer():
    st.markdown(
        """
        <div class="footer-bar">
            Powered by
            <a href="https://python.langchain.com" target="_blank">LangChain</a> +
            <a href="https://www.trychroma.com" target="_blank">Chroma</a> +
            <a href="https://openrouter.ai" target="_blank">OpenRouter</a>
            &nbsp;|&nbsp; Hosted on
            <a href="https://huggingface.co/spaces" target="_blank">🤗 HF Spaces</a> &
            <a href="https://streamlit.io/cloud" target="_blank">Streamlit Cloud</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.1
    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = 4000

    # Draw sidebar
    render_sidebar()

    # --- Initialization Block ---
    if st.session_state.qa_chain is None:
        _placeholder = st.empty()

        with _placeholder.container():
            # Progress steps UI
            steps = [
                ("🔐", "Validating credentials..."),
                ("📥", "Loading resume from Google Docs..."),
                ("✂️", "Splitting text into chunks..."),
                ("🔢", "Building vector store..."),
                ("🤖", "Initializing AI chain..."),
            ]

            step_placeholders = []
            for icon, label in steps:
                sp = st.empty()
                sp.markdown(
                    f'<div class="init-step"><span class="icon">{icon}</span> {label}</div>',
                    unsafe_allow_html=True,
                )
                step_placeholders.append(sp)

        try:
            # Step 0 — Validate secrets
            doc_url = get_secret("RESUME_URL")
            openrouter_api_key = get_secret("OPENROUTER_API_KEY")

            if not all([doc_url, openrouter_api_key]):
                step_placeholders[0].markdown(
                    '<div class="init-step"><span class="icon">❌</span> Missing secrets</div>',
                    unsafe_allow_html=True,
                )
                st.error(
                    "❌ **Missing required secrets:** `RESUME_URL` or `OPENROUTER_API_KEY`. "
                    "Please configure them in your `.env` file or Streamlit secrets."
                )
                st.stop()

            step_placeholders[0].markdown(
                '<div class="init-step done"><span class="icon">✅</span> Credentials validated</div>',
                unsafe_allow_html=True,
            )

            # Step 1 — Fetch resume
            step_placeholders[1].markdown(
                '<div class="init-step active"><span class="icon">⏳</span> Loading resume from Google Docs...</div>',
                unsafe_allow_html=True,
            )
            resume_text = get_google_docs_content(doc_url)
            step_placeholders[1].markdown(
                '<div class="init-step done"><span class="icon">✅</span> Resume loaded</div>',
                unsafe_allow_html=True,
            )

            # Step 2 — Split
            step_placeholders[2].markdown(
                '<div class="init-step active"><span class="icon">⏳</span> Splitting text into chunks...</div>',
                unsafe_allow_html=True,
            )
            documents = [Document(page_content=resume_text)]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.get("chunk_size", 500),
                chunk_overlap=50,
            )
            chunks = text_splitter.split_documents(documents)
            step_placeholders[2].markdown(
                f'<div class="init-step done"><span class="icon">✅</span> {len(chunks)} chunks created</div>',
                unsafe_allow_html=True,
            )

            # Step 3 — Vector store
            step_placeholders[3].markdown(
                '<div class="init-step active"><span class="icon">⏳</span> Building vector store...</div>',
                unsafe_allow_html=True,
            )
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
            st.session_state.vectorstore = vectorstore
            retriever = vectorstore.as_retriever()
            step_placeholders[3].markdown(
                '<div class="init-step done"><span class="icon">✅</span> Vector store ready</div>',
                unsafe_allow_html=True,
            )

            # Step 4 — AI chain
            step_placeholders[4].markdown(
                '<div class="init-step active"><span class="icon">⏳</span> Initializing AI chain...</div>',
                unsafe_allow_html=True,
            )
            llm = get_llm(temperature=st.session_state.get("temperature", 0.1))

            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the following context and conversation history.

Context from resume:
{context}

Conversation history:
{history}

Question: {input}"""
            )

            # Helper function to format documents
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            def get_history():
                messages = st.session_state.get("messages", [])
                return "\n".join(
                    f"User: {m['content']}"
                    if m["role"] == "user"
                    else f"Assistant: {m['content']}"
                    for m in messages
                )

            # Build chain manually (LCEL)
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "history": lambda x: get_history(),
                    "input": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            st.session_state.qa_chain = rag_chain
            step_placeholders[4].markdown(
                '<div class="init-step done"><span class="icon">✅</span> AI chain ready</div>',
                unsafe_allow_html=True,
            )

            # Clear the progress placeholder after a beat
            time.sleep(0.6)
            _placeholder.empty()
            st.toast("✅ Ready! Ask me anything about the resume.", icon="🚀")

        except Exception as e:
            _placeholder.empty()
            st.error(
                f"❌ **Initialization Error:**\n\n```\n{e}\n```\n\n"
                "Make sure your Google Doc is shared with the Service Account email."
            )
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown(
                    """
                    - **Google Doc sharing:** Ensure the doc is shared with the service account email.
                    - **API Key:** Verify `OPENROUTER_API_KEY` is set correctly.
                    - **Service Account:** Check that `service_account_json` or `service_account_path` is configured.
                    - **Doc URL:** The `RESUME_URL` must be a valid Google Docs link.
                    """
                )
            st.stop()

    # --- Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Welcome Card ---
    render_welcome()

    # --- Quick Question Buttons ---
    if len(st.session_state.get("messages", [])) == 0:
        questions = [
            "What is the work experience?",
            "What skills are listed?",
            "Summarize the education",
            "What projects have they built?",
            "List all certifications",
        ]
        cols = st.columns(len(questions))
        for i, q in enumerate(questions):
            with cols[i]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state["pending_query"] = q
                    st.rerun()

    # Process a pending quick-question click
    if st.session_state.get("pending_query"):
        prompt = st.session_state.pop("pending_query")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response_text = "⚠️ An error occurred."
        with st.chat_message("assistant"):
            typing = st.empty()
            typing.markdown(
                '<div class="typing-indicator"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )
            try:
                response_placeholder = st.empty()
                full_response = ""
                for chunk in st.session_state.qa_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                response_text = full_response if full_response.strip() else "Empty response."
                typing.empty()
            except Exception as e:
                typing.empty()
                response_text = f"❌ **Error:** {e}"
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # --- Chat Input ---
    if prompt := st.chat_input("Ask me anything about the resume... 💬"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Generate Assistant Response ---
        response_text = "⚠️ An error occurred, and no response was generated."

        with st.chat_message("assistant"):
            # Typing indicator
            typing = st.empty()
            typing.markdown(
                '<div class="typing-indicator"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )

            try:
                if st.session_state.qa_chain is None:
                    raise ValueError(
                        "The AI Chain is not initialized. Please reload the page."
                    )

                # Stream the response token by token
                response_placeholder = st.empty()
                full_response = ""

                for chunk in st.session_state.qa_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                # Final render without cursor
                response_placeholder.markdown(full_response)

                if full_response.strip():
                    response_text = full_response
                else:
                    response_text = "The model returned an empty response. Please check your API key and model configuration."

                typing.empty()

            except Exception as e:
                typing.empty()
                response_text = f"❌ **Error:** {e}"
                st.markdown(response_text)

        # Save to history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

    # --- Footer ---
    render_footer()


if __name__ == "__main__":
    main()
