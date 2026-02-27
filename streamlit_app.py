import streamlit as st
import os
import re
import markdown
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# FIX: Use standard langchain.chains instead of langchain_classic
from langchain_community.chains import RetrievalQA

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

# Page Config
st.set_page_config(page_title="RAG Resume Chatbot", page_icon="📄", layout="wide")

# --- Core Functions ---

def get_llm(temperature=0.1):
    return ChatOpenAI(
        model=get_secret("MODEL_NAME", "glm-4.7-flash"),
        api_key=get_secret("ZAI_API_KEY"),
        base_url="https://api.z.ai/api/paas/v4/",
        temperature=temperature,
    )

def extract_text_from_doc(doc):
    """Extract plain text from Google Docs API response"""
    text = []
    content = doc.get("body").get("content")
    if not content: return ""
    
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
    match = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
    if not match:
        raise ValueError("Invalid Google Doc URL format")
    doc_id = match.group(1)

    # 2. Load Credentials directly from Secrets (No temp file needed)
    # st.secrets["service_account_json"] is already a dictionary
    creds_dict = dict(st.secrets["service_account_json"])
    
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, 
        scopes=["https://www.googleapis.com/auth/documents.readonly"]
    )

    # 3. Build Service and Fetch
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()
    
    return extract_text_from_doc(doc)

# --- PDF Generation (Optional) ---

def markdown_to_html(content, doc_type):
    # (Your existing markdown_to_html function remains unchanged)
    # ... omitted for brevity, include your original function here ...
    pass 

# --- Main App Logic ---

def main():
    st.title("📄 RAG Resume Chatbot")
    
    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar
    with st.sidebar:
        if WEASYPRINT_AVAILABLE:
            st.header("📋 PDF Generation")
            # ... (Your PDF generation UI code) ...
        else:
            st.warning("⚠️ PDF generation disabled (WeasyPrint missing). Chat works fine!")
        
        st.divider()
        st.markdown(f"**Model:** {get_secret('MODEL_NAME', 'glm-4.7-flash')}")

    # --- Initialization Block ---
    if st.session_state.qa_chain is None:
        try:
            doc_url = get_secret("RESUME_URL")
            zai_api_key = get_secret("ZAI_API_KEY")
            
            # Validate essential secrets
            if not all([doc_url, zai_api_key]):
                st.error("❌ Missing secrets: RESUME_URL or ZAI_API_KEY")
                st.stop()

            with st.spinner("🔄 Loading Resume from Google Docs..."):
                # Fetch Text
                resume_text = get_google_docs_content(doc_url)
                documents = [Document(page_content=resume_text)]

                # Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(documents)

            with st.spinner("🔢 Creating Vector Store..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
                st.session_state.vectorstore = vectorstore # Save for PDF generation
                
            with st.spinner("🤖 Initializing AI..."):
                llm = get_llm()
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm, 
                    retriever=vectorstore.as_retriever()
                )

            st.success("✅ Ready! Ask me about the resume.")

        except Exception as e:
            st.error(f"❌ Initialization Error: {e}")
            st.info("Ensure your Google Doc is shared with the Service Account email.")
            st.stop()

    # --- Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the resume..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(prompt)
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()