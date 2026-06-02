import streamlit as st
import streamlit_analytics
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import tempfile
from weasyprint import HTML
import markdown

# Vision review imports
import base64
import requests

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

streamlit_analytics.start_tracking()


def get_llm(temperature=0.1):
    return ChatOpenAI(
        model="openrouter/free",
        api_key=st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


# Page configuration
st.set_page_config(
    page_title="RAG Resume Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def review_pdf_with_vision(pdf_bytes, doc_type="resume"):
    """Convert PDF to image, send to OpenRouter vision model for layout review."""
    if not PYMUPDF_AVAILABLE:
        return "⚠️ Vision review skipped: PyMuPDF not installed."

    # Convert all PDF pages to PNG images
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_b64 = []
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images_b64.append(base64.b64encode(img_bytes).decode())
    doc.close()

    # Build the prompt based on doc type
    if doc_type == "cover_letter":
        review_prompt = """Review this cover letter for formatting and layout quality. Check:
1. Is ALL text fully visible with no cutoff, overlap, or crowding?
2. Is the professional letter format correct (date, address, salutation, body paragraphs, closing)?
3. Are margins and spacing consistent and professional?
4. Is the font size readable and consistent throughout?
5. Does it look like a clean, professional business letter?

Rate the layout 1-10. List EVERY specific issue found, even minor ones.
Format your response as:
**Rating: X/10**
**Issues:**
- [issue 1]
**Strengths:**
- [strength 1]"""
    else:
        review_prompt = """Review this resume for formatting and layout quality. Check:
1. Is ALL text fully visible with no right-side cutoff or overlap between elements?
2. Is the visual hierarchy clear (name prominent, section headers distinct, job titles stand out)?
3. Are dates and job titles properly aligned without overlapping each other?
4. Is spacing consistent between sections, bullets, and paragraphs?
5. Is the resume ATS-friendly (clean text, standard fonts, no columns/tables)?

Rate the layout 1-10. List EVERY specific issue found, even minor ones.
Format your response as:
**Rating: X/10**
**Issues:**
- [issue 1]
**Strengths:**
- [strength 1]"""

    # Build message content with all pages
    content_parts = []
    for img_b64 in images_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    content_parts.append({"type": "text", "text": review_prompt})

    # Call OpenRouter vision API
    api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
    vision_model = st.secrets.get("VISION_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                ],
                "max_tokens": 1000,
            },
            timeout=60,
        )

        result = response.json()

        if "error" in result:
            return f"⚠️ Vision review error: {result['error'].get('message', result['error'])}"

        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Vision review failed: {e}"


def extract_dates(resume_context):
    """Extract date ranges from resume context"""
    import re

    date_patterns = [
        r"\d{4}",
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s*\d{4}",
        r"(?:Present|Current)",
    ]
    dates_found = []
    for pattern in date_patterns:
        dates_found.extend(re.findall(pattern, resume_context, re.IGNORECASE))
    return list(set(dates_found))


def generate_resume_pdf(job_description, company_name):
    """Generate a tailored resume PDF based on job description."""
    docs = st.session_state.vectorstore.similarity_search(
        "education skills experience work history qualifications",
        k=15,
    )
    resume_context = "\n\n".join([doc.page_content for doc in docs])

    name_prompt = f"""
    Extract ONLY candidate's full name from this resume context.
    Return ONLY name, nothing else.
    Resume:
    {resume_context}
    """

    llm = get_llm(temperature=0.1)

    name_response = llm.invoke(name_prompt)
    candidate_name = name_response.content.strip()
    if len(candidate_name.split()) > 5:
        candidate_name = " ".join(candidate_name.split()[:2])
    candidate_name = candidate_name.replace("#", "").replace("*", "").strip()

    prompt = f"""
    Create a professional 1-page resume for {company_name if company_name else "this company"}.

    CANDIDATE NAME: {candidate_name}
    USE THIS EXACT NAME AT THE TOP

    Resume Context:
    {resume_context}

    Job Description:
    {job_description}

    REQUIREMENTS:
    1. Start with: # {{candidate_name}}
    2. Summary: 2-3 sentences highlighting fit for this specific role
    3. Skills: group by category (Languages, Frameworks, Tools, Cloud), keep only skills relevant to the JD
    4. Experience: up to 3 most relevant positions, 2-3 bullets each starting with strong action verbs
    5. Education: 1-2 lines
    6. TOTAL LIMIT: Under 3000 characters
    7. No filler words (leverage, utilize, synergize, streamline, facilitate)
    8. Use standard ATS-friendly format with clear section headers
    9. Be honest - only include real experience from the resume context

    OUTPUT FORMAT (Markdown):
    # {candidate_name}
    [Contact info line]

    ## Summary
    [2-3 sentences]

    ## Skills
    - **Languages**: [list]
    - **Frameworks**: [list]
    - **Tools**: [list]

    ## Experience
    ### [Position]
    **[Company]** | [Dates]
    - [Action verb] [specific achievement with metrics if available]
    - [Action verb] [specific achievement]

    ## Education
    [Degree] from [School]

    ## Certifications
    [If any relevant]
    """

    dates_in_resume = extract_dates(resume_context)
    if dates_in_resume:
        prompt += f"\nDATES: {', '.join(dates_in_resume[:3])}"

    response = llm.invoke(prompt)
    resume_markdown = response.content.strip()

    if not resume_markdown.startswith("#"):
        resume_markdown = f"# {candidate_name}\n\n{resume_markdown}"

    if len(resume_markdown) > 3000:
        resume_markdown = resume_markdown[:3000]

    html_content = markdown_to_html(resume_markdown, "resume")
    pdf_bytes = HTML(string=html_content).write_pdf()

    return pdf_bytes, resume_markdown


def generate_cover_letter_pdf(job_description, company_name):
    """Generate a cover letter PDF based on job description."""
    docs = st.session_state.vectorstore.similarity_search(
        "entire resume summary and all experience", k=10
    )
    resume_context = "\n\n".join([doc.page_content for doc in docs])

    name_prompt = f"""
    Extract ONLY candidate's full name from this resume context.
    Return ONLY name, nothing else.
    Resume:
    {resume_context}
    """

    llm = get_llm(temperature=0.1)
    name_response = llm.invoke(name_prompt)
    candidate_name = name_response.content.strip().replace("#", "").replace("*", "").strip()
    if len(candidate_name.split()) > 5:
        candidate_name = " ".join(candidate_name.split()[:2])

    prompt = f"""
    Write a professional cover letter for {company_name if company_name else "this company"} based on this resume and job description.
    
    Candidate Name: {candidate_name}
    
    Resume:
    {resume_context}
    
    Job Description:
    {job_description}
    
    Create a compelling cover letter that:
    1. Has a proper business letter format (date, company address placeholder, salutation, body, closing with name)
    2. Addresses the specific role and company
    3. Highlights 2-3 key qualifications that match the job
    4. Shows enthusiasm and fit for the role
    5. Is professional and concise (300-400 words)
    6. Uses information from the resume but tailors it to this job
    
    Output in clean Markdown format with the letter structure.
    """

    llm = get_llm(temperature=0.3)
    response = llm.invoke(prompt)
    letter_markdown = response.content

    html_content = markdown_to_html(letter_markdown, "cover_letter")
    pdf_bytes = HTML(string=html_content).write_pdf()

    return pdf_bytes


def markdown_to_html(markdown_content, doc_type):
    """Convert markdown to HTML with styling."""
    if doc_type == "resume":
        template = """
        <html>
        <head>
            <style>
                body {{ 
                    font-family: 'Arial', sans-serif; 
                    max-width: 700px; 
                    margin: 0 auto; 
                    padding: 20px 30px;
                    font-size: 9pt;
                    line-height: 1.2;
                }}
                h1 {{ 
                    color: #2c3e50; 
                    font-size: 14pt;
                    font-weight: bold;
                    border-bottom: 2px solid #3498db; 
                    padding-bottom: 5px;
                    margin-bottom: 10px;
                }}
                h2 {{ 
                    color: #2c3e50; 
                    font-size: 10pt;
                    font-weight: bold;
                    text-transform: uppercase;
                    margin-top: 12px;
                    margin-bottom: 5px;
                }}
                h3 {{ 
                    color: #34495e; 
                    font-size: 9pt;
                    font-weight: bold;
                    margin-top: 8px;
                    margin-bottom: 3px;
                }}
                p {{ 
                    margin: 3px 0;
                    line-height: 1.2;
                }}
                ul {{ 
                    line-height: 1.2; 
                    margin: 3px 0;
                    padding-left: 15px;
                }}
                li {{ 
                    margin-bottom: 2px;
                    font-size: 9pt;
                }}
                strong {{
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """
    else:  # cover_letter
        template = """
        <html>
        <head>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    max-width: 600px; 
                    margin: 0 auto; 
                    padding: 40px; 
                    line-height: 1.6;
                    font-size: 11pt;
                }}
                p {{ margin-bottom: 15px; }}
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """

    html = markdown.markdown(markdown_content)
    return template.format(content=html)


# Sidebar - PDF Generation
with st.sidebar:
    st.header("📋 PDF Generation")

    with st.expander("Generate Custom Resume & Cover Letter", expanded=False):
        st.write("Paste a job description to get a tailored resume and cover letter.")

        company_name = st.text_input(
            "Company Name (optional)", placeholder="e.g., Google, Microsoft, Amazon"
        )

        job_description = st.text_area(
            "Job Description", height=200, placeholder="Paste job description here..."
        )

        if st.button("Generate PDF Documents", type="primary"):
            if job_description:
                # Step 1: Generate PDFs
                with st.spinner("📄 Generating your customized documents..."):
                    resume_pdf, resume_md = generate_resume_pdf(
                        job_description, company_name
                    )
                    cover_letter_pdf = generate_cover_letter_pdf(
                        job_description, company_name
                    )

                    st.session_state.resume_pdf = resume_pdf
                    st.session_state.resume_md = resume_md
                    st.session_state.cover_letter_pdf = cover_letter_pdf
                    st.session_state.company_name = company_name

                # Step 2: Multi-model eval (parallel)
                with st.spinner("🔬 Running multi-model resume evaluation (3 models)..."):
                    from resume_eval import evaluate_resume, format_eval_report
                    eval_result = evaluate_resume(
                        resume_md, job_description,
                        api_key=st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
                    )
                    st.session_state.eval_result = eval_result
                    st.session_state.eval_report = format_eval_report(eval_result)

                # Step 3: Auto-run vision review on both
                if PYMUPDF_AVAILABLE:
                    with st.spinner("👁️ Running AI vision review on both documents..."):
                        st.session_state.resume_review = review_pdf_with_vision(
                            resume_pdf, "resume"
                        )
                        st.session_state.cover_review = review_pdf_with_vision(
                            cover_letter_pdf, "cover_letter"
                        )
                else:
                    st.session_state.resume_review = "⚠️ Vision review skipped (PyMuPDF not available)."
                    st.session_state.cover_review = "⚠️ Vision review skipped (PyMuPDF not available)."

                st.success("✅ Documents generated and reviewed!")

        # Show download + reviews if PDFs exist
        if "resume_pdf" in st.session_state:
            st.divider()
            st.write("📥 Download your documents:")
            st.download_button(
                label="📄 Download Resume",
                data=st.session_state.resume_pdf,
                file_name=f"{st.session_state.company_name or 'custom'}_resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.download_button(
                label="✉️ Download Cover Letter",
                data=st.session_state.cover_letter_pdf,
                file_name=f"{st.session_state.company_name or 'custom'}_cover_letter.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            # Vision review results (auto-generated)
            st.divider()
            st.markdown("### 👁️ AI Vision Review")

            if st.session_state.get("resume_review"):
                with st.expander("📊 Resume Review", expanded=True):
                    st.markdown(st.session_state.resume_review)

            if st.session_state.get("cover_review"):
                with st.expander("📊 Cover Letter Review", expanded=True):
                    st.markdown(st.session_state.cover_review)

            # Multi-model eval report
            if st.session_state.get("eval_report"):
                st.divider()
                st.markdown(st.session_state.eval_report)

    st.divider()
    st.markdown(
        """
**ℹ️ About**

This app uses:
- 📄 Google Docs API
- 🔢 Vector embeddings
- 🤖 OpenRouter (Free)
- 👁️ AI Vision Review (automatic)
    """
    )

# Main area - Title and description
st.title("📄 RAG Resume Chatbot")
st.markdown(
    """
This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions about your resume.
It loads your resume from Google Docs, creates embeddings, and uses OpenRouter (Llama 3.1).

👈 Check the **sidebar** (☰ menu at top-left) for **PDF generation tools**!
"""
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


def extract_text_from_doc(doc):
    """Extract plain text from Google Docs API response"""
    text = []
    content = doc.get("body").get("content")

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


# Auto-initialize on startup
if st.session_state.qa_chain is None:
    try:
        doc_url = st.secrets.get("RESUME_URL", os.getenv("RESUME_URL"))
        service_account_json = st.secrets.get(
            "service_account_json", os.getenv("service_account_json")
        )
        openrouter_api_key = st.secrets.get(
            "OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY")
        )

        if not all([doc_url, service_account_json, openrouter_api_key]):
            st.error(
                "❌ Missing required secrets. Please set: RESUME_URL, service_account_json, OPENROUTER_API_KEY"
            )
            st.stop()

        with st.spinner("📄 Loading your resume from Google Docs..."):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(service_account_json)
                creds_path = f.name

            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/documents.readonly"],
            )

            doc_id = doc_url.split("/d/")[1].split("/")[0]
            service = build("docs", "v1", credentials=creds)
            doc = service.documents().get(documentId=doc_id).execute()

            content = extract_text_from_doc(doc)
            documents = [Document(page_content=content)]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            chunks = text_splitter.split_documents(documents)

        with st.spinner("🔢 Creating embeddings and vector store..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            st.session_state.vectorstore = vectorstore

        with st.spinner("🤖 Loading AI model..."):
            llm = ChatOpenAI(
                model="openrouter/free",
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7,
            )

            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the following context and conversation history.

Context from resume:
{context}

Conversation history:
{history}

Question: {input}"""
            )

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

            st.session_state.qa_chain = (
                {
                    "context": retriever | format_docs,
                    "history": lambda x: get_history(),
                    "input": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

        st.success("✅ Resume loaded! You can now ask questions.")

    except Exception as e:
        st.error(f"❌ Initialization error: {str(e)}")
        st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(
    "Copy and paste a job description or Ask a question about the resume!"
):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown(
    """
---
**Made with ❤️ using Streamlit, LangChain, and OpenRouter**

- 📄 Resume loaded from Google Docs
- 🔢 Embeddings: sentence-transformers/all-MiniLM-L6-v2
- 🤖 AI Model: OpenRouter (Llama 3.1)
- 👁️ Vision Review: OpenRouter Vision (Free, automatic)
"""
)

streamlit_analytics.stop_tracking()
