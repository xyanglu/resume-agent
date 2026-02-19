import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_classic.chains import RetrievalQA
import os
import tempfile
from weasyprint import HTML
import markdown


def get_llm(temperature=0.1):
    if os.getenv("USE_GEMINI", "false").lower() == "true":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )
    else:
        return ChatOpenAI(
            model=os.getenv("MODEL_NAME", "glm-4.7-flash"),
            api_key=os.getenv("ZAI_API_KEY"),
            base_url="https://api.z.ai/api/paas/v4/",
            temperature=temperature,
        )


# Page configuration
st.set_page_config(
    page_title="RAG Resume Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def extract_dates(resume_context):
    """Extract date ranges from resume context"""
    import re

    date_patterns = [
        r"\d{4}",  # Years like 2020
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s*\d{4}",  # Jan 2020
        r"(?:Present|Current)",  # Current positions
    ]
    dates_found = []
    for pattern in date_patterns:
        dates_found.extend(re.findall(pattern, resume_context, re.IGNORECASE))
    return list(set(dates_found))


def generate_resume_pdf(job_description, company_name):
    """Generate a tailored resume PDF based on job description."""
    # Get resume content from RAG
    docs = st.session_state.vectorstore.similarity_search(
        "education skills experience work history qualifications",
        k=15,  # Increase to get more context
    )
    resume_context = "\n\n".join([doc.page_content for doc in docs])

    # Step 1: Extract name first
    name_prompt = f"""
    Extract ONLY candidate's full name from this resume context.
    Return ONLY name, nothing else.
    Resume:
    {resume_context}
    """

    llm = get_llm(temperature=0.1)

    name_response = llm.invoke(name_prompt)
    candidate_name = name_response.content.strip()
    # Clean up name response
    if len(candidate_name.split()) > 5:  # If too many words, take first 2
        candidate_name = " ".join(candidate_name.split()[:2])
    candidate_name = candidate_name.replace("#", "").replace("*", "").strip()

    # Step 2: Generate resume with extracted name
    prompt = f"""
    Create a 1-page resume for {company_name if company_name else "this company"}.

    CANDIDATE NAME: {candidate_name}
    USE THIS EXACT NAME AT THE TOP

    Resume Context:
    {resume_context}

    Job Description:
    {job_description}

    STRICT REQUIREMENTS:
    1. Start with: # {candidate_name}
    2. Summary: 2 sentences, 20 words max
    3. Skills: 5 bullet points, 5 words each
    4. Experience: 1 job only, 2 bullets, 8 words each
    5. Education: 1 line, 10 words max
    6. TOTAL LIMIT: Under 800 characters
    7. No filler, no descriptions

    OUTPUT FORMAT:
    # {candidate_name}
    [20-word summary]

    ## Skills
    - [5-word skill]
    - [5-word skill]
    - [5-word skill]
    - [5-word skill]
    - [5-word skill]

    ## Experience
    ### [Position]
    **[Company]** | [Dates]
    - [8-word bullet]
    - [8-word bullet]

    ## Education
    [Degree] from [School]
    """

    dates_in_resume = extract_dates(resume_context)
    if dates_in_resume:
        prompt += f"\nDATES: {', '.join(dates_in_resume[:3])}"

    response = llm.invoke(prompt)
    resume_markdown = response.content.strip()

    # Ensure name is at the top
    if not resume_markdown.startswith("#"):
        resume_markdown = f"# {candidate_name}\n\n{resume_markdown}"

    # Truncate if too long
    if len(resume_markdown) > 800:
        resume_markdown = resume_markdown[:800]

    # Convert to HTML then PDF
    html_content = markdown_to_html(resume_markdown, "resume")
    pdf_bytes = HTML(string=html_content).write_pdf()

    return pdf_bytes


def generate_cover_letter_pdf(job_description, company_name):
    """Generate a cover letter PDF based on job description."""
    # Get resume content from RAG
    docs = st.session_state.vectorstore.similarity_search(
        "entire resume summary and all experience", k=10
    )
    resume_context = "\n\n".join([doc.page_content for doc in docs])

    # Generate cover letter
    prompt = f"""
    Write a professional cover letter for {company_name if company_name else "this company"} based on this resume and job description.
    
    Resume:
    {resume_context}
    
    Job Description:
    {job_description}
    
    Create a compelling cover letter that:
    1. Addresses the specific role and company
    2. Highlights 2-3 key qualifications that match the job
    3. Shows enthusiasm and fit for the role
    4. Is professional and concise (300-400 words)
    5. Uses information from the resume but tailors it to this job
    
    Output in clean Markdown format.
    """

    llm = get_llm(temperature=0.3)

    response = llm.invoke(prompt)
    letter_markdown = response.content

    # Convert to HTML then PDF
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
                with st.spinner("📄 Generating your customized documents..."):
                    # Generate PDFs with company name
                    resume_pdf = generate_resume_pdf(job_description, company_name)
                    cover_letter_pdf = generate_cover_letter_pdf(
                        job_description, company_name
                    )

                    # Save to session state
                    st.session_state.resume_pdf = resume_pdf
                    st.session_state.cover_letter_pdf = cover_letter_pdf
                    st.session_state.company_name = company_name
                    st.success("✅ PDFs generated successfully!")

        # Show download buttons if PDFs exist
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

    st.divider()
    model = os.getenv("MODEL_NAME", "glm-4.7-flash")
    st.markdown(f"""
    **ℹ️ About**

    This app uses:
    - 📄 Google Docs API
    - 🔢 Vector embeddings
    - 🤖 Z.AI {model}
    """)

# Main area - Title and description
st.title("📄 RAG Resume Chatbot")
model = os.getenv("MODEL_NAME", "glm-4.7-flash")
st.markdown(f"""
This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions about your resume.
It loads your resume from Google Docs, creates embeddings, and uses Z.AI's {model}.

👈 Check the **sidebar** (☰ menu at top-left) for **PDF generation tools**!
""")

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
        # Check required secrets
        doc_url = os.getenv("RESUME_URL")
        service_account_json = os.getenv("service_account_json")
        zai_api_key = os.getenv("ZAI_API_KEY")

        if not all([doc_url, service_account_json, zai_api_key]):
            st.error(
                "❌ Missing required secrets. Please set: RESUME_URL, service_account_json, ZAI_API_KEY"
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
            if os.getenv("USE_GEMINI", "false").lower() == "true":
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.7,
                )
            else:
                llm = ChatOpenAI(
                    model=os.getenv("MODEL_NAME", "glm-4.7-flash"),
                    api_key=zai_api_key,
                    base_url="https://api.z.ai/api/paas/v4/",
                    temperature=0.7,
                )

            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, retriever=retriever
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
if prompt := st.chat_input("Ask a question about the resume..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.run(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
model = os.getenv("MODEL_NAME", "glm-4.7-flash")
st.markdown(f"""
---
**Made with ❤️ using Streamlit, LangChain, and Z.AI**

- 📄 Resume loaded from Google Docs
- 🔢 Embeddings: sentence-transformers/all-MiniLM-L6-v2
- 🤖 AI Model: Z.AI ({model})
""")
