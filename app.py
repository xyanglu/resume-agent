import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_classic.chains import RetrievalQA
import os
import tempfile
from weasyprint import HTML
import markdown

# Page configuration
st.set_page_config(
    page_title="RAG Resume Chatbot",
    page_icon="📄",
    layout="wide"
)

def extract_dates(resume_context):
    """Extract date ranges from resume context"""
    import re
    date_patterns = [
        r'\d{4}',  # Years like 2020
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s*\d{4}',  # Jan 2020
        r'(?:Present|Current)',  # Current positions
    ]
    dates_found = []
    for pattern in date_patterns:
        dates_found.extend(re.findall(pattern, resume_context, re.IGNORECASE))
    return list(set(dates_found))

def generate_resume_pdf(job_description,company_name):
    """Generate a tailored resume PDF based on job description."""
    # Get resume content from RAG
    docs = st.session_state.vectorstore.similarity_search(
        "education skills experience work history qualifications", 
        k=15  # Increase to get more context
    )
    resume_context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate tailored resume
    prompt = f"""
    You are a professional resume writer. Create an ULTRA-CONCISE 1-page resume for {company_name if company_name else "this company"}.

    CRITICAL RULES - DO NOT BREAK THESE:
    1. FIND and EXTRACT the candidate's NAME from the resume - MUST be at the very top
    2. ONLY use information from the resume below
    3. If dates are not in the resume, DO NOT make them up - use "Present" or omit
    4. DO NOT invent job titles, companies, or durations
    5. DO NOT add skills not listed in the resume
    6. Keep work experiences exactly as they appear in the resume
    7. If information is missing, leave it out rather than inventing

    Resume:
    {resume_context}

    Job Description:
    {job_description}

    REQUIREMENTS - STRICTLY FOLLOW (MUST FIT ON 1 PAGE):
    1. HEADER: "# [Full Name]" - extract name from resume context
    2. Professional Summary: Exactly 2 sentences, 25 words max
    3. Skills: Exactly 5 bullet points, 5-7 words each
    4. Experience: Only TOP 2 most relevant positions, 2 bullet points each, 8-10 words per bullet
    5. Education: Degree and school only, 10 words max
    6. ABSOLUTE LIMIT: Total output under 1500 characters (not words!)
    7. DELETE everything else - keep only essential info
    8. Each bullet point must be under 10 words
    9. No fluff, no filler words, no descriptions

    Output in clean Markdown format (MUST be under 1500 characters total):
    # [CANDIDATE FULL NAME]
    [2 sentences, 25 words max]

    ## Skills
    - [skill 1 - 7 words max]
    - [skill 2 - 7 words max]
    - [skill 3 - 7 words max]
    - [skill 4 - 7 words max]
    - [skill 5 - 7 words max]

    ## Experience
    ### [Most Recent Position]
    **[Company Name]** | [Dates]
    - [achievement 1 - 10 words max]
    - [achievement 2 - 10 words max]

    ### [Previous Position]
    **[Company Name]** | [Dates]
    - [achievement 1 - 10 words max]
    - [achievement 2 - 10 words max]

    ## Education
    **[Degree]** from [School]
    """
    
    dates_in_resume = extract_dates(resume_context)
    prompt += f"""
    DATES FOUND IN RESUME: {', '.join(dates_in_resume)}
    ONLY use these dates - do not invent others.
    """

    llm = ChatOpenAI(
        model="glm-4.7-flash",
        openai_api_key=os.getenv("ZAI_API_KEY"),
        openai_api_base="https://api.z.ai/api/paas/v4/",
        temperature=0.3
    )
    
    response = llm.invoke(prompt)
    resume_markdown = response.content
    
    # Convert to HTML then PDF
    html_content = markdown_to_html(resume_markdown, "resume")
    pdf_bytes = HTML(string=html_content).write_pdf()
    
    return pdf_bytes

def generate_cover_letter_pdf(job_description,company_name):
    """Generate a cover letter PDF based on job description."""
    # Get resume content from RAG
    docs = st.session_state.vectorstore.similarity_search("entire resume summary and all experience", k=10)
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
    
    llm = ChatOpenAI(
        model="glm-4.7-flash",
        openai_api_key=os.getenv("ZAI_API_KEY"),
        openai_api_base="https://api.z.ai/api/paas/v4/",
        temperature=0.3
    )
    
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

# Title and description
st.title("📄 RAG Resume Chatbot")
st.markdown("""
This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions about your resume.
It loads your resume from Google Docs, creates embeddings, and uses Z.AI's GLM-4.7-Flash.
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def extract_text_from_doc(doc):
    """Extract plain text from Google Docs API response"""
    text = []
    content = doc.get('body').get('content')
    
    for element in content:
        if 'paragraph' in element:
            paragraph = element['paragraph']
            for elem in paragraph.get('elements', []):
                if 'textRun' in elem:
                    text.append(elem['textRun']['content'])
        elif 'table' in element:
            table = element['table']
            for row in table.get('tableRows', []):
                for cell in row.get('tableCells', []):
                    for elem in cell.get('content', []):
                        if 'paragraph' in elem:
                            for p_elem in elem['paragraph'].get('elements', []):
                                if 'textRun' in p_elem:
                                    text.append(p_elem['textRun']['content'])
    
    return ''.join(text)

# Auto-initialize on startup
if st.session_state.qa_chain is None:
    try:
        # Check required secrets
        doc_url = os.getenv("RESUME_URL")
        service_account_json = os.getenv("service_account_json")
        zai_api_key = os.getenv("ZAI_API_KEY")
        
        if not all([doc_url, service_account_json, zai_api_key]):
            st.error("❌ Missing required secrets. Please set: RESUME_URL, service_account_json, ZAI_API_KEY")
            st.stop()
        
        with st.spinner("📄 Loading your resume from Google Docs..."):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(service_account_json)
                creds_path = f.name
            
            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/documents.readonly']
            )
            
            doc_id = doc_url.split('/d/')[1].split('/')[0]
            service = build('docs', 'v1', credentials=creds)
            doc = service.documents().get(documentId=doc_id).execute()
            
            content = extract_text_from_doc(doc)
            documents = [Document(page_content=content)]
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
                model="glm-4.7-flash",
                openai_api_key=zai_api_key,
                openai_api_base="https://api.z.ai/api/paas/v4/",
                temperature=0.7
            )
            
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=retriever
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


# 2. PDF generation expander 
st.header("📄 Customize Your Application")

with st.expander("Generate Custom Resume & Cover Letter", expanded=False):
    st.write("Paste a job description to get a tailored resume and cover letter.")
    
    company_name = st.text_input(
        "Company Name (optional)",
        placeholder="e.g., Google, Microsoft, Amazon"
    )
    
    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder="Paste the job description here..."
    )
    
    if st.button("Generate PDF Documents", type="primary"):
        if job_description:
            with st.spinner("📄 Generating your customized documents..."):
                # Generate PDFs with company name
                resume_pdf = generate_resume_pdf(job_description, company_name)
                cover_letter_pdf = generate_cover_letter_pdf(job_description, company_name)
                
                # Save to session state
                st.session_state.resume_pdf = resume_pdf
                st.session_state.cover_letter_pdf = cover_letter_pdf
                st.session_state.company_name = company_name

    # Show download buttons if PDFs exist
    if "resume_pdf" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            # Use company name in filename
            company = st.session_state.company_name or "custom"
            st.download_button(
                label="📥 Download Custom Resume",
                data=st.session_state.resume_pdf,
                file_name=f"{company}_resume.pdf",
                mime="application/pdf"
            )
        with col2:
            st.download_button(
                label="📥 Download Cover Letter",
                data=st.session_state.cover_letter_pdf,
                file_name=f"{company}_cover_letter.pdf",
                mime="application/pdf"
            )

# Footer
st.divider()
st.markdown("""
---
**Made with ❤️ using Streamlit, LangChain, and HuggingFace**

- 📄 Resume loaded from Google Docs
- 🔢 Embeddings: sentence-transformers/all-MiniLM-L6-v2
- 🤖 AI Model: Z.AI (glm-4.7-flash)
""")