import os
import requests
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Page configuration
st.set_page_config(
    page_title="Resume Agent Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_document_content():
    """Retrieves content from a public Google Doc using the export feature."""
    DOCUMENT_ID = os.getenv('DOCUMENT_ID')
    
    if not DOCUMENT_ID:
        st.error("DOCUMENT_ID environment variable not set")
        return None
    
    export_url = f"https://docs.google.com/document/d/{DOCUMENT_ID}/export?format=txt"

    try:
        with st.spinner("Fetching resume content from Google Docs..."):
            response = requests.get(export_url)
            response.raise_for_status()

            full_text = response.text.strip()
            st.success("Resume content loaded successfully!")
            return full_text

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching document: {e}")
        return None

def initialize_llm():
    """Initialize the LLM based on configuration."""
    if os.getenv("USE_GEMINI", "false").lower() == "true":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("GOOGLE_API_KEY environment variable not set")
            return None
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            streaming=True
        )
    else:
        api_key = os.getenv('ZAI_API_KEY')
        model_name = os.getenv('MODEL_NAME', 'glm-4.7-flash')

        if not api_key:
            st.error("ZAI_API_KEY environment variable not set")
            return None

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.z.ai/api/paas/v4/",
            streaming=True
        )

def generate_response(message, llm, resume_text):
    """Generate AI response with streaming."""
    prompt_template = os.getenv('PROMPT_TEMPLATE')
    
    if not prompt_template:
        prompt_template = """You are a representative for the candidate. Answer questions ONLY about the candidate based on the provided resume information below.

For questions about programming, software development, and technical skills: Focus on the candidate's software engineering expertise, technologies, and contributions.
For questions about machine learning, AI, or data science: Emphasize the candidate's ML/AI knowledge, certifications, and relevant projects.
For other career questions: Provide a balanced view of their professional experience.

If a question is not related to the candidate's professional background or cannot be answered from the resume, politely say that you can only answer questions about the candidate's professional background and experience.

Resume:
{resume_text}

Question: {message}

Answer:"""
    
    prompt = prompt_template.format(resume_text=resume_text, message=message)
    
    try:
        full_response = ""
        message_placeholder = st.empty()
        
        for chunk in llm.stream(prompt):
            if chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        error_msg = f"Sorry, there was an error generating the response: {str(e)}. Please try again."
        st.error(error_msg)
        return error_msg

def main():
    # Header
    st.title("🤖 Candidate Resume Chatbot")
    st.markdown("Ask questions about the candidate. This chatbot only provides information from their resume.")
    
    # Load resume content using cache
    resume_text = get_document_content()
    
    # Initialize LLM
    llm = initialize_llm()
    
    if resume_text is None or llm is None:
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the candidate..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_response(prompt, llm, resume_text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with resume preview
    with st.sidebar:
        st.header("📄 Resume Preview")
        if resume_text:
            preview_text = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
            st.text_area("Resume Content", preview_text, height=300, disabled=True)
            
            st.markdown("---")
            st.markdown(f"**Total characters:** {len(resume_text)}")
            
            if st.button("🔄 Clear Cache & Reload"):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("• Ask questions about the candidate's experience")
        st.markdown("• Focus on technical skills, projects, or career background")
        st.markdown("• The bot only answers based on resume information")

if __name__ == "__main__":
    main()
