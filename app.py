import os.path
import os
import requests

from langchain_openai import ChatOpenAI
import gradio as gr

# Extract document ID from the URL (use env var for security in production)
DOCUMENT_ID = os.getenv('DOCUMENT_ID')

def get_document_content():
    """Retrieves content from a public Google Doc using the export feature."""
    export_url = f"https://docs.google.com/document/d/{DOCUMENT_ID}/export?format=txt"

    try:
        print("Fetching resume content from Google Docs export...")
        response = requests.get(export_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        full_text = response.text.strip()
        print("Resume Content Fetched:")
        print(full_text[:500] + "..." if len(full_text) > 500 else full_text)  # Print truncated version

        return full_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching document: {e}")
        return None

if __name__ == '__main__':
    # Z.AI API configuration - use environment variable for production, fallback to local for development
    API_KEY = os.getenv('ZAI_API_KEY')

    # Initialize the LLM
    llm = ChatOpenAI(
        model="glm-4.5-flash",
        openai_api_key=API_KEY,
        openai_api_base="https://api.z.ai/api/paas/v4/",
        streaming=True
    )

    # Retrieve resume content
    resume_text = get_document_content()

    if resume_text:
        # Define the chat function with streaming
        def chat(message, history):
            if not message.strip():
                yield "Please ask a question about the candidate."
                return

            # Get prompt template from environment variable with fallback
            prompt_template = os.getenv('PROMPT_TEMPLATE')

            # Create prompt with resume context
            prompt = prompt_template.format(resume_text=resume_text, message=message)

            try:
                full_response = ""
                for chunk in llm.stream(prompt):
                    if chunk.content:
                        full_response += chunk.content
                        yield full_response
            except Exception as e:
                yield f"Sorry, there was an error generating the response: {str(e)}. Please try again."

        # Launch Gradio interface
        iface = gr.ChatInterface(
            fn=chat,
            title="Candidate Resume Chatbot",
            description="Ask questions about the candidate. This chatbot only provides information from their resume."
        )
        iface.launch()
    else:
        print("Failed to retrieve resume content. Cannot start chatbot.")
        print("Please ensure your Google Docs credentials are set up properly.")
