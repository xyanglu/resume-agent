# Resume Agent Chatbot 🤖

An AI-powered chatbot that answers questions about a candidate using their resume from Google Docs. This Space demonstrates dynamic document integration and conversational AI deployment.

## Features

-   **Dynamic Resume Integration**: Fetches the latest resume content from Google Docs automatically
-   **Intelligent Q&A**: Uses advanced language models (Z.AI) to provide accurate, context-aware responses
-   **Candidate-Focused**: Politely declines questions not related to the candidate's background
-   **Web Interface**: Clean, responsive chat interface built with Gradio

## How It Works

1. **Resume Source**: The chatbot loads resume content from a publicly shared Google Doc
2. **AI Processing**: Questions are processed using Z.AI's GLM models with the resume as context
3. **Smart Responses**: Answers are based solely on the provided resume information
4. **Safety Filters**: Non-candidate questions are handled gracefully

## Technical Stack

-   **Frontend**: Gradio (Web UI)
-   **Backend**: Python, LangChain
-   **AI Model**: Z.AI GLM-4.5-flash (via LangChain)
-   **Data Source**: Google Docs (public export)
-   **Hosting**: Hugging Face Spaces

## Usage

Simply type questions about the candidate in the chat interface. Examples:

-   "What are the candidate's strongest technical skills?"
-   "Tell me about their work experience at Cryptomate"
-   "What education qualifications do they have?"

## Deployment

This app is hosted on Hugging Face Spaces for free. The code automatically fetches fresh resume data on each startup.

For more details on setup and deployment, see the [project repository](https://github.com/xyanglu/resume-agent).

## Privacy & Security

-   Only candidate-related questions are answered
-   Resume data comes from a public Google Doc
-   No personal data stored beyond the sessionv
-   API keys are managed securely through environment variables

---

Built with ❤️ for showcasing AI capabilities in recruitment technology.
