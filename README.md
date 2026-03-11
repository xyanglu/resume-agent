---
title: Resume Agent Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Resume Agent Chatbot 🤖

An AI-powered chatbot that answers questions about a candidate using their resume from Google Docs. This Space demonstrates dynamic document integration and conversational AI deployment.

## Changelog

### [2026-03-11] - Latest Updates

### [2026-03-11]
#### Added
- Added Streamlit app with full RAG chatbot functionality
- Configured local deployment setup with .streamlit/secrets.toml
- Added chat history as context for improved conversation continuity

#### Changed
- Switched app.py to OpenRouter API for flexible model access
- Switched to OpenRouter/free model for cost-effective usage
- Added chat history to both apps for better context tracking

#### Fixed
- Fixed messages initialization in get_history function
- Reverted changes when chat history wasn't needed with simplified approach

### [2026-03-09]
#### Fixed
- Fixed package name for Debian compatibility

### [2026-03-03]
#### Changed
- Moved changelog to top of README for better visibility
- Added GitHub-style changelog to README.md

### [2026-02-27]
#### Changed
- Fixed formatting issues where output was not following prompt instructions properly
- Enhanced prompt engineering to ensure better adherence to formatting requirements

### [2026-02-26]
#### Added
- Changed output format from Markdown to PDF for better document sharing and professional presentation
- Implemented PDF generation with proper formatting for resumes and cover letters

### [2026-02-25]
#### Added
- Feature to generate custom resumes and cover letters based on user requirements
- Dynamic content generation tailored to specific job descriptions and user preferences

### [2026-02-24]
#### Changed
- Implemented redundancy using multiple LLM models (Z.AI GLM and Google Gemini)
- Added ability to switch between models for reliability and cost optimization
- Integrated free tier Google Gemini 3 Flash for backup and testing

### [2026-02-23]
#### Changed
- Migrated from basic prompt-based system to RAG (Retrieval-Augmented Generation)
- Improved accuracy and context awareness using LangChain QA chains
- Enhanced document retrieval and answer generation pipeline

### [2026-02-22]
#### Added
- Implemented Google Docs scraping to dynamically fetch resume content
- Integrated resume content into the prompt for context-aware responses
- Added 5-minute caching to reduce API calls and improve performance

### [2026-02-21]
#### Added
- Multi-platform UI support with both Gradio and Streamlit interfaces
- Hosted on Hugging Face Spaces for easy access
- Dual web UI options for different user preferences

### [2026-02-19]
#### Added
- Initial chatbot implementation with basic Q&A functionality
- Candidate-focused conversation system
- Safety filters to decline non-candidate questions

---

## Features

- **Dynamic Resume Integration**: Fetches the latest resume content from Google Docs automatically
- **Smart Caching**: Resume content is cached for 5 minutes to reduce API calls and improve performance
- **Intelligent Q&A**: Uses advanced language models (OpenRouter, Z.AI or Google Gemini) to provide accurate, context-aware responses
- **Multi-LLM Support**: Switch between OpenRouter/free, Z.AI GLM models or Google Gemini 3 Flash (free tier)
- **Chat History**: Maintains conversation context across multiple interactions for better follow-up questions
- **Candidate-Focused**: Politely declines questions not related to candidate's background
- **Dual Interface**: Available in both Gradio and Streamlit versions

## How It Works

1. **Resume Source**: The chatbot loads resume content from a publicly shared Google Doc
2. **Smart Caching**: Content is cached for 5 minutes to minimize API calls and improve response times
3. **AI Processing**: Questions are processed using OpenRouter's free model or other providers with the resume as context
4. **Chat History**: Previous conversation context is maintained for follow-up questions
5. **Smart Responses**: Answers are based solely on the provided resume information
6. **Safety Filters**: Non-candidate questions are handled gracefully

## Technical Stack

- **Frontend**: Gradio & Streamlit (Dual Web UI options)
- **Backend**: Python, LangChain
- **AI Model**: OpenRouter/free (default), Z.AI GLM-4.7-flash or Google Gemini 3 Flash (via LangChain)
- **Data Source**: Google Docs (public export)
- **Caching**: Streamlit cache (5min TTL) & File-based cache (Gradio)
- **Hosting**: Hugging Face Spaces
- **API Integration**: OpenRouter for flexible model access

## Usage

### Running the Apps

**Streamlit Version (Recommended):**

```bash
streamlit run streamlit_app.py
```

**Gradio Version:**

```bash
python app.py
```

### Example Questions

Simply type questions about the candidate in the chat interface:

- "What are the candidate's strongest technical skills?"
- "Tell me about their work experience at Cryptomate"
- "What education qualifications do they have?"
- "Follow-up: And what about their other projects?"

### Caching

Both versions implement smart caching:

- **Streamlit**: Uses `@st.cache_data(ttl=300)` for 5-minute cache
- **Gradio**: Uses file-based cache (`resume_cache.json`) with 5-minute TTL
- Cache can be cleared manually in Streamlit via the sidebar button

### Switching LLM Providers

The app supports multiple LLM providers. To switch between them, set the following environment variables:

**Option 1: OpenRouter/free (Default)**
```bash
USE_OPENROUTER=true
OPENROUTER_API_KEY=your_openrouter_api_key
MODEL_NAME=openrouter/free
```

**Option 2: Z.AI**
```bash
USE_GEMINI=false
ZAI_API_KEY=your_zai_api_key
MODEL_NAME=glm-4.7-flash
```

**Option 3: Google Gemini (Free Tier)**
```bash
USE_GEMINI=true
GOOGLE_API_KEY=your_google_api_key
```

OpenRouter/free provides:
- Free access to multiple models
- No rate limiting for basic usage
- Flexible model switching

Google Gemini 3 Flash includes a free tier with:
- 15 requests per minute
- 1M tokens per minute
- No cost for typical usage

## Deployment

This app is hosted on Hugging Face Spaces for free. The code automatically fetches fresh resume data on each startup.

For more details on setup and deployment, see the [project repository](https://github.com/xyanglu/resume-agent).

## Privacy & Security

- Only candidate-related questions are answered
- Resume data comes from a public Google Doc
- No personal data stored beyond the session
- API keys are managed securely through environment variables
- Chat history is maintained only during the current session

---

Built with ❤️ for showcasing AI capabilities in recruitment technology.