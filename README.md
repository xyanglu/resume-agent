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

## Features

- **Dynamic Resume Integration**: Fetches the latest resume content from Google Docs automatically
- **Smart Caching**: Resume content is cached for 5 minutes to reduce API calls and improve performance
- **Intelligent Q&A**: Uses advanced language models (Z.AI) to provide accurate, context-aware responses
- **Candidate-Focused**: Politely declines questions not related to the candidate's background
- **Dual Interface**: Available in both Gradio and Streamlit versions

## How It Works

1. **Resume Source**: The chatbot loads resume content from a publicly shared Google Doc
2. **Smart Caching**: Content is cached for 5 minutes to minimize API calls and improve response times
3. **AI Processing**: Questions are processed using Z.AI's GLM models with the resume as context
4. **Smart Responses**: Answers are based solely on the provided resume information
5. **Safety Filters**: Non-candidate questions are handled gracefully

## Technical Stack

- **Frontend**: Gradio & Streamlit (Dual Web UI options)
- **Backend**: Python, LangChain
- **AI Model**: Z.AI GLM-4.5-flash (via LangChain)
- **Data Source**: Google Docs (public export)
- **Caching**: Streamlit cache (5min TTL) & File-based cache (Gradio)
- **Hosting**: Hugging Face Spaces

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

### Caching

Both versions implement smart caching:

- **Streamlit**: Uses `@st.cache_data(ttl=300)` for 5-minute cache
- **Gradio**: Uses file-based cache (`resume_cache.json`) with 5-minute TTL
- Cache can be cleared manually in Streamlit via the sidebar button

## Deployment

This app is hosted on Hugging Face Spaces for free. The code automatically fetches fresh resume data on each startup.

For more details on setup and deployment, see the [project repository](https://github.com/xyanglu/resume-agent).

## Privacy & Security

- Only candidate-related questions are answered
- Resume data comes from a public Google Doc
- No personal data stored beyond the sessionv
- API keys are managed securely through environment variables

---

Built with ❤️ for showcasing AI capabilities in recruitment technology.
