---
title: Resume Agent Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Resume Agent Chatbot

An AI-powered RAG chatbot that answers questions about a candidate using their resume pulled from Google Docs. Built with LangChain, supports multiple LLM providers (OpenRouter, Z.AI GLM, Google Gemini).

## Features

- **Dynamic Resume Integration** — Fetches the latest resume from Google Docs automatically
- **RAG Pipeline** — LangChain QA chains for context-aware retrieval and generation
- **Multi-LLM Support** — OpenRouter/free, Z.AI GLM, or Google Gemini 3 Flash
- **Chat History** — Maintains conversation context for follow-up questions
- **Smart Caching** — 5-minute TTL to reduce API calls
- **Candidate-Focused** — Declines questions unrelated to the candidate's background
- **Dual Interface** — Gradio and Streamlit versions included

## Tech Stack

- **Frontend**: Gradio & Streamlit
- **Backend**: Python, LangChain
- **AI Models**: OpenRouter/free, Z.AI GLM-4.7-flash, Google Gemini 3 Flash
- **Data Source**: Google Docs (public export)
- **Hosting**: Hugging Face Spaces

## Quick Start

```bash
# Streamlit (recommended)
streamlit run streamlit_app.py

# Gradio
python app.py
```

Set your API key via environment variables:

```bash
# OpenRouter (default)
OPENROUTER_API_KEY=your_key

# Z.AI
ZAI_API_KEY=your_key

# Google Gemini
GOOGLE_API_KEY=your_key
```

## Live Demo

Hosted on Hugging Face Spaces.

## How It Works

1. Resume content is fetched from a public Google Doc
2. Content is cached for 5 minutes to minimize API calls
3. User questions are processed through a LangChain QA chain with the resume as context
4. Chat history is maintained for follow-up questions
5. Non-candidate questions are politely declined
