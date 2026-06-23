---
title: Resume Agent Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# RAG Resume Agent

A deployed resume Q&A agent that answers recruiter-facing questions from structured resume data.

Live demo: https://huggingface.co/spaces/xyanglu/resume-agent-chatbot

Instead of treating a resume as a static PDF, this project makes the resume queryable. It uses retrieval over a structured `resume.json` source of truth, then answers questions with role context and resume-grounded evidence.

## What it does

- Answers questions about experience, projects, skills, and role fit
- Uses structured resume data instead of scraping a PDF
- Supports role-aware context through tracked job links
- Grounds responses in retrieved resume content
- Includes an access gate for recruiter/job-specific links
- Runs as a Streamlit app on Hugging Face Spaces

Example questions:

- What AI systems has this candidate built?
- Where has he used LangGraph, RAG, or vector search?
- How does his backend engineering experience connect to AI engineering?
- What should be emphasized for this role?

## Tech stack

- Python
- Streamlit
- LangChain
- LangGraph patterns
- Chroma vector search
- OpenRouter / Z.AI model providers
- Hugging Face Spaces
- GitHub-hosted `resume.json` source of truth

## Architecture

1. Resume data is maintained in `resume.json`
2. The app formats the structured resume into retrievable text
3. Text chunks are embedded and stored in Chroma
4. User questions are routed through retrieval and chat context
5. Responses are generated from retrieved resume evidence
6. Optional job-link parameters add role context for targeted Q&A

## Why this exists

Recruiters and hiring managers do not always ask the same questions a resume answers directly.

This project explores a simple idea: a candidate profile should be inspectable. Instead of guessing which bullets matter for each reader, the resume can answer targeted questions while staying tied to the documented source of truth.

The goal is not to decorate experience. The goal is to make real experience easier to inspect.

## Related open-source work

I also submitted public PRs to NousResearch/hermes-agent, focused on reliability and messaging workflows in an AI agent framework:

- Signal document attachment handling: https://github.com/NousResearch/hermes-agent/pull/38728
- Gateway replay for buffered Signal messages: https://github.com/NousResearch/hermes-agent/pull/37643
- Voice transcript preservation through context compression: https://github.com/NousResearch/hermes-agent/pull/37640
- Rate-limit message queue for gateway sessions: https://github.com/NousResearch/hermes-agent/pull/31092
- CJK search tokenization and session lineage fixes: https://github.com/NousResearch/hermes-agent/pull/24048
- Signal long-message reassembly and attachment classification: https://github.com/NousResearch/hermes-agent/pull/24047
- Cron 429 retry queue with exponential backoff: https://github.com/NousResearch/hermes-agent/pull/24046
- Gateway dead-letter queue for rate-limited messages: https://github.com/NousResearch/hermes-agent/pull/24045

These PRs are listed as submitted public contributions, not claimed as merged unless GitHub shows them merged.

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

Environment variables:

```bash
ZAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

The deployed Space uses Streamlit with `app.py` as the entry point.
