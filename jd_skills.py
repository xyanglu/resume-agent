"""Skill extraction for pasted job descriptions (no Streamlit dependency)."""
import re

# Canonical skills that may appear in a pasted JD, with the wordings that signal them.
# Deliberately a closed vocabulary: the previous version split the whole JD on whitespace and
# called every word a skill, so "Boston", "financial" and "build" showed up as requirements.
JD_SKILL_TERMS = {
    "Python": [r"\bpython\b"],
    "LLM APIs": [r"\bllm apis?\b", r"\bopenai\b", r"\banthropic\b", r"\bclaude\b"],
    "RAG": [r"\brag\b", r"retrieval[- ]augmented"],
    "LangChain": [r"\blangchain\b"],
    "LangGraph": [r"\blanggraph\b"],
    "Agno": [r"\bagno\b"],
    "LlamaIndex": [r"\bllamaindex\b"],
    "Agentic workflows": [r"agentic", r"\bagents?\b"],
    "Vector databases": [r"vector (?:database|db|store)s?", r"\bpinecone\b", r"\bweaviate\b",
                         r"\bpgvector\b", r"\bchroma\b", r"\bfaiss\b"],
    "Azure": [r"\bazure\b"],
    "AWS": [r"\baws\b"],
    "GCP": [r"\bgcp\b", r"google cloud"],
    "Prompt engineering": [r"prompt engineering"],
    "LLM evaluation": [r"evaluation (?:datasets|frameworks?)", r"\bevals?\b",
                       r"scoring frameworks?", r"automated scoring"],
    "Fine-tuning": [r"fine[- ]tun"],
    "Temporal": [r"\btemporal\b"],
    "Workflow orchestration": [r"workflow orchestration"],
    "ETL": [r"\betl\b", r"\belt\b"],
    "SQL": [r"\bsql\b"],
    "Docker": [r"\bdocker\b", r"kubernetes"],
    "CI/CD": [r"ci/cd", r"github actions"],
    "FastAPI": [r"\bfastapi\b"],
    "Java": [r"\bjava\b"],
    "Spring": [r"\bspring\b"],
    "Data quality": [r"data quality"],
    "Observability": [r"observab"],
    "Latency and cost": [r"\blatency\b"],
    "AI coding assistants": [r"ai coding assistant"],
    "Financial services": [r"financial services", r"financial data"],
}
MAX_JD_SKILLS = 14


def extract_jd_skills(raw_text, limit=MAX_JD_SKILLS):
    """Return the canonical skills a JD actually asks for, in the order they appear.

    Only vocabulary terms count, so prose nouns never become chips.
    """
    text = raw_text.lower()
    found = []
    for name, patterns in JD_SKILL_TERMS.items():
        positions = [m.start() for pat in patterns for m in [re.search(pat, text)] if m]
        if positions:
            found.append((min(positions), name))
    found.sort()
    return [name for _, name in found][:limit]


