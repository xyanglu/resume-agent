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

# Resume-side signals, keyed by the same canonical labels as JD_SKILL_TERMS.
# Why this exists: the JD side speaks in labels ("Agentic workflows", "LLM APIs") while the
# resume speaks in evidence ("agent orchestration (LangGraph, MCP)", "OpenAI-compatible").
# Matching label strings against resume text reported real skills as gaps, which also lowered
# the headline match score and pushed the chat model into calling them gaps.
RESUME_SIGNALS = {
    "Agentic workflows": [r"\bagentic\b", r"\bmulti[- ]agent\b", r"agent orchestrat", r"\blanggraph\b",
                          r"\bmcp\b", r"tool[- ]calling", r"\bautonomous\b", r"\bagents?\b"],
    "LLM APIs": [r"\bllm apis?\b", r"\bopenai\b", r"\banthropic\b", r"\bclaude\b", r"\bopenrouter\b",
                 r"\bdeepseek\b", r"\bgemini\b", r"hugging face"],
    "Latency and cost": [r"\blatency\b", r"\bthroughput\b", r"\bp95\b", r"\bcost\b"],
    "Data quality": [r"data quality", r"\bvalidat", r"\bintegrity\b", r"\bdedup", r"\baccuracy\b"],
    "Financial services": [r"\bfinanc", r"\bbank", r"\bfintech\b", r"\btrading\b", r"\binsurance\b",
                           r"\bpayments?\b"],
    "Azure": [r"\bazure\b"],
    "AWS": [r"\baws\b", r"amazon web services"],
    "GCP": [r"\bgcp\b", r"google cloud"],
    "RAG": [r"\brag\b", r"retrieval[- ]augmented", r"\bretrieval\b"],
    "Vector databases": [r"vector (?:database|db|store)", r"\bchroma\b", r"\bfaiss\b", r"\bpinecone\b",
                         r"\bpgvector\b", r"\bweaviate\b", r"\bembedding"],
    "Prompt engineering": [r"prompt engineering", r"system prompt", r"few[- ]shot"],
    "LLM evaluation": [r"\bevals?\b", r"\bevaluation", r"\bjudge\b", r"\bbenchmark", r"\bscoring\b"],
    "Observability": [r"observab", r"\bmonitoring\b", r"\btracing\b", r"\blogging\b"],
    "AI coding assistants": [r"\bcopilot\b", r"\bcursor\b", r"claude code", r"ai coding"],
    "Workflow orchestration": [r"\borchestrat", r"\bairflow\b", r"\bdag\b", r"\btemporal\b"],
    "Fine-tuning": [r"fine[- ]tun", r"\blora\b", r"\bqlora\b", r"\bsft\b", r"\bpeft\b"],
    "Temporal": [r"\btemporal\b"],
    "ETL": [r"\betl\b", r"\belt\b"],
    "SQL": [r"\bsql\b"],
    "Docker": [r"\bdocker\b", r"\bkubernetes\b"],
    "CI/CD": [r"ci/cd", r"github actions"],
    "FastAPI": [r"\bfastapi\b"],
    "Java": [r"\bjava\b"],
    "Spring": [r"\bspring\b"],
    "Python": [r"\bpython\b"],
    "LangChain": [r"\blangchain\b"],
    "LangGraph": [r"\blanggraph\b"],
    "Agno": [r"\bagno\b"],
    "LlamaIndex": [r"\bllamaindex\b"],
}


def resume_signals(skill):
    """Resume-side regexes for one canonical skill label (empty list if unknown)."""
    return RESUME_SIGNALS.get(str(skill).strip(), []) or RESUME_SIGNALS.get(str(skill).lower(), [])


def skill_in_text(skill, haystack):
    """True when the resume text shows evidence of this skill.

    Word-boundary regexes on purpose: substring tests let "rag" match inside unrelated
    words and made every label a coin flip.
    """
    if not haystack:
        return False
    text = haystack.lower()
    for pat in resume_signals(skill):
        if re.search(pat, text):
            return True
    label = str(skill).lower().strip()
    for variant in {label, label.replace("-", " "), label.replace("-", ""), label.rstrip("s")}:
        if variant and re.search(r"\b" + re.escape(variant) + r"\b", text):
            return True
    return False




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


