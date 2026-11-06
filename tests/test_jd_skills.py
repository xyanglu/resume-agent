import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jd_skills import extract_jd_skills

BRAID_JD = """
Braid Senior AI Engineer
locations Boston Jersey City New York
Build and iterate on LLM pipelines, RAG systems, and agentic workflows using Agno.
Hands-on with LLM APIs in production - Anthropic Claude and/or OpenAI
Experience with agentic frameworks - Agno, LangChain, LlamaIndex, or comparable
Strong Python proficiency
Azure or AWS cloud experience
Vector databases - Pinecone, Weaviate, pgvector, or comparable
Active user of AI coding assistants in daily workflow
build evaluation datasets and automated scoring frameworks
"""


def test_extracts_real_requirements():
    got = extract_jd_skills(BRAID_JD)
    for want in ["Python", "RAG", "Agno", "LangChain", "Azure", "AWS",
                 "Vector databases", "LLM APIs", "LLM evaluation"]:
        assert want in got, want


def test_prose_words_are_not_skills():
    got = extract_jd_skills(BRAID_JD)
    banned = {"boston", "jersey", "financial", "build", "senior", "engineer", "locations"}
    assert not (banned & {g.lower() for g in got})


def test_empty_jd_returns_empty_list():
    assert extract_jd_skills("") == []


def test_limit_is_respected():
    assert len(extract_jd_skills(BRAID_JD, limit=3)) == 3
