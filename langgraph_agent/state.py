"""
Typed state for the LangGraph resume agent.

State flows through the graph and carries all context between nodes.
This is the core LangGraph pattern — typed state that each node reads and writes.
"""

from typing import TypedDict, Optional, Any
from pydantic import BaseModel


class JobDescription(BaseModel):
    """Extracted job description details."""
    url: str = ""
    raw_text: str = ""
    title: str = ""
    company: str = ""
    key_requirements: list[str] = []


class EvalResult(BaseModel):
    """Multi-dimension evaluation result."""
    skills_match: int = 0
    relevance: int = 0
    keyword_coverage: int = 0
    factual_grounding: int = 0
    overall_fit: int = 0
    notes: str = ""


class AgentState(TypedDict, total=False):
    """
    The state object that flows through every node in the graph.
    
    LangGraph uses TypedDict for state — each node can read any key
    and must write back the keys it modifies.
    """
    # Input
    jd_url: str
    
    # Populated by fetch_jd node
    jd_raw_text: str
    jd_title: str
    jd_company: str
    jd_key_requirements: list[str]
    fetch_error: str
    
    # Source resume (loaded once)
    source_resume: str
    source_resume_json: dict
    
    # Populated by generate_resume node
    tailored_resume_md: str
    tailored_resume_html: str
    tailored_resume_pdf: bytes
    generation_round: int
    
    # Populated by check_pages / tighten node
    page_count: int
    needs_tightening: bool
    
    # Populated by evaluate_resume node
    eval_skills_match: int
    eval_relevance: int
    eval_keyword_coverage: int
    eval_factual_grounding: int
    eval_overall_fit: int
    eval_notes: str
    
    # Metadata
    error: str
    status: str
    
    # Internal: LLM client injected at runtime
    _llm: Any
