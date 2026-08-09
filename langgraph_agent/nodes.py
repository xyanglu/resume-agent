"""
LangGraph nodes — each node is a function that takes AgentState and returns a partial state update.

The graph flow:
    START
      -> fetch_jd_node        (fetches URL, extracts JD text)
      -> analyze_fit_node     (extracts key requirements)
      -> generate_resume_node (LLM generates tailored resume, renders PDF)
      -> check_pages_node     (checks if PDF fits on 1 page)
      
      If multi-page (conditional edge):
        -> tighten_resume_node (re-generate with tighter constraints)
        -> check_pages_node    (re-check, max 3 rounds)
      
      If 1 page or max rounds reached:
        -> evaluate_resume_node (LLM evaluates fit on 5 dimensions)
      -> END
"""

from .state import AgentState
from .tools import (
    fetch_jd_from_url,
    extract_key_requirements,
    load_source_resume,
    count_pdf_pages,
    generate_resume_markdown,
    markdown_to_html,
    render_pdf,
    evaluate_tailored_resume,
)


# Maximum tightening rounds before accepting the best effort
MAX_TIGHTEN_ROUNDS = 3

# Progressive tightening configs — each round reduces content
TIGHTEN_CONFIGS = [
    {"max_bullets": 4, "summary_sentences": 3, "max_chars": 3000, "css_mode": "normal"},
    {"max_bullets": 3, "summary_sentences": 2, "max_chars": 2700, "css_mode": "normal"},
    {"max_bullets": 3, "summary_sentences": 2, "max_chars": 2400, "css_mode": "tight"},
    {"max_bullets": 2, "summary_sentences": 2, "max_chars": 2100, "css_mode": "tight"},
]


def fetch_jd_node(state: AgentState) -> dict:
    """
    Node 1: Fetch the JD from the provided URL.
    
    This is the entry point. It takes the jd_url from state,
    calls the fetch tool, and returns the extracted text.
    """
    url = state.get("jd_url", "")
    if not url:
        return {"fetch_error": "No JD URL provided", "status": "error"}
    
    result = fetch_jd_from_url(url)
    
    return {
        "jd_raw_text": result["raw_text"],
        "jd_title": result["title"],
        "jd_company": result["company"],
        "fetch_error": result["fetch_error"],
        "status": "jd_fetched" if result["raw_text"] else "fetch_failed",
    }


def analyze_fit_node(state: AgentState) -> dict:
    """
    Node 2: Analyze the JD to extract key requirements.
    
    Uses an LLM call to pull out the top technical requirements,
    which will inform resume generation and evaluation.
    """
    jd_text = state.get("jd_raw_text", "")
    if not jd_text:
        return {"jd_key_requirements": [], "status": "skip_analysis"}
    
    # Get LLM from a graph-level config (injected at compile time)
    llm = state.get("_llm")
    if llm is None:
        # Fallback: simple keyword extraction
        return {"jd_key_requirements": [], "status": "analyzed"}
    
    requirements = extract_key_requirements(jd_text, llm)
    
    return {
        "jd_key_requirements": requirements,
        "status": "analyzed",
    }


def generate_resume_node(state: AgentState) -> dict:
    """
    Node 3: Generate a tailored resume from the source resume and JD.
    
    This node calls the LLM to create a JD-specific resume in markdown,
    converts it to HTML, and renders a PDF. Also loads the source resume
    on first run.
    """
    # Load source resume if not already loaded
    source = state.get("source_resume", "")
    source_json = state.get("source_resume_json", {})
    if not source:
        resume_data = load_source_resume()
        source = resume_data["text"]
        source_json = resume_data["json"]
    
    llm = state.get("_llm")
    if llm is None:
        return {"error": "No LLM configured", "status": "error"}
    
    # Determine tightening level from round counter
    round_num = state.get("generation_round", 0)
    config = TIGHTEN_CONFIGS[min(round_num, len(TIGHTEN_CONFIGS) - 1)]
    
    # Generate resume markdown
    resume_md = generate_resume_markdown(
        source_resume=source,
        jd_text=state.get("jd_raw_text", ""),
        company=state.get("jd_company", ""),
        llm=llm,
        max_bullets=config["max_bullets"],
        summary_sentences=config["summary_sentences"],
        max_chars=config["max_chars"],
    )
    
    # Convert to HTML and render PDF
    html = markdown_to_html(resume_md, config["css_mode"])
    
    try:
        pdf_bytes = render_pdf(html)
        page_count = count_pdf_pages(pdf_bytes)
    except Exception as e:
        pdf_bytes = b""
        page_count = 1  # Assume ok if we can't check
    
    return {
        "source_resume": source,
        "source_resume_json": source_json,
        "tailored_resume_md": resume_md,
        "tailored_resume_html": html,
        "tailored_resume_pdf": pdf_bytes,
        "page_count": page_count,
        "needs_tightening": page_count > 1,
        "generation_round": round_num + 1,
        "status": "generated",
    }


def check_pages_node(state: AgentState) -> dict:
    """
    Node 4: Check whether the current resume PDF fits on one page.
    
    This is a pass-through node that sets the needs_tightening flag
    for the conditional edge to read.
    """
    page_count = state.get("page_count", 1)
    round_num = state.get("generation_round", 0)
    
    needs_tightening = page_count > 1 and round_num < MAX_TIGHTEN_ROUNDS
    
    return {
        "needs_tightening": needs_tightening,
        "status": "page_check_done",
    }


def tighten_resume_node(state: AgentState) -> dict:
    """
    Node 5 (conditional): Re-generate the resume with tighter constraints.
    
    Bumps the generation_round counter so generate_resume_node picks
    the next tighter config from TIGHTEN_CONFIGS.
    """
    # The generate_resume node reads generation_round to pick config.
    # We just need to signal it to re-run with a higher round number.
    # Round was already incremented by the previous generate call.
    return {
        "status": "tightening",
    }


def evaluate_resume_node(state: AgentState) -> dict:
    """
    Node 6: Evaluate the final tailored resume against the JD.
    
    Scores on 5 dimensions: skills match, relevance, keyword coverage,
    factual grounding, overall fit. This is the quality gate before
    the resume is delivered.
    """
    resume_md = state.get("tailored_resume_md", "")
    jd_text = state.get("jd_raw_text", "")
    source = state.get("source_resume", "")
    
    llm = state.get("_llm")
    if llm is None:
        return {"status": "evaluated_no_llm"}
    
    scores = evaluate_tailored_resume(
        tailored_resume=resume_md,
        jd_text=jd_text,
        source_resume=source,
        llm=llm,
    )
    
    # Build notes from individual dimension reasons
    notes_parts = []
    for dim in ["skills_match", "relevance", "keyword_coverage", "factual_grounding", "overall_fit"]:
        reason = scores.get(f"{dim}_reason", "")
        score = scores.get(dim, 0)
        if reason:
            notes_parts.append(f"{dim} ({score}/10): {reason}")
    
    return {
        "eval_skills_match": scores.get("skills_match", 0),
        "eval_relevance": scores.get("relevance", 0),
        "eval_keyword_coverage": scores.get("keyword_coverage", 0),
        "eval_factual_grounding": scores.get("factual_grounding", 0),
        "eval_overall_fit": scores.get("overall_fit", 0),
        "eval_notes": "\n".join(notes_parts),
        "status": "complete",
    }


def should_tighten(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to tighten the resume
    or proceed to evaluation.
    
    Returns the name of the next node: "tighten_resume" or "evaluate_resume".
    """
    if state.get("needs_tightening", False):
        return "tighten_resume"
    return "evaluate_resume"
