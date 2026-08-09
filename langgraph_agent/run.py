#!/usr/bin/env python3
"""
CLI entry point for the LangGraph Resume Tailoring Agent.

Usage:
    python -m langgraph_agent.run "https://www.linkedin.com/jobs/view/4451002955/"

Outputs:
    - resume-tailored.md    (tailored resume markdown)
    - resume-tailored.pdf   (1-page PDF)
    - eval-report.json      (5-dimension evaluation scores)

Environment:
    ZAI_API_KEY  or  OPENROUTER_API_KEY  (for LLM calls)
"""

import sys
import os
import json

# Add parent dir to path so we can import langchain_openai
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langgraph_agent.graph import build_resume_agent


def get_llm(temperature=0.3):
    """Create LLM client from available API keys."""
    zai_key = os.getenv("ZAI_API_KEY")
    if zai_key:
        print("Using Z.AI (glm-4.7-flash)")
        return ChatOpenAI(
            model="glm-4.7-flash",
            api_key=zai_key,
            base_url="https://api.zai.chat/v1",
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
    
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        print("Using OpenRouter (auto)")
        return ChatOpenAI(
            model="openrouter/auto",
            api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    
    print("ERROR: Set ZAI_API_KEY or OPENROUTER_API_KEY")
    sys.exit(1)


def print_eval_report(result: dict):
    """Print a readable evaluation report."""
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    
    dims = [
        ("Skills Match", result.get("eval_skills_match", 0)),
        ("Relevance", result.get("eval_relevance", 0)),
        ("Keyword Coverage", result.get("eval_keyword_coverage", 0)),
        ("Factual Grounding", result.get("eval_factual_grounding", 0)),
        ("Overall Fit", result.get("eval_overall_fit", 0)),
    ]
    
    for label, score in dims:
        bar = "█" * score + "░" * (10 - score)
        emoji = "🟢" if score >= 7 else ("🟡" if score >= 5 else "🔴")
        print(f"  {emoji} {label:20s} {bar} {score}/10")
    
    avg = sum(s for _, s in dims) / len(dims) if dims else 0
    print(f"\n  Average: {avg:.1f}/10")
    print(f"  Pages: {result.get('page_count', '?')}")
    print(f"  Generation rounds: {result.get('generation_round', 0)}")
    print(f"  Status: {result.get('status', '?')}")
    
    notes = result.get("eval_notes", "")
    if notes:
        print(f"\n{'─' * 50}")
        print("Detailed notes:")
        print(notes)
    
    print("=" * 50)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m langgraph_agent.run <jd_url>")
        print("Example: python -m langgraph_agent.run 'https://www.linkedin.com/jobs/view/4451002955/'")
        sys.exit(1)
    
    jd_url = sys.argv[1]
    print(f"Resume Agent starting")
    print(f"JD URL: {jd_url}")
    print()
    
    # Build LLM
    llm = get_llm()
    
    # Build the graph
    print("Building LangGraph...")
    agent = build_resume_agent()
    
    # Invoke with initial state
    # The _llm key injects our LLM client into the state for nodes to use
    print("Running agent...")
    result = agent.invoke({
        "jd_url": jd_url,
        "_llm": llm,
        "generation_round": 0,
    })
    
    # Check for errors
    if result.get("fetch_error"):
        print(f"\nFetch error: {result['fetch_error']}")
        print("The URL may require authentication (LinkedIn login).")
        print("Try using a direct JD URL or paste the JD text.")
    
    if result.get("error"):
        print(f"\nError: {result['error']}")
        sys.exit(1)
    
    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    resume_md = result.get("tailored_resume_md", "")
    if resume_md:
        md_path = os.path.join(output_dir, "resume-tailored.md")
        with open(md_path, "w") as f:
            f.write(resume_md)
        print(f"\nResume markdown: {md_path}")
    
    pdf_bytes = result.get("tailored_resume_pdf")
    if pdf_bytes:
        pdf_path = os.path.join(output_dir, "resume-tailored.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"Resume PDF: {pdf_path}")
    
    # Save eval report
    eval_data = {
        "jd_url": jd_url,
        "jd_title": result.get("jd_title", ""),
        "jd_company": result.get("jd_company", ""),
        "page_count": result.get("page_count", 0),
        "generation_rounds": result.get("generation_round", 0),
        "scores": {
            "skills_match": result.get("eval_skills_match", 0),
            "relevance": result.get("eval_relevance", 0),
            "keyword_coverage": result.get("eval_keyword_coverage", 0),
            "factual_grounding": result.get("eval_factual_grounding", 0),
            "overall_fit": result.get("eval_overall_fit", 0),
        },
        "notes": result.get("eval_notes", ""),
    }
    eval_path = os.path.join(output_dir, "eval-report.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Evaluation report: {eval_path}")
    
    # Print report to console
    print_eval_report(result)


if __name__ == "__main__":
    main()
