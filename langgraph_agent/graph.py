"""
The LangGraph StateGraph definition.

This is the core of the agent — a directed graph that orchestrates
the resume tailoring workflow. Each node transforms state, and
conditional edges control the flow.

Graph structure:

    START
      |
    fetch_jd
      |
    analyze_fit
      |
    generate_resume  <-------+
      |                      |
    check_pages              |
      |                      |
      +--[multi-page]---> tighten_resume (loops back to generate_resume)
      |
      +--[1 page]-------> evaluate_resume
                            |
                          END

Key LangGraph concepts demonstrated:
1. TypedDict state (state.py)
2. Node functions that transform state (nodes.py)
3. Conditional edges (should_tighten)
4. Loop with max iterations (MAX_TIGHTEN_ROUNDS)
5. Tool functions called by nodes (tools.py)
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    fetch_jd_node,
    analyze_fit_node,
    generate_resume_node,
    check_pages_node,
    tighten_resume_node,
    evaluate_resume_node,
    should_tighten,
)


def build_resume_agent():
    """
    Build and compile the resume tailoring agent graph.
    
    Returns a compiled LangGraph that can be invoked with:
        result = agent.invoke({"jd_url": "https://...", "_llm": llm})
    
    The _llm key in the initial state injects the LLM client
    that nodes use for generation and evaluation.
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("fetch_jd", fetch_jd_node)
    graph.add_node("analyze_fit", analyze_fit_node)
    graph.add_node("generate_resume", generate_resume_node)
    graph.add_node("check_pages", check_pages_node)
    graph.add_node("tighten_resume", tighten_resume_node)
    graph.add_node("evaluate_resume", evaluate_resume_node)
    
    # Set entry point
    graph.set_entry_point("fetch_jd")
    
    # Linear edges
    graph.add_edge("fetch_jd", "analyze_fit")
    graph.add_edge("analyze_fit", "generate_resume")
    graph.add_edge("generate_resume", "check_pages")
    
    # Conditional edge: tighten or evaluate?
    graph.add_conditional_edges(
        "check_pages",
        should_tighten,
        {
            "tighten_resume": "generate_resume",  # Loop back to re-generate
            "evaluate_resume": "evaluate_resume",  # Proceed to evaluation
        },
    )
    
    # tighten_resume is a pass-through; the actual work happens in
    # generate_resume which reads the incremented round counter.
    # We skip it and route directly back via the conditional edge mapping.
    
    # Final edge to END
    graph.add_edge("evaluate_resume", END)
    
    # Compile and return
    return graph.compile()


# Build the agent on import for convenience
resume_agent = build_resume_agent()
