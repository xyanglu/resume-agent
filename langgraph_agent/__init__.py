"""
LangGraph Resume Tailoring Agent

State graph that:
1. Fetches a JD from a URL (tool-calling)
2. Generates a tailored resume from resume.json (tool-calling)
3. Evaluates the resume against the JD (tool-calling)
4. Auto-tightens to fit one page (conditional loop)
5. Outputs final resume + eval report

Architecture:
    START -> fetch_jd -> analyze_fit -> generate_resume -> check_pages
                                                       |-> (multi-page) -> tighten_resume -> check_pages
                                                       |-> (1 page) -> evaluate_resume -> END

This is a real LangGraph StateGraph with typed state, tool-calling nodes,
and conditional edges. Designed to be demonstrable in interviews.
"""
