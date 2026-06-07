"""
Recruiter-facing resume audit module.
Evaluates resume content for:
1. Red flags a recruiter would notice
2. Context-aware emphasis/downplay based on role type
3. Response filtering — rewrite to respect omit/downplay rules
"""

import json
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("/Users/yang/.hermes/.env")

OR_KEY = os.getenv("OPENROUTER_API_KEY", "")
AUDIT_MODEL = "google/gemma-4-31b-it:free"
TODAY = datetime.now().strftime("%B %d, %Y")

RISK_PROMPT = """You are a hiring manager pre-screening a resume. Today's date is {today}. Your job is to flag ANY content that could hurt the candidate's chances. Be critical.

RESUME:
{resume}

IMPORTANT CONTEXT:
- Today is {today}. Certifications dated before this year are NOT future-dated.
- If the skills list looks truncated at the end, it is a display artifact, not a real resume issue. The full skills list includes LangGraph, MCP, agent evaluation, prompt engineering, Anthropic API.

For each issue you find, output JSON with:
- "issue": short description
- "severity": 1-10 (10 = dealbreaker)
- "recruiter_reaction": how a recruiter sees this (1 sentence)
- "better_frame": how to present favorably, or "omit" if should be removed

If the resume is clean, return an empty array.

Output STRICTLY as JSON with NO extra text:
{{"flags": [...]}}"""

CONTEXT_PROMPT = """You are a resume strategist. Today's date is {today}. Given a resume and a role title, determine which parts to EMPHASIZE and which to DOWNPLAY or OMIT.

RESUME:
{resume}

ROLE: {role_title}

A recruiter will visit this agent and ask about the candidate's background. The agent should:
- Lead with relevant strengths for this role type
- Handle weak points proactively
- Omit or reframe details that hurt more than help

Output JSON:
{{
  "role_category": "backend" | "ai-ml" | "fullstack" | "startup" | "contract" | "general",
  "emphasize": ["bullet point to lead with", "specific experience to highlight"],
  "downplay": ["thing to mention briefly if asked, not lead with"],
  "omit": ["thing to never volunteer"],
  "framing_advice": "1-2 sentences on overall positioning"
}}"""

FILTER_PROMPT = """You are a response gatekeeper for a resume chatbot. Today's date is {today}. You check agent responses against rules before they go to a recruiter.

DRAFT RESPONSE:
{response}

OMIT LIST (never mention these):
{omit_list}

DOWNPLAY LIST (gloss over if asked, don't elaborate):
{downplay_list}

FRAMING ADVICE:
{framing_advice}

Rules:
1. If the response mentions anything from the OMIT list, remove or rewrite those sentences.
2. If the response elaborates on anything from the DOWNPLAY list, shorten it to 1 sentence max.
3. If the response contradicts the framing advice, rewrite to match.
4. Keep everything else exactly the same.

Output ONLY the filtered response text. No JSON, no commentary, no quotes.
If the response is clean, output it verbatim."""


def _call_model(prompt, model_id=AUDIT_MODEL):
    """Call OpenRouter and return raw content."""
    if not OR_KEY:
        return None
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.2,
            },
            timeout=45,
        )
        data = resp.json()
        if "error" in data:
            return None
        content = data["choices"][0]["message"]["content"].strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content
    except Exception:
        return None


def audit_recruiter_risks(resume_text):
    """Check resume for content that could hurt with recruiters.
    Returns list of flags or empty list on failure."""
    if not resume_text or not OR_KEY:
        return []
    prompt = RISK_PROMPT.format(today=TODAY, resume=resume_text)
    raw = _call_model(prompt)
    if not raw:
        return []
    try:
        result = json.loads(raw)
        flags = result.get("flags", [])
        flags.sort(key=lambda f: f.get("severity", 0), reverse=True)
        return flags
    except (json.JSONDecodeError, KeyError):
        return []


def analyze_role_context(resume_text, role_title):
    """Determine emphasis/downplay/omit strategy for a specific role.
    Returns dict or empty dict on failure."""
    if not resume_text or not role_title or not OR_KEY:
        return {}
    prompt = CONTEXT_PROMPT.format(today=TODAY, resume=resume_text, role_title=role_title)
    raw = _call_model(prompt)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def filter_response(response_text, role_context=None):
    """Filter agent response to respect omit/downplay rules.
    Returns filtered text, or original if no role_context or API fails."""
    if not response_text or not role_context or not OR_KEY:
        return response_text

    omit_list = role_context.get("omit", [])
    downplay_list = role_context.get("downplay", [])
    framing = role_context.get("framing_advice", "")

    if not omit_list and not downplay_list and not framing:
        return response_text

    prompt = FILTER_PROMPT.format(
        today=TODAY,
        response=response_text,
        omit_list="\n".join(f"- {item}" for item in omit_list) if omit_list else "None",
        downplay_list="\n".join(f"- {item}" for item in downplay_list) if downplay_list else "None",
        framing_advice=framing or "None",
    )

    raw = _call_model(prompt)
    if not raw:
        return response_text
    return raw


def format_risk_report(flags):
    """Format flags into readable markdown."""
    if not flags:
        return "No major red flags detected."
    lines = ["Recruiter Risk Audit"]
    for f in flags:
        sev = f.get("severity", 5)
        badge = "HIGH" if sev >= 7 else ("MED" if sev >= 4 else "LOW")
        lines.append(f"\n[{badge}] {f.get('issue', 'Unknown')}")
        lines.append(f"> Recruiter sees: {f.get('recruiter_reaction', '')}")
        lines.append(f"> Fix: {f.get('better_frame', '')}")
    return "\n".join(lines)


def format_context_guide(ctx):
    """Format context analysis into instructions for the agent."""
    if not ctx:
        return ""
    parts = []
    cat = ctx.get("role_category", "general")
    parts.append(f"Role category: {cat}")
    if ctx.get("emphasize"):
        parts.append("\nLead with:\n- " + "\n- ".join(ctx["emphasize"]))
    if ctx.get("downplay"):
        parts.append("\nDownplay:\n- " + "\n- ".join(ctx["downplay"]))
    if ctx.get("omit"):
        parts.append("\nNever volunteer:\n- " + "\n- ".join(ctx["omit"]))
    if ctx.get("framing_advice"):
        parts.append(f"\nStrategy: {ctx['framing_advice']}")
    return "\n\n".join(parts)
