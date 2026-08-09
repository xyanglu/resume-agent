"""
Tools available to the LangGraph resume agent.

In LangGraph, tools are functions that nodes call. Each tool does one thing
and returns structured data. The graph orchestrates the flow between tools.

These are NOT LangChain @tool decorators — they are plain Python functions
called directly by graph nodes. This keeps the code clear and debuggable,
and shows you understand that LangGraph nodes are just functions that
transform state.
"""

import re
import httpx
import json

from .state import AgentState


def fetch_jd_from_url(url: str) -> dict:
    """
    Fetch job description text from a LinkedIn or generic URL.
    
    For LinkedIn URLs, extracts the job posting text.
    For raw URLs, returns the page text.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        with httpx.Client(follow_redirects=True, timeout=15) as client:
            resp = client.get(url, headers=headers)
            text = resp.text
        
        # Try to extract title and company from LinkedIn HTML
        title = ""
        company = ""
        
        # LinkedIn title patterns
        title_match = re.search(r'<h1[^>]*class="[^"]*t-24[^"]*"[^>]*>(.*?)</h1>', text, re.S)
        if title_match:
            title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
        
        # Generic <title> tag fallback
        if not title:
            title_tag = re.search(r'<title>(.*?)</title>', text, re.S | re.I)
            if title_tag:
                title = title_tag.group(1).strip()
                # Clean up common prefixes
                title = re.sub(r'\s*[-|]\s*LinkedIn.*$', '', title)
                title = re.sub(r'\s*[-|]\s*Jobs.*$', '', title)
        
        # Company extraction
        company_match = re.search(r'class="[^"]*job-details-jobs-unified-top-card__company-name[^"]*"[^>]*>(.*?)</a>', text, re.S)
        if company_match:
            company = re.sub(r'<[^>]+>', '', company_match.group(1)).strip()
        
        # Extract visible text (strip HTML tags, scripts, styles)
        # Remove scripts and styles
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.S | re.I)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.S | re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Try to find the "About the job" section
        about_match = re.search(r'About the job(.+?)(?:Set alert|Benefits found|Show more)', text, re.S | re.I)
        if about_match:
            jd_text = about_match.group(1).strip()
        else:
            # Use full text if no About section found (might be a raw JD page)
            jd_text = text[:8000]
        
        return {
            "raw_text": jd_text,
            "title": title,
            "company": company,
            "fetch_error": "",
        }
    except Exception as e:
        return {
            "raw_text": "",
            "title": "",
            "company": "",
            "fetch_error": str(e)[:200],
        }


def extract_key_requirements(jd_text: str, llm) -> list[str]:
    """
    Use an LLM to extract the top requirements from a JD.
    Returns a list of requirement strings.
    """
    prompt = f"""Extract the 8-10 most important technical requirements from this job description.
Return ONLY a JSON array of strings, nothing else.

Job Description:
{jd_text[:4000]}
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # Handle markdown code blocks
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    
    try:
        reqs = json.loads(content)
        if isinstance(reqs, list):
            return [str(r) for r in reqs][:10]
    except json.JSONDecodeError:
        pass
    
    # Fallback: split by lines
    return [line.strip() for line in content.split("\n") if line.strip()][:10]


def load_source_resume(resume_path: str = None) -> dict:
    """
    Load resume.json and return both the JSON dict and formatted text.
    """
    import os
    if resume_path is None:
        resume_path = os.path.expanduser(
            "~/Documents/interview-prep/resume.json"
        )
    
    with open(resume_path) as f:
        data = json.load(f)
    
    # Format as text
    lines = []
    lines.append(f"# {data.get('name', '')}")
    c = data.get("contact", {})
    parts = [v for v in [c.get("location"), c.get("email"), c.get("linkedin")] if v]
    lines.append(" | ".join(parts))
    lines.append("")
    
    for exp in data.get("experience", []):
        lines.append(f"## {exp.get('title')} at {exp.get('company')}")
        lines.append(f"({exp.get('dates')})")
        for b in exp.get("bullets", []):
            lines.append(f"- {b}")
        lines.append("")
    
    for proj in data.get("projects", []):
        lines.append(f"## {proj.get('title')}")
        for b in proj.get("bullets", []):
            lines.append(f"- {b}")
        lines.append("")
    
    lines.append("## Skills")
    for sg in data.get("skills", []):
        items = ", ".join(sg.get("items", []))
        lines.append(f"- **{sg.get('category')}**: {items}")
    
    lines.append("")
    lines.append("## Education")
    for edu in data.get("education", []):
        lines.append(f"- {edu.get('degree')} at {edu.get('school')}")
    
    return {
        "json": data,
        "text": "\n".join(lines),
    }


def count_pdf_pages(pdf_bytes: bytes) -> int:
    """Count PDF pages using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = doc.page_count
        doc.close()
        return count
    except ImportError:
        # Fallback: rough estimate based on content length
        if len(pdf_bytes) < 15000:
            return 1
        return 2


def generate_resume_markdown(
    source_resume: str,
    jd_text: str,
    company: str,
    llm,
    max_bullets: int = 3,
    summary_sentences: int = 2,
    max_chars: int = 2700,
) -> str:
    """
    Generate a tailored resume in markdown format using the LLM.
    """
    candidate_name = "Yang Lu"
    
    prompt = f"""
    Create a professional 1-page resume for {company or "this company"}.

    CANDIDATE NAME: {candidate_name}
    USE THIS EXACT NAME AT THE TOP

    Resume Context:
    {source_resume}

    Job Description:
    {jd_text[:3000]}

    REQUIREMENTS:
    1. Start with: # {candidate_name}
    2. Summary: {summary_sentences} sentences highlighting fit for this specific role
    3. Skills: group by category, keep only skills relevant to the JD
    4. Experience: up to 3 most relevant positions, {max_bullets} bullets each
    5. Education: 1-2 lines
    6. TOTAL LIMIT: Under {max_chars} characters
    7. No filler words (leverage, utilize, synergize, streamline, facilitate)
    8. Be honest - only include real experience from the resume context
    9. Every bullet must be concise with concrete metrics where available

    OUTPUT FORMAT (Markdown):
    # {candidate_name}
    [Contact info line]

    ## Summary
    [{summary_sentences} sentences]

    ## Skills
    - **Languages**: [list]
    - **Frameworks**: [list]
    - **Tools**: [list]

    ## Experience
    ### [Position]
    **[Company]** | [Dates]
    - [bullet]

    ## Education
    [Degree] from [School]
    """
    
    response = llm.invoke(prompt)
    resume_md = response.content.strip()
    
    if not resume_md.startswith("#"):
        resume_md = f"# {candidate_name}\n\n{resume_md}"
    
    if len(resume_md) > max_chars:
        resume_md = resume_md[:max_chars]
    
    return resume_md


def markdown_to_html(md_content: str, css_mode: str = "normal") -> str:
    """
    Convert markdown resume to styled HTML for PDF generation.
    css_mode: "normal" or "tight" (smaller fonts/spacing for overflow control).
    """
    import markdown
    
    if css_mode == "tight":
        body_css = """
            font-family: 'Arial', sans-serif;
            max-width: 700px; margin: 0 auto;
            padding: 15px 25px;
            font-size: 8.5pt; line-height: 1.15;
        """
    else:
        body_css = """
            font-family: 'Arial', sans-serif;
            max-width: 700px; margin: 0 auto;
            padding: 20px 30px;
            font-size: 9pt; line-height: 1.2;
        """
    
    template = f"""
    <html>
    <head><style>
        body {{ {body_css} }}
        h1 {{ color: #2c3e50; font-size: 14pt; font-weight: bold;
              border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        h2 {{ color: #2c3e50; font-size: 10pt; font-weight: bold;
              text-transform: uppercase; margin-top: 12px; }}
        h3 {{ color: #34495e; font-size: 9pt; font-weight: bold; }}
        ul {{ padding-left: 15px; }}
        li {{ margin-bottom: 2px; font-size: 9pt; }}
        strong {{ font-weight: 600; }}
    </style></head>
    <body>{markdown.markdown(md_content)}</body>
    </html>
    """
    return template


def render_pdf(html_content: str) -> bytes:
    """
    Render HTML to PDF bytes.
    
    Tries weasyprint first (best HTML fidelity); falls back to reportlab
    (works without pango system libs, e.g. on macOS without brew deps).
    """
    try:
        from weasyprint import HTML
        return HTML(string=html_content).write_pdf()
    except (ImportError, OSError):
        pass
    
    # Fallback: reportlab-based renderer for the resume HTML.
    # Parses the simple HTML produced by markdown_to_html.
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from io import BytesIO
    import re
    
    # Simple HTML → text extraction (strip tags, keep bullets as lines)
    text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.S | re.I)
    text = re.sub(r'<h1[^>]*>', '\n# ', text)
    text = re.sub(r'<h2[^>]*>', '\n## ', text)
    text = re.sub(r'<h3[^>]*>', '\n### ', text)
    text = re.sub(r'<li[^>]*>', '\n- ', text)
    text = re.sub(r'<p[^>]*>', '\n', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    
    lines = [l.strip() for l in text.split('\n')]
    
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
        leftMargin=0.55*inch, rightMargin=0.55*inch,
        topMargin=0.35*inch, bottomMargin=0.3*inch)
    
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle('Name', parent=styles['Heading1'], fontSize=18, spaceAfter=2, textColor=HexColor('#1a1a1a'))
    contact_style = ParagraphStyle('Contact', parent=styles['Normal'], fontSize=9, textColor=HexColor('#555555'), spaceAfter=4)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=11, spaceBefore=10, spaceAfter=3, textColor=HexColor('#1a1a1a'))
    summary_style = ParagraphStyle('Summary', parent=styles['Normal'], fontSize=9.5, leading=13, textColor=HexColor('#333333'), alignment=TA_JUSTIFY, spaceAfter=4)
    company_style = ParagraphStyle('Company', parent=styles['Normal'], fontSize=9.5, leading=12, textColor=HexColor('#333333'), spaceAfter=1)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontSize=9, leading=12, leftIndent=15, bulletIndent=5, spaceAfter=2, textColor=HexColor('#444444'))
    skill_cat_style = ParagraphStyle('SkillCat', parent=styles['Normal'], fontSize=9.5, fontName='Helvetica-Bold', spaceAfter=1, textColor=HexColor('#333333'))
    skill_item_style = ParagraphStyle('SkillItem', parent=styles['Normal'], fontSize=9, leading=11, leftIndent=10, spaceAfter=3, textColor=HexColor('#555555'))
    dates_style = ParagraphStyle('Dates', parent=styles['Normal'], fontSize=9, textColor=HexColor('#777777'), spaceAfter=2)
    
    elements = []
    section_name = None
    for line in lines:
        if not line:
            continue
        if line.startswith('### '):
            elements.append(Paragraph(f"<b>{line[4:]}</b>", company_style))
        elif line.startswith('## '):
            section_name = line[3:].strip().upper()
            elements.append(Paragraph(section_name, section_style))
        elif line.startswith('# '):
            elements.append(Paragraph(line[2:], name_style))
        elif line.startswith('- '):
            if section_name == 'SKILLS':
                # Skill category line (e.g. "- **Languages**: ...")
                if '**' in line:
                    cat, rest = line[2:].split('**', 1)
                    rest = rest.lstrip(':').strip()
                    elements.append(Paragraph(cat.strip(), skill_cat_style))
                    elements.append(Paragraph(rest, skill_item_style))
                else:
                    elements.append(Paragraph(line[2:], bullet_style))
            else:
                elements.append(Paragraph(f"•  {line[2:]}", bullet_style))
        else:
            # Determine style based on context
            if section_name == 'SUMMARY':
                elements.append(Paragraph(line, summary_style))
            elif section_name == 'EDUCATION':
                elements.append(Paragraph(line, company_style))
            else:
                elements.append(Paragraph(line, company_style))
    
    doc.build(elements)
    return buf.getvalue()


def evaluate_tailored_resume(
    tailored_resume: str,
    jd_text: str,
    source_resume: str,
    llm,
) -> dict:
    """
    Evaluate the tailored resume against the JD on 5 dimensions.
    Returns scores 1-10 for each dimension plus notes.
    """
    prompt = f"""You are an expert technical recruiter. Score this tailored resume against the job description.

Job Description:
{jd_text[:2500]}

Source Resume Evidence (ground truth):
{source_resume[:2500]}

Tailored Resume:
{tailored_resume[:2500]}

Score each dimension 1-10:
1. Skills Match: Does the resume highlight JD-required skills?
2. Relevance: Does the summary/experience directly address the role?
3. Keyword Coverage: Are key JD terms present?
4. Factual Grounding: Is every claim supported by the source resume?
5. Overall Fit: Would you forward this to a hiring manager?

Return ONLY valid JSON:
{{
  "skills_match": {{"score": N, "reason": "..."}},
  "relevance": {{"score": N, "reason": "..."}},
  "keyword_coverage": {{"score": N, "reason": "..."}},
  "factual_grounding": {{"score": N, "reason": "..."}},
  "overall_fit": {{"score": N, "reason": "..."}}
}}
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    
    try:
        scores = json.loads(content)
        result = {}
        for dim in ["skills_match", "relevance", "keyword_coverage", "factual_grounding", "overall_fit"]:
            if dim in scores and "score" in scores[dim]:
                result[dim] = int(scores[dim]["score"])
                result[f"{dim}_reason"] = scores[dim].get("reason", "")
            else:
                result[dim] = 0
                result[f"{dim}_reason"] = "Parse error"
        return result
    except (json.JSONDecodeError, KeyError, ValueError):
        return {
            "skills_match": 0,
            "relevance": 0,
            "keyword_coverage": 0,
            "factual_grounding": 0,
            "overall_fit": 0,
            "notes": "Evaluation parse error",
        }
