"""Convert resume.json to plain text for the resume agent."""

import json
import os

RESUME_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "Documents", "interview-prep", "resume.json")

def resume_to_text(path=RESUME_JSON):
    with open(path) as f:
        data = json.load(f)

    lines = []

    # Name + contact
    lines.append(f"# {data['name']}")
    c = data.get("contact", {})
    loc = c.get("location", "")
    email = c.get("email", "")
    linkedin = c.get("linkedin", "")
    site = c.get("website", "")
    hf = c.get("huggingface", "")
    lines.append(f"{loc} | {email} | {linkedin} | {site} | {hf}")
    lines.append("")

    # Summary
    summary = data.get("summary", "")
    if summary:
        lines.append("## Summary")
        lines.append(summary)
        lines.append("")

    # Experience
    lines.append("## Experience")
    for exp in data.get("experience", []):
        title = exp.get("title", "")
        company = exp.get("company", "")
        loc = exp.get("location", "")
        dates = exp.get("dates", "")
        lines.append(f"### {title}")
        lines.append(f"**{company}** | {loc} | {dates}")
        for bullet in exp.get("bullets", []):
            lines.append(f"- {bullet}")
        lines.append("")

    # Projects
    lines.append("## Projects")
    for proj in data.get("projects", []):
        title = proj.get("title", "")
        subtitle = proj.get("subtitle", "")
        lines.append(f"### {title}")
        if subtitle:
            lines.append(f"*{subtitle}*")
        for bullet in proj.get("bullets", []):
            lines.append(f"- {bullet}")
        lines.append("")

    # Education
    lines.append("## Education")
    for edu in data.get("education", []):
        school = edu.get("school", "")
        degree = edu.get("degree", "")
        dates = edu.get("dates", "")
        lines.append(f"- {degree} — {school} ({dates})")
    lines.append("")

    # Skills
    lines.append("## Skills")
    for skill_group in data.get("skills", []):
        cat = skill_group.get("category", "")
        items = skill_group.get("items", [])
        lines.append(f"- **{cat}**: {', '.join(items)}")

    return "\n".join(lines)


if __name__ == "__main__":
    text = resume_to_text()
    print(text)
    print("\n\n---")
    print(f"Total chars: {len(text)}")
