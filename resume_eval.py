"""
Multi-model resume evaluation module.
Sends (resume, JD) to multiple LLMs and aggregates fit scores.

Supports three backends:
1. OpenRouter (free tier) - best effort, availability varies
2. ZAI API - reliable fallback if OPENROUTER_API_KEY = ZAI key
3. OpenCode Zen - for future use (requires ZEN_API_KEY)

The system picks models dynamically based on what responds.
"""

import requests
import json
import os
import concurrent.futures

# Model pools — tried in order, first N that respond are used
OPENROUTER_FREE_MODELS = [
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Gemma 4 31B",
        "provider": "Google",
        "backend": "openrouter",
    },
    {
        "id": "qwen/qwen3-coder:free",
        "name": "Qwen3 Coder",
        "provider": "Alibaba",
        "backend": "openrouter",
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "Llama 3.3 70B",
        "provider": "Meta",
        "backend": "openrouter",
    },
    {
        "id": "moonshotai/kimi-k2.6:free",
        "name": "Kimi K2.6",
        "provider": "Moonshot",
        "backend": "openrouter",
    },
    {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "name": "Nemotron Super 120B",
        "provider": "NVIDIA",
        "backend": "openrouter",
    },
    {
        "id": "z-ai/glm-4.5-air:free",
        "name": "GLM 4.5 Air",
        "provider": "Z.AI",
        "backend": "openrouter",
    },
]

# ZAI models (uses same OpenAI-compatible endpoint)
ZAI_MODELS = [
    {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "provider": "ZAI",
        "backend": "zai",
    },
]

# Zen models (future — requires opencode/ZEN API key)
ZEN_MODELS = [
    {
        "id": "Qwen3-235B-A22B",
        "name": "Qwen3 235B",
        "provider": "Zen",
        "backend": "zen",
    },
    {
        "id": "DeepSeek-R1",
        "name": "DeepSeek R1",
        "provider": "Zen",
        "backend": "zen",
    },
]

# Minimum models needed for a valid eval
MIN_MODELS = 1
TARGET_MODELS = 3

EVAL_PROMPT = """You are an expert technical recruiter evaluating how well a tailored resume matches a job description. Be honest and critical.

JOB DESCRIPTION:
{job_description}

SOURCE RESUME EVIDENCE:
{source_resume}

TAILORED RESUME:
{resume}

Score each dimension 1-10 with a brief reason:

1. **Skills Match**: Does the resume highlight skills the JD explicitly requires?
2. **Relevance**: Does the summary and experience directly address the role's focus?
3. **Keyword Coverage**: Are key terms from the JD present (technologies, frameworks, domains)?
4. **Factual Grounding**: Is every factual claim in the tailored resume supported by the source resume evidence? Treat absent information as unsupported; do not infer experience.
5. **Overall Fit**: Would you forward this resume to a hiring manager for this role?

Output STRICTLY as JSON:
{{
  "skills_match": {{"score": N, "reason": "..."}},
  "relevance": {{"score": N, "reason": "..."}},
  "keyword_coverage": {{"score": N, "reason": "..."}},
  "factual_grounding": {{"score": N, "reason": "..."}},
  "overall_fit": {{"score": N, "reason": "..."}}
}}

Do NOT add any text outside the JSON."""


def _call_model(model_config, prompt, api_key, api_base=None, timeout=60):
    """Call a single model and return parsed scores or None."""
    backend = model_config.get("backend", "openrouter")

    # Determine endpoint and auth
    if backend == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    elif backend == "zai":
        url = os.getenv("ZAI_BASE_URL", "https://api.zai.chat/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {os.getenv('ZAI_API_KEY', api_key)}",
            "Content-Type": "application/json",
        }
    elif backend == "zen":
        # Zen uses OpenCode's endpoint
        url = os.getenv("ZEN_BASE_URL", "http://localhost:11434/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {os.getenv('ZEN_API_KEY', '')}",
            "Content-Type": "application/json",
        }
    else:
        return {
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": f"Unknown backend: {backend}",
        }

    try:
        response = requests.post(
            url, headers=headers,
            json={
                "model": model_config["id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        data = response.json()

        if "error" in data:
            return {
                "model": model_config["name"],
                "provider": model_config["provider"],
                "error": data["error"].get("message", "unknown error")[:100],
            }

        content = data["choices"][0]["message"]["content"].strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        scores = json.loads(content)

        # Validate all expected keys exist
        expected = ["skills_match", "relevance", "keyword_coverage", "factual_grounding", "overall_fit"]
        for key in expected:
            if key not in scores or "score" not in scores[key]:
                return {
                    "model": model_config["name"],
                    "provider": model_config["provider"],
                    "error": f"Missing key: {key}",
                }

        return {
            "model": model_config["name"],
            "provider": model_config["provider"],
            "scores": scores,
        }
    except json.JSONDecodeError as e:
        return {
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": f"JSON parse error: {str(e)[:100]}",
        }
    except requests.exceptions.Timeout:
        return {
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": "Timeout (60s)",
        }
    except Exception as e:
        return {
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": str(e)[:100],
        }


def evaluate_resume(resume_markdown, job_description, source_resume, api_key=None):
    """
    Run multi-model evaluation on a generated resume vs JD.
    
    Dynamically selects models from available backends:
    - OpenRouter free tier (if OPENROUTER_API_KEY set)
    - ZAI (if ZAI_API_KEY set)  
    - Zen (if ZEN_API_KEY set)

    Args:
        resume_markdown: The generated resume content (markdown string)
        job_description: The original job description
        source_resume: Canonical resume evidence used to check every factual claim
        api_key: OpenRouter API key (falls back to env var)

    Returns:
        dict with 'results' (per-model), 'aggregate' (averaged scores),
        'disagreements' (dimensions with high variance), 'confidence'
    """
    if not source_resume.strip():
        return {
            "error": "Source resume evidence is required",
            "results": [],
            "aggregate": {},
        }

    api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    zai_key = os.getenv("ZAI_API_KEY", "")
    zen_key = os.getenv("ZEN_API_KEY", "")

    # Build model pool from available backends
    models = []
    if api_key:
        models.extend(OPENROUTER_FREE_MODELS)
    if zai_key:
        models.extend(ZAI_MODELS)
    if zen_key:
        models.extend(ZEN_MODELS)

    if not models:
        return {
            "error": "No API keys configured (need OPENROUTER_API_KEY, ZAI_API_KEY, or ZEN_API_KEY)",
            "results": [],
            "aggregate": {},
        }

    # Limit evidence sharing and API traffic to the advertised judge count.
    models = models[:TARGET_MODELS]

    prompt = EVAL_PROMPT.format(
        job_description=job_description[:3000],
        resume=resume_markdown[:3000],
        source_resume=source_resume,
    )

    # Run all models in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(_call_model, model, prompt, api_key): model
            for model in models
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Calculate aggregate scores (average across successful models)
    dimensions = ["skills_match", "relevance", "keyword_coverage", "factual_grounding", "overall_fit"]
    successful = [r for r in results if "scores" in r]
    failed = [r for r in results if "error" in r]

    if not successful:
        return {
            "error": "All models failed",
            "results": results,
            "aggregate": {},
        }

    aggregate = {}
    for dim in dimensions:
        scores_list = [r["scores"][dim]["score"] for r in successful]
        avg = sum(scores_list) / len(scores_list)
        reasons = [r["scores"][dim]["reason"] for r in successful]
        aggregate[dim] = {
            "score": round(avg, 1),
            "scores": scores_list,
            "reasons": reasons,
        }

    # Find disagreements (variance > 2 between models)
    disagreements = []
    for dim in dimensions:
        scores_list = aggregate[dim]["scores"]
        if max(scores_list) - min(scores_list) > 2:
            disagreements.append({
                "dimension": dim.replace("_", " ").title(),
                "spread": f"{min(scores_list)}-{max(scores_list)}",
                "model_reasons": {
                    r["model"]: r["scores"][dim]["reason"]
                    for r in successful
                },
            })

    # Overall confidence (inverse of disagreement count)
    confidence = max(0, 100 - len(disagreements) * 20)

    return {
        "results": results,
        "aggregate": aggregate,
        "disagreements": disagreements,
        "confidence": confidence,
        "models_succeeded": len(successful),
        "models_failed": len(failed),
    }


def format_eval_report(eval_result):
    """Format evaluation results as a readable markdown report."""
    if "error" in eval_result and not eval_result.get("aggregate"):
        return f"⚠️ Evaluation failed: {eval_result['error']}"

    lines = []

    # Aggregate scores
    aggregate = eval_result.get("aggregate", {})
    if aggregate:
        lines.append("### 📊 Multi-Model Evaluation")
        lines.append(f"**Models:** {eval_result['models_succeeded']} succeeded, {eval_result['models_failed']} failed")
        lines.append("")

        dim_labels = {
            "skills_match": "🛠 Skills Match",
            "relevance": "🎯 Relevance",
            "keyword_coverage": "🔑 Keyword Coverage",
            "factual_grounding": "✅ Factual Grounding",
            "overall_fit": "📈 Overall Fit",
        }

        lines.append("| Dimension | Score | Models |")
        lines.append("|---|---|---|")

        for dim, label in dim_labels.items():
            if dim in aggregate:
                data = aggregate[dim]
                score = data["score"]
                scores_str = ", ".join(str(s) for s in data["scores"])
                # Color coding
                if score >= 7:
                    badge = "🟢"
                elif score >= 5:
                    badge = "🟡"
                else:
                    badge = "🔴"
                lines.append(f"| {label} | {badge} {score}/10 | {scores_str} |")

        lines.append("")

    # Per-model details
    results = eval_result.get("results", [])
    successful = [r for r in results if "scores" in r]
    if successful:
        lines.append("### 🤖 Model Details")
        for r in successful:
            lines.append(f"**{r['model']}** ({r['provider']})")
            for dim, data in r["scores"].items():
                lines.append(f"- {dim.replace('_', ' ').title()}: **{data['score']}**/10 — {data['reason']}")
            lines.append("")

    # Disagreements
    disagreements = eval_result.get("disagreements", [])
    if disagreements:
        lines.append("### ⚠️ Model Disagreements")
        lines.append("These dimensions had significant divergence between models:")
        for d in disagreements:
            lines.append(f"**{d['dimension']}** (spread: {d['spread']})")
            for model, reason in d["model_reasons"].items():
                lines.append(f"- {model}: {reason}")
            lines.append("")

    # Failed models
    failed = [r for r in results if "error" in r]
    if failed:
        lines.append("### ❌ Failed Models")
        for r in failed:
            lines.append(f"- **{r['model']}**: {r['error']}")

    return "\n".join(lines)
