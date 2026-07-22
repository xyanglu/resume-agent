import resume_eval


def _successful_result(model, prompt, api_key):
    assert "SOURCE RESUME EVIDENCE" in prompt
    assert "Alaia used MySQL" in prompt
    return {
        "model": model["name"],
        "provider": model["provider"],
        "scores": {
            name: {"score": 8, "reason": "Supported"}
            for name in (
                "skills_match",
                "relevance",
                "keyword_coverage",
                "factual_grounding",
                "overall_fit",
            )
        },
    }


def test_resume_evaluation_includes_source_resume(monkeypatch):
    monkeypatch.setattr(resume_eval, "OPENROUTER_FREE_MODELS", [
        {"id": "fake", "name": "Fake", "provider": "Test", "backend": "openrouter"}
    ])
    monkeypatch.setattr(resume_eval, "_call_model", _successful_result)

    result = resume_eval.evaluate_resume(
        resume_markdown="Tailored resume",
        job_description="Job description",
        source_resume="Alaia used MySQL",
        api_key="test-key",
    )

    assert result["models_succeeded"] == 1
    assert result["aggregate"]["factual_grounding"]["score"] == 8


def test_resume_evaluation_requires_source_evidence():
    result = resume_eval.evaluate_resume(
        resume_markdown="Tailored resume",
        job_description="Job description",
        source_resume="",
        api_key="test-key",
    )

    assert result["error"] == "Source resume evidence is required"
    assert result["aggregate"] == {}


def test_resume_evaluation_limits_source_sharing_to_target_model_count(monkeypatch):
    models = [
        {"id": f"fake-{index}", "name": f"Fake {index}", "provider": "Test"}
        for index in range(5)
    ]
    calls = []

    def record_call(model, prompt, api_key):
        calls.append(model["id"])
        return _successful_result(model, prompt, api_key)

    monkeypatch.setattr(resume_eval, "OPENROUTER_FREE_MODELS", models)
    monkeypatch.setattr(resume_eval, "_call_model", record_call)

    result = resume_eval.evaluate_resume(
        resume_markdown="Tailored resume",
        job_description="Job description",
        source_resume="Alaia used MySQL",
        api_key="test-key",
    )

    assert result["models_succeeded"] == resume_eval.TARGET_MODELS
    assert len(calls) == resume_eval.TARGET_MODELS
