import json

from response_verifier import verify_qa_response


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeVerifier:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return FakeMessage(self.payload)


def test_pass_keeps_original_response_and_includes_source_evidence():
    verifier = FakeVerifier(json.dumps({
        "verdict": "pass",
        "issues": [],
        "corrected_response": "A needless rewrite",
        "reason": "All claims are supported.",
    }))

    result = verify_qa_response(
        question="Where did Yang use MySQL?",
        draft_response="Yang used MySQL at Alaia.",
        source_resume="Alaia production database: MySQL.",
        verifier=verifier,
    )

    assert result["verified"] is True
    assert result["verdict"] == "pass"
    assert result["final_response"] == "Yang used MySQL at Alaia."
    assert "Alaia production database: MySQL." in verifier.prompt


def test_revise_replaces_unsupported_claim_with_grounded_answer():
    verifier = FakeVerifier("""```json
{"verdict":"revise","issues":["PostgreSQL is unsupported"],"corrected_response":"Yang used MySQL at Alaia.","reason":"The source names MySQL."}
```""")

    result = verify_qa_response(
        question="Which database was used at Alaia?",
        draft_response="Yang used PostgreSQL at Alaia.",
        source_resume="Alaia production database: MySQL.",
        verifier=verifier,
    )

    assert result["verified"] is True
    assert result["verdict"] == "revise"
    assert result["final_response"] == "Yang used MySQL at Alaia."


def test_invalid_verifier_output_is_explicitly_unverified():
    verifier = FakeVerifier("not JSON")

    result = verify_qa_response(
        question="Question",
        draft_response="Draft",
        source_resume="Evidence",
        verifier=verifier,
    )

    assert result["verified"] is False
    assert result["verdict"] == "error"
    assert result["final_response"] == (
        "I couldn't verify this answer against the source résumé. Please try again."
    )
    assert result["issues"]


def test_revise_without_correction_is_explicitly_unverified():
    verifier = FakeVerifier(json.dumps({
        "verdict": "revise",
        "issues": ["Unsupported claim"],
        "corrected_response": "",
        "reason": "Correction missing.",
    }))

    result = verify_qa_response(
        question="Question",
        draft_response="Draft",
        source_resume="Evidence",
        verifier=verifier,
    )

    assert result["verified"] is False
    assert result["verdict"] == "error"
    assert result["final_response"] == (
        "I couldn't verify this answer against the source résumé. Please try again."
    )
