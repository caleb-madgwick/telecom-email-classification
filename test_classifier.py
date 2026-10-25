"""Tests for the classifier and evaluation metrics.

No real API calls are made: SDK clients are replaced with MagicMock objects
that return canned responses, so these tests are free and run offline.
"""

import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx2  # the HTTP library both SDKs are built on
import openai
import pytest

import classifier
from classifier import (
    CATEGORIES,
    SYSTEM_PROMPT,
    ClassificationError,
    MissingAPIKeyError,
    classify_email,
    load_emails,
    validate_response,
)
from evaluator import (
    DATASETS,
    accuracy,
    estimated_cost,
    find_disagreements,
    precision_recall,
    select_sample,
)

EMAIL = {
    "id": 1,
    "subject": "Charged twice",
    "body": "I was charged twice this month.",
    "expected_category": "billing",
    "expected_escalation": False,
}
VALID_JSON = json.dumps(
    {
        "category": "billing",
        "escalate_to_human": False,
        "reasoning": "Customer reports a duplicate charge.",
    }
)


# --- Fake API responses -----------------------------------------------------


def fake_openai_client(output_text, input_tokens=120, output_tokens=25):
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=output_text,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return client


def fake_anthropic_client(text, input_tokens=130, output_tokens=30):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return client


def api_request():
    return httpx2.Request("POST", "https://api.example.com")


def status_response(status_code):
    return httpx2.Response(status_code, request=api_request())


# --- Dataset ----------------------------------------------------------------


def test_dataset_has_20_emails_per_category():
    emails = load_emails()
    counts = Counter(e["expected_category"] for e in emails)
    assert counts == {category: 20 for category in CATEGORIES}


def test_dataset_ids_are_unique():
    ids = [e["id"] for e in load_emails()]
    assert len(ids) == len(set(ids))


def test_hard_dataset_is_valid_and_separate_from_original():
    hard_path, _ = DATASETS["hard"]
    hard = load_emails(hard_path)  # also checks every label is valid
    hard_ids = {e["id"] for e in hard}
    original_ids = {e["id"] for e in load_emails()}

    assert len(hard) == 40
    assert len(hard_ids) == 40  # unique
    assert not hard_ids & original_ids  # no overlap with the original 60
    assert all(e.get("note") for e in hard)  # every label is explained


def test_load_emails_rejects_unknown_category(tmp_path):
    path = tmp_path / "emails.json"
    path.write_text(json.dumps([{**EMAIL, "expected_category": "sales"}]))
    with pytest.raises(ValueError, match="unknown category"):
        load_emails(path)


def test_load_emails_rejects_missing_fields(tmp_path):
    path = tmp_path / "emails.json"
    path.write_text(json.dumps([{"id": 1, "subject": "Hi"}]))
    with pytest.raises(ValueError, match="missing"):
        load_emails(path)


def test_load_emails_rejects_non_boolean_escalation(tmp_path):
    path = tmp_path / "emails.json"
    path.write_text(json.dumps([{**EMAIL, "expected_escalation": "yes"}]))
    with pytest.raises(ValueError, match="true or false"):
        load_emails(path)


# --- Response validation ----------------------------------------------------


def test_validate_response_accepts_valid_json():
    assert validate_response(VALID_JSON) == {
        "category": "billing",
        "escalate_to_human": False,
        "reasoning": "Customer reports a duplicate charge.",
    }


@pytest.mark.parametrize(
    "raw_text",
    [
        "billing",  # not JSON at all
        '```json\n{"category": "billing", "escalate_to_human": false, "reasoning": "x"}\n```',
        '["billing"]',  # JSON, but not an object
        '{"category": "sales", "escalate_to_human": false, "reasoning": "x"}',
        '{"escalate_to_human": false, "reasoning": "x"}',  # missing category
        '{"category": "billing", "reasoning": "x"}',  # missing escalate_to_human
        '{"category": "billing", "escalate_to_human": "true", "reasoning": "x"}',
        '{"category": "billing", "escalate_to_human": 1, "reasoning": "x"}',
        '{"category": "billing", "escalate_to_human": false}',  # missing reasoning
        '{"category": "billing", "escalate_to_human": false, "reasoning": "  "}',
    ],
)
def test_validate_response_rejects_invalid_output(raw_text):
    with pytest.raises(ClassificationError):
        validate_response(raw_text)


# --- API keys -----------------------------------------------------------------


def test_missing_api_key_fails_clearly(monkeypatch):
    # Stop load_dotenv from re-reading the real .env file during the test.
    monkeypatch.setattr(classifier, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY is not set"):
        classifier.create_client("openai")


# --- classify_email with mocked providers -------------------------------------


def test_classify_email_with_openai():
    client = fake_openai_client(VALID_JSON)
    result = classify_email(EMAIL, "openai", client)

    assert result["predicted"] == "billing"
    assert result["predicted_escalation"] is False
    assert result["reasoning"] == "Customer reports a duplicate charge."
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 25
    assert result["latency_s"] is not None
    assert result["error"] is None


def test_classify_email_with_anthropic():
    client = fake_anthropic_client(VALID_JSON)
    result = classify_email(EMAIL, "anthropic", client)

    assert result["predicted"] == "billing"
    assert result["input_tokens"] == 130
    assert result["error"] is None


def test_both_providers_receive_the_same_prompt_and_email():
    openai_client = fake_openai_client(VALID_JSON)
    anthropic_client = fake_anthropic_client(VALID_JSON)
    classify_email(EMAIL, "openai", openai_client)
    classify_email(EMAIL, "anthropic", anthropic_client)

    openai_kwargs = openai_client.responses.create.call_args.kwargs
    anthropic_kwargs = anthropic_client.messages.create.call_args.kwargs
    assert openai_kwargs["instructions"] == SYSTEM_PROMPT
    assert anthropic_kwargs["system"] == SYSTEM_PROMPT
    assert openai_kwargs["input"] == anthropic_kwargs["messages"][0]["content"]


def test_invalid_model_output_is_recorded_as_error():
    client = fake_anthropic_client(
        '{"category": "sales", "escalate_to_human": false, "reasoning": "x"}'
    )
    result = classify_email(EMAIL, "anthropic", client)

    assert result["predicted"] is None
    assert result["error"].startswith("ClassificationError")
    # The call still happened, so its tokens still count towards cost.
    assert result["output_tokens"] == 30


def test_transient_api_error_is_recorded_not_raised():
    client = MagicMock()
    client.messages.create.side_effect = anthropic.APIConnectionError(
        request=api_request()
    )
    result = classify_email(EMAIL, "anthropic", client)

    assert result["predicted"] is None
    assert result["error"].startswith("APIConnectionError")
    assert result["latency_s"] is None


def test_authentication_error_stops_the_run():
    client = MagicMock()
    client.responses.create.side_effect = openai.AuthenticationError(
        "Invalid API key", response=status_response(401), body=None
    )
    with pytest.raises(openai.AuthenticationError):
        classify_email(EMAIL, "openai", client)


# --- Metrics ------------------------------------------------------------------


def make_result(expected, predicted, model="claude-haiku-5-5", **overrides):
    return {
        "expected": expected,
        "predicted": predicted,
        "expected_escalation": False,
        "predicted_escalation": False,
        "model": model,
        "input_tokens": 0,
        "output_tokens": 0,
        **overrides,
    }


# 4 emails: 2 correct, 1 wrong, 1 failed (predicted=None).
RESULTS = [
    make_result("billing", "billing"),
    make_result("billing", "general_enquiry"),
    make_result("technical_support", "technical_support"),
    make_result("general_enquiry", None),
]


def test_accuracy_counts_errors_as_wrong():
    assert accuracy(RESULTS) == 0.5


def test_precision_and_recall():
    # billing: predicted once (correct), actually twice -> P = 1/1, R = 1/2
    assert precision_recall(RESULTS, "billing") == (1.0, 0.5)
    # general_enquiry: predicted once (wrong), actually once (missed) -> 0, 0
    assert precision_recall(RESULTS, "general_enquiry") == (0.0, 0.0)


def test_precision_and_recall_are_none_when_undefined():
    # Nothing is predicted or expected to escalate, so neither can be calculated.
    fields = {"predicted": "predicted_escalation", "expected": "expected_escalation"}
    assert precision_recall(RESULTS, True, **fields) == (None, None)


def test_escalation_metrics():
    def escalation(expected, predicted):
        return make_result(
            "billing", "billing", expected_escalation=expected, predicted_escalation=predicted
        )

    results = [
        escalation(True, True),  # caught
        escalation(True, False),  # missed
        escalation(False, True),  # false alarm
        escalation(False, False),  # correctly left alone
        escalation(False, None),  # failed call: never counts as escalated
    ]
    fields = {"predicted": "predicted_escalation", "expected": "expected_escalation"}

    assert accuracy(results, **fields) == 2 / 5
    # Predicted True twice, 1 correct -> P = 1/2. Truly True twice, 1 found -> R = 1/2.
    assert precision_recall(results, True, **fields) == (0.5, 0.5)


def test_estimated_cost():
    result = make_result(
        "billing", "billing", input_tokens=1_000_000, output_tokens=1_000_000
    )
    # $0.10 per 1M input + $0.50 per 1M output
    assert estimated_cost(result) == pytest.approx(0.60)


def test_find_disagreements():
    results = [
        # Email 1: both models give the same answer.
        make_result("billing", "billing", model="a", email_id=1),
        make_result("billing", "billing", model="b", email_id=1),
        # Email 2: different categories (model a is right, b is wrong).
        make_result("billing", "billing", model="a", email_id=2),
        make_result("billing", "general_enquiry", model="b", email_id=2),
        # Email 3: same category, different escalation decision.
        make_result("billing", "billing", model="a", email_id=3, predicted_escalation=True),
        make_result("billing", "billing", model="b", email_id=3),
    ]
    assert sorted(find_disagreements(results)) == [2, 3]


def test_select_sample_takes_n_from_each_category():
    sample = select_sample(load_emails(), per_category=2)
    counts = Counter(e["expected_category"] for e in sample)
    assert counts == {category: 2 for category in CATEGORIES}
