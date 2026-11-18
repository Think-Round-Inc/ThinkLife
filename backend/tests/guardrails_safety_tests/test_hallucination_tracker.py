
"""
User Story 2.2 - Task 4
Hallucination Observation Tracker Tests

Tests:
- Observation recording
- Hallucination rate calculation
- Unevaluated responses
- Invalid observation statuses
- Invalid groundedness scores
- Agent filtering
- Malformed log handling
- Invalid reason rejection
- Successful persistence
- Failed persistence handling
"""

import json

import pytest

from brain.guardrails.hallucination_tracker import HallucinationTracker


def test_observation_recording(tmp_path):
    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    observation = tracker.observe(
        status="flagged",
        agent_name="zoe",
        groundedness_score=0.3,
        reason="unsupported_claim",
    )

    assert observation["status"] == "flagged"
    assert observation["agent_name"] == "zoe"
    assert observation["groundedness_score"] == 0.3
    assert len(tracker.load_observations()) == 1


def test_hallucination_rate(tmp_path):
    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    tracker.observe(status="flagged")
    tracker.observe(status="unflagged")
    tracker.observe(status="unflagged")
    tracker.observe(status="unevaluated")

    result = tracker.hallucination_rate()

    assert result["total"] == 4
    assert result["evaluated"] == 3
    assert result["flagged"] == 1
    assert result["unflagged"] == 2
    assert result["unevaluated"] == 1
    assert result["rate"] == pytest.approx(1 / 3)


def test_no_evaluated_responses(tmp_path):
    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    tracker.observe(status="unevaluated")

    result = tracker.hallucination_rate()

    assert result["evaluated"] == 0
    assert result["rate"] is None


@pytest.mark.parametrize(
    "invalid_status",
    ["unknown", "", None],
)
def test_invalid_status_rejected(tmp_path, invalid_status):
    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    with pytest.raises(ValueError):
        tracker.observe(status=invalid_status)


@pytest.mark.parametrize(
    "invalid_score",
    [-0.1, 1.1, float("nan"), True],
)
def test_invalid_groundedness_score(tmp_path, invalid_score):
    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    with pytest.raises(ValueError):
        tracker.observe(
            status="flagged",
            groundedness_score=invalid_score,
        )


def test_agent_filtering(tmp_path):
    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    tracker.observe(status="flagged", agent_name="zoe")
    tracker.observe(status="unflagged", agent_name="other")

    result = tracker.hallucination_rate(agent_name="zoe")

    assert result["total"] == 1
    assert result["flagged"] == 1
    assert result["rate"] == 1.0


def test_malformed_log_line_is_skipped(tmp_path):
    log_path = tmp_path / "observations.jsonl"

    log_path.write_text(
        'not valid json\n'
        + json.dumps({"status": "flagged"}) + "\n",
        encoding="utf-8",
    )

    tracker = HallucinationTracker(log_path=log_path)
    result = tracker.hallucination_rate()

    assert result["total"] == 1
    assert result["flagged"] == 1


# --------------------------------------------------
# Additional Task 4 Privacy and Reliability Tests
# --------------------------------------------------


def test_invalid_reason_rejected(tmp_path):
    """Reject arbitrary reason text to protect privacy."""

    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    with pytest.raises(ValueError):
        tracker.observe(
            status="flagged",
            reason="Sensitive response text",
        )

    assert tracker.load_observations() == []


def test_successful_persistence(tmp_path):
    """Verify successful observation logging."""

    tracker = HallucinationTracker(
        log_path=tmp_path / "observations.jsonl"
    )

    observation = tracker.observe(
        status="flagged",
        reason="unsupported_claim",
    )

    assert observation["persisted"] is True

    stored = tracker.load_observations()

    assert len(stored) == 1
    assert "persisted" not in stored[0]
    assert stored[0]["reason"] == "unsupported_claim"


def test_failed_persistence_does_not_raise(tmp_path):
    """Logging failures should not interrupt observation recording."""

    # A directory cannot be opened as a regular log file.
    invalid_log_path = tmp_path / "not_a_file"
    invalid_log_path.mkdir()

    tracker = HallucinationTracker(
        log_path=invalid_log_path
    )

    observation = tracker.observe(
        status="unevaluated",
        reason="insufficient_evidence",
    )

    assert observation["persisted"] is False

