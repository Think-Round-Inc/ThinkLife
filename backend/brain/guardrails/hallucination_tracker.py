
"""
User Story 2.2 - Task 4
Hallucination Observation Tracker

Tracks suspected hallucinations without storing
user messages or generated response text.
"""

import json
import logging
import math
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_STATUSES = {"flagged", "unflagged", "unevaluated"}

# Only predefined reason codes are permitted.
# This prevents arbitrary response text from being logged.
VALID_REASONS = {
    "unsupported_claim",
    "contradicted_by_evidence",
    "insufficient_evidence",
    "grounded_in_evidence",
    "not_applicable",
}


class HallucinationTracker:
    def __init__(self, log_path="logs/hallucination_observations.jsonl"):
        self.log_path = Path(log_path)

    def observe(
        self,
        status,
        agent_name="zoe",
        groundedness_score=None,
        reason=None,
    ):
        """Record an observation without storing response text."""

        # Validate observation status.
        if status not in VALID_STATUSES:
            raise ValueError("Invalid observation status")

        # Restrict reasons to predefined, non-sensitive codes.
        if reason is not None and reason not in VALID_REASONS:
            raise ValueError("Invalid observation reason")

        # Validate optional groundedness score.
        if groundedness_score is not None:
            if (
                isinstance(groundedness_score, bool)
                or not isinstance(groundedness_score, (int, float))
                or not math.isfinite(groundedness_score)
                or not 0 <= groundedness_score <= 1
            ):
                raise ValueError(
                    "Groundedness score must be between 0 and 1"
                )

        observation = {
            "observation_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "agent_name": agent_name,
            "status": status,
            "groundedness_score": groundedness_score,
            "reason": reason,
        }

        # Track whether the write completed successfully.
        # This field is returned but not stored in the JSONL file.
        observation["persisted"] = False

        try:
            self.log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Exclude the operational persisted flag from the log.
            stored_observation = {
                key: value
                for key, value in observation.items()
                if key != "persisted"
            }

            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(stored_observation) + "\n"
                )

            observation["persisted"] = True

        except (OSError, TypeError, ValueError):
            logger.exception(
                "Failed to persist hallucination observation"
            )

        return observation

    def load_observations(self):
        """Read valid observations from the JSONL log."""

        if not self.log_path.exists():
            return []

        observations = []

        try:
            with self.log_path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        observation = json.loads(line)

                        if (
                            isinstance(observation, dict)
                            and observation.get("status")
                            in VALID_STATUSES
                        ):
                            observations.append(observation)

                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Skipping malformed observation"
                        )

        except OSError:
            logger.exception(
                "Failed to read hallucination observations"
            )

        return observations

    def hallucination_rate(self, agent_name=None, since_timestamp=None):
        """Calculate the suspected hallucination observation rate."""

        observations = self.load_observations()

        if agent_name is not None:
            observations = [
                observation
                for observation in observations
                if observation.get("agent_name") == agent_name
            ]

        if since_timestamp is not None:
            observations = [
                observation
                for observation in observations
                if isinstance(
                    observation.get("timestamp"), (int, float)
                )
                and not isinstance(
                    observation.get("timestamp"), bool
                )
                and math.isfinite(observation["timestamp"])
                and observation["timestamp"] >= since_timestamp
            ]

        flagged = sum(
            observation["status"] == "flagged"
            for observation in observations
        )

        unflagged = sum(
            observation["status"] == "unflagged"
            for observation in observations
        )

        unevaluated = sum(
            observation["status"] == "unevaluated"
            for observation in observations
        )

        evaluated = flagged + unflagged

        rate = flagged / evaluated if evaluated else None

        return {
            "total": len(observations),
            "evaluated": evaluated,
            "flagged": flagged,
            "unflagged": unflagged,
            "unevaluated": unevaluated,
            "rate": rate,
        }

