"""
Base classes and interfaces for evaluators.

All evaluators should inherit from BaseEvaluator and implement the evaluate() method.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import json
import logging
import os
import time

from evaluators.types import EvaluationInput, EvaluationResult, EvaluatorType

logger = logging.getLogger(__name__)

# Default model per judge provider (free-tier-first: see backend/docs/EVAL_METRICS.md)
_DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "ollama": "llama3.1:8b",
}


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.

    Subclasses must implement:
    - evaluate(): The main evaluation logic
    - evaluator_type: Class property indicating LLM_BASED or RULE_BASED
    """

    evaluator_type: EvaluatorType = EvaluatorType.RULE_BASED  # Override in subclass

    def __init__(self, pass_threshold: float = 0.7):
        """
        Initialize evaluator.

        Args:
            pass_threshold: Minimum score (0.0-1.0) for evaluation to pass
        """
        self.pass_threshold = pass_threshold
        self.name = self.__class__.__name__.replace("Evaluator", "").lower()

    @abstractmethod
    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        """
        Evaluate the bot response.

        Args:
            evaluation_input: Standardized input containing question, response, and optional context

        Returns:
            EvaluationResult with score, pass/fail, and explanation
        """
        pass

    def _create_result(
        self,
        score: float,
        explanation: str,
        label: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> EvaluationResult:
        """
        Helper to create standardized evaluation result.

        Args:
            score: Normalized score (0.0 to 1.0)
            explanation: Human-readable rationale
            label: Optional category label
            metadata: Optional additional data

        Returns:
            EvaluationResult
        """
        return EvaluationResult(
            evaluator_name=self.name,
            score=score,
            passed=score >= self.pass_threshold,
            label=label,
            explanation=explanation,
            metadata=metadata or {}
        )


class LLMJudgeEvaluator(BaseEvaluator):
    """
    Base class for evaluators that use LLM-as-judge.

    Provides common functionality for calling LLM with structured prompts.
    The judge provider is pluggable (EVAL_JUDGE_PROVIDER=openai|gemini|ollama, or
    the `provider=` constructor arg) so the whole suite can run on a free-tier
    model (gemini, or a fully local ollama) without an OpenAI key.
    """

    evaluator_type = EvaluatorType.LLM_BASED

    _JUDGE_SYSTEM_PROMPT = "You are an expert evaluator. Respond with valid JSON only."
    _MAX_RETRIES = 3
    _RETRY_BACKOFF_SECONDS = 2.0

    def __init__(
        self,
        pass_threshold: float = 0.7,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize LLM-based evaluator.

        Args:
            pass_threshold: Minimum score for passing
            provider: "openai" | "gemini" | "ollama". Defaults to EVAL_JUDGE_PROVIDER
                env var, then "openai" if unset.
            model: Judge model name. Defaults to EVAL_JUDGE_MODEL env var, then a
                sensible per-provider default (see _DEFAULT_JUDGE_MODELS).
        """
        super().__init__(pass_threshold)
        self.provider = (provider or os.getenv("EVAL_JUDGE_PROVIDER", "openai")).lower()
        self.model = model or os.getenv("EVAL_JUDGE_MODEL") or _DEFAULT_JUDGE_MODELS.get(self.provider, "gpt-4o-mini")
        # Lazy: client is created on first LLM call so the evaluator can be
        # instantiated without API keys set (e.g. for unit tests or rule-based
        # subclasses that override evaluate() before calling call_llm_judge).
        self._client = None

    @abstractmethod
    def build_judge_prompt(self, evaluation_input: EvaluationInput) -> str:
        """
        Build the prompt for the LLM judge.

        This should be implemented by each subclass to create evaluator-specific prompts.

        Args:
            evaluation_input: Input data for evaluation

        Returns:
            Formatted prompt string
        """
        pass

    def call_llm_judge(self, prompt: str) -> dict:
        """
        Call the configured judge provider with the prompt and parse the JSON response.

        Expected LLM response format (JSON):
        {
            "score": float (0.0-1.0),
            "label": str,
            "explanation": str
        }

        Retries with backoff on transient errors (rate limits on free tiers in
        particular), then falls back to a zero-score "error" result so a flaky
        judge call never crashes the whole evaluation run.

        Args:
            prompt: The judge prompt

        Returns:
            Parsed JSON response from LLM
        """
        call_fn = {
            "openai": self._call_openai,
            "gemini": self._call_gemini,
            "ollama": self._call_ollama,
        }.get(self.provider)

        if call_fn is None:
            return {
                "score": 0.0,
                "label": "error",
                "explanation": f"Unknown EVAL_JUDGE_PROVIDER: {self.provider!r} (use openai|gemini|ollama)",
            }

        last_error: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES):
            try:
                raw = call_fn(prompt)
                return json.loads(raw)
            except Exception as e:
                last_error = e
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(f"Judge call failed (attempt {attempt + 1}/{self._MAX_RETRIES}), retrying in {delay:.0f}s: {e}")
                    time.sleep(delay)

        return {
            "score": 0.0,
            "label": "error",
            "explanation": f"LLM judge error ({self.provider}/{self.model}): {last_error}",
        }

    def _call_openai(self, prompt: str) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # Deterministic for consistency
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        from google.genai.types import GenerateContentConfig

        response = self._client.models.generate_content(
            model=self.model,
            contents=f"{self._JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
            config=GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return response.text

    def _call_ollama(self, prompt: str) -> str:
        if self._client is None:
            import httpx
            self._client = httpx.Client(base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"), timeout=60.0)

        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        """
        Evaluate using LLM-as-judge.

        Args:
            evaluation_input: Standardized input

        Returns:
            EvaluationResult
        """
        prompt = self.build_judge_prompt(evaluation_input)
        llm_response = self.call_llm_judge(prompt)

        return self._create_result(
            score=llm_response.get("score", 0.0),
            explanation=llm_response.get("explanation", ""),
            label=llm_response.get("label"),
            metadata={"llm_response": llm_response}
        )
