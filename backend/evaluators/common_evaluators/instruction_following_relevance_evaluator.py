"""
Instruction Following and Relevance Evaluator

Type: LLM-BASED

Epic 4.2.3 metric. See backend/docs/EVAL_METRICS.md for the full framework.

Scores 4 sub-dimensions (each 0-1) and combines them into a weighted composite:
- directness (0.4): answers what was actually asked
- completeness (0.3): covers the key sub-parts of the question
- focus (0.2): no irrelevant filler or tangents
- context_fit (0.1): uses conversation_history / user_context when present

composite = 0.4*directness + 0.3*completeness + 0.2*focus + 0.1*context_fit
pass_threshold = 0.75

For Zoe bot: Checks that responses address emotional needs without going off-topic.
Evaluates appropriate redirects for out-of-scope requests.

For FAQ bot: Will verify responses directly answer the documentation question.
"""

from evaluators.base import LLMJudgeEvaluator
from evaluators.types import EvaluationInput, EvaluationResult

_WEIGHTS = {"directness": 0.4, "completeness": 0.3, "focus": 0.2, "context_fit": 0.1}


class InstructionFollowingRelevanceEvaluator(LLMJudgeEvaluator):
    """
    LLM-based evaluator for instruction following and relevance, with 4 scored
    sub-dimensions so dashboards can see which one regresses.
    """

    def __init__(self, pass_threshold: float = 0.75, provider=None, model=None):
        """
        Initialize instruction following evaluator.

        Args:
            pass_threshold: Minimum composite score to pass (default 0.75)
        """
        super().__init__(pass_threshold=pass_threshold, provider=provider, model=model)

    def build_judge_prompt(self, evaluation_input: EvaluationInput) -> str:
        """
        Build the LLM judge prompt for relevance evaluation.

        Args:
            evaluation_input: Input containing question and bot response

        Returns:
            Formatted prompt string
        """
        history_section = ""
        if evaluation_input.conversation_history:
            turns = "\n".join(
                f"  {m.get('role', 'user')}: {m.get('content', '')}"
                for m in evaluation_input.conversation_history[-6:]
            )
            history_section = f"\nCONVERSATION HISTORY (most recent last):\n{turns}\n"

        context_section = ""
        if evaluation_input.user_context:
            context_section = f"\nUSER CONTEXT: {evaluation_input.user_context}\n"

        prompt = f"""You are a relevance evaluator for chatbot responses. Score the response on 4 independent sub-dimensions.
{history_section}{context_section}
USER QUESTION:
{evaluation_input.question}

BOT RESPONSE:
{evaluation_input.bot_response}

Score each sub-dimension from 0.0 to 1.0:

1. **directness**: Does the response answer what was actually asked (not a different, adjacent question)?
2. **completeness**: Does it cover the key sub-parts of the question? (not dismissively brief, not padded)
3. **focus**: Is it free of irrelevant filler or tangents that don't serve the user's question?
4. **context_fit**: If conversation history or user context is given above, does the response actually use it (not ignore it)? If no history/context was given, score this 1.0 (not applicable).

Provide your evaluation as JSON:
{{
    "directness": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "focus": <float 0.0-1.0>,
    "context_fit": <float 0.0-1.0>,
    "label": "<highly_relevant|somewhat_relevant|off_topic>",
    "explanation": "<brief explanation covering all 4 dimensions>"
}}
"""
        return prompt

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        prompt = self.build_judge_prompt(evaluation_input)
        llm_response = self.call_llm_judge(prompt)

        sub_scores = {k: float(llm_response.get(k, 0.0)) for k in _WEIGHTS}
        composite = sum(sub_scores[k] * w for k, w in _WEIGHTS.items())

        return self._create_result(
            score=composite,
            explanation=llm_response.get("explanation", ""),
            label=llm_response.get("label"),
            metadata={"sub_scores": sub_scores, "llm_response": llm_response},
        )
