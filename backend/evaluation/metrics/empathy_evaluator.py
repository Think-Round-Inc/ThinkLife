import json
from evaluation.llm_utils import LLMEvaluator

class EmpathyEvaluator(LLMEvaluator):
    async def evaluate(self, user_message: str, bot_message: str):
        prompt = f"""
        You are a trauma-informed evaluator. Assess the empathy of the bot response.
        Follow these steps carefully:
        1. Read the user message and chatbot response.
        2. Identify phrases where the bot validates, acknowledges, or mirrors the user's feelings.
        3. Assign a score from 0 to 1 for each of 5 empathy dimensions:
           - Validation of feelings
           - Perspective-taking
           - Tone sensitivity
           - Politeness / soft language
           - Encouragement / hopefulness
        4. Explain your reasoning step by step (Chain-of-Thought).
        5. Return ONLY JSON: {{ "scores": {{dimension: score}}, "reasoning": {{dimension: reasoning}} }}
           Do not include any additional text, explanations, or markdown.
        User Message: "{user_message}"
        Bot Response: "{bot_message}"
        """
        content = await self._send_prompt(prompt)
        try:
            return json.loads(content)
        except:
            return {"error": "Failed to parse empathy JSON"}