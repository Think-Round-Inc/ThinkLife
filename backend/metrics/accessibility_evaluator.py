import json
from typing import Dict, Any, Optional
from datetime import datetime
from evaluation.llm_utils import LLMEvaluator

class AccessibilityEvaluator(LLMEvaluator):
    async def evaluate(
        self,
        user_message: str,
        bot_message: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not bot_message or not bot_message.strip():
            return {
                "accessibility_score": 0.0,
                "error": "Empty response provided",
                "evaluation_timestamp": datetime.now().isoformat()
            }
        try:
            evaluation_prompt = f"""
Evaluate the language accessibility of this empathetic bot response:
USER MESSAGE: "{user_message}"
BOT RESPONSE: "{bot_message}"
Is this bot response linguistically accessible and clear for someone seeking emotional support?
Consider:
- Is the language clear and understandable?
- Would this make sense to someone in emotional distress?
- Are there unnecessarily complex or confusing terms?
- Is the tone appropriate for the emotional context?
Return only JSON:
{{
    "accessibility_score": <number between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the score>"
}}
"""
            llm_response = await self._send_prompt(evaluation_prompt)
            try:
                parsed_response = json.loads(llm_response)
                if "error" in parsed_response:
                    return {"accessibility_score": 0.5, "reasoning": parsed_response["error"]}
            except json.JSONDecodeError:
                pass
            parsed_result = self._parse_llm_response(llm_response)
            if parsed_result:
                result = parsed_result
            else:
                result = {"accessibility_score": 0.5, "reasoning": "Failed to parse LLM response"}
            result.update({
                "evaluation_timestamp": datetime.now().isoformat(),
                "response_length": len(bot_message.split()),
                "evaluation_method": "llm_judge"
            })
            return result
        except Exception as e:
            return {
                "accessibility_score": 0.0,
                "error": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }

    def _parse_llm_response(self, llm_response: str) -> Optional[Dict[str, Any]]:
        try:
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                parsed = json.loads(json_str)
                if "accessibility_score" in parsed:
                    score = float(parsed["accessibility_score"])
                    parsed["accessibility_score"] = max(0.0, min(1.0, score))
                    return parsed
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return None