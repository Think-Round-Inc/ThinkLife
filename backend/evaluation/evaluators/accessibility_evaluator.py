"""
Accessibility Evaluator

Evaluates language accessibility and clarity in bot responses using LLM-as-judge
with LangFuse observability for tracking communication effectiveness.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from langfuse.decorators import observe, langfuse_context

logger = logging.getLogger(__name__)


class AccessibilityEvaluator:
    """Evaluates language accessibility using LLM-as-judge"""

    def __init__(self, llm_client=None):
        """
        Initialize the accessibility evaluator
        
        Args:
            llm_client: LLM client for running evaluations
        """
        self.llm_client = llm_client

    @observe(name="evaluate_accessibility")
    async def evaluate(
        self,
        user_message: str,
        bot_message: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate language accessibility
        
        Args:
            user_message: User's message for context
            bot_message: Bot response text to evaluate
            context: Additional context including emotional state
            
        Returns:
            Dict containing accessibility score and reasoning
        """
        # Validate input
        if not bot_message or not bot_message.strip():
            result = {
                "accessibility_score": 0.0,
                "error": "Empty response provided",
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message="Empty response",
                output=result
            )
            
            return result
        
        # Track in LangFuse
        langfuse_context.update_current_trace(
            name="accessibility_evaluation",
            user_id=context.get("user_id") if context else None,
            session_id=context.get("session_id") if context else None,
            metadata={
                "evaluator": "accessibility",
                "message_length": len(bot_message),
                "word_count": len(bot_message.split())
            }
        )
        
        try:
            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(user_message, bot_message)
            
            # Call LLM-as-judge
            with langfuse_context.observe_llm_call(
                name="accessibility_llm_judge",
                model="gemini-1.5-flash",
                input=prompt
            ) as llm_span:
                result = await self._call_llm(prompt)
                llm_span.update(output=result)
            
            # Parse result
            parsed_result = self._parse_result(result)
            
            # Add metadata
            parsed_result.update({
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluator": "accessibility",
                "response_length_words": len(bot_message.split()),
                "response_length_chars": len(bot_message),
                "evaluation_method": "llm_judge"
            })
            
            # Track output
            langfuse_context.update_current_observation(
                output=parsed_result,
                metadata={
                    "success": True,
                    "accessibility_score": parsed_result.get("accessibility_score", 0.0)
                }
            )
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Accessibility evaluation failed: {e}")
            
            result = {
                "accessibility_score": 0.0,
                "error": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=str(e),
                output=result,
                metadata={"success": False}
            )
            
            return result

    def _build_evaluation_prompt(self, user_message: str, bot_message: str) -> str:
        """Build evaluation prompt for accessibility assessment"""
        return f"""
Evaluate the language accessibility and clarity of this bot response for someone seeking emotional support.

USER MESSAGE: "{user_message}"
BOT RESPONSE: "{bot_message}"

Consider these aspects:
1. Language Clarity: Is the language clear and easy to understand?
2. Emotional Accessibility: Would this make sense to someone in emotional distress?
3. Complexity: Are there unnecessarily complex or confusing terms?
4. Tone Appropriateness: Is the tone appropriate for the emotional context?
5. Sentence Structure: Are sentences well-structured and easy to follow?
6. Cultural Sensitivity: Does it avoid jargon and culturally-specific references?

Return ONLY valid JSON:
{{
    "accessibility_score": <number between 0.0 and 1.0>,
    "reasoning": "<brief explanation of the score>",
    "clarity_score": <number 0.0-1.0>,
    "complexity_score": <number 0.0-1.0>,
    "tone_appropriateness_score": <number 0.0-1.0>,
    "suggestions": ["<improvement 1>", "<improvement 2>"] or []
}}
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for evaluation"""
        if not self.llm_client:
            return json.dumps({
                "accessibility_score": 0.5,
                "reasoning": "No LLM client available"
            })
        
        try:
            if hasattr(self.llm_client, 'generate_content_async'):
                response = await self.llm_client.generate_content_async(prompt)
                return response.text
            elif hasattr(self.llm_client, 'chat'):
                response = await self.llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            else:
                return json.dumps({
                    "accessibility_score": 0.5,
                    "reasoning": "Unsupported LLM client"
                })
                
        except Exception as e:
            return json.dumps({
                "accessibility_score": 0.5,
                "reasoning": f"LLM call failed: {str(e)}"
            })

    def _parse_result(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            if isinstance(llm_response, str):
                # Find JSON in response
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = llm_response[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    
                    # Validate and clamp scores
                    score_fields = ["accessibility_score", "clarity_score", "complexity_score", "tone_appropriateness_score"]
                    for field in score_fields:
                        if field in parsed and isinstance(parsed[field], (int, float)):
                            parsed[field] = max(0.0, min(1.0, float(parsed[field])))
                    
                    return parsed
            
            if isinstance(llm_response, dict):
                return llm_response
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse accessibility evaluation: {e}")
        
        return {
            "accessibility_score": 0.5,
            "reasoning": "Failed to parse LLM response"
        }

