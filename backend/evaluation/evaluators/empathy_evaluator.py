"""
Empathy Evaluator

Assesses the empathy and emotional validation in bot responses using LLM-as-judge
with LangFuse observability for tracking and monitoring.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime
from langfuse.decorators import observe, langfuse_context

logger = logging.getLogger(__name__)


class EmpathyEvaluator:
    """Evaluates empathy in bot responses using LLM-as-judge"""

    def __init__(self, llm_client=None):
        """
        Initialize the empathy evaluator
        
        Args:
            llm_client: LLM client for running evaluations (Gemini or OpenAI)
        """
        self.llm_client = llm_client
        self.dimensions = [
            "validation_of_feelings",
            "perspective_taking",
            "tone_sensitivity",
            "politeness_soft_language",
            "encouragement_hopefulness"
        ]

    @observe(name="evaluate_empathy")
    async def evaluate(
        self,
        user_message: str,
        bot_message: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate empathy in bot response
        
        Args:
            user_message: User's message
            bot_message: Bot's response
            context: Additional context for evaluation
            
        Returns:
            Dict with empathy scores and reasoning
        """
        # Track input in LangFuse
        langfuse_context.update_current_trace(
            name="empathy_evaluation",
            user_id=context.get("user_id") if context else None,
            session_id=context.get("session_id") if context else None,
            metadata={
                "evaluator": "empathy",
                "message_length": len(bot_message),
            }
        )
        
        try:
            # Create evaluation prompt
            prompt = self._build_evaluation_prompt(user_message, bot_message)
            
            # Score using LLM-as-judge
            with langfuse_context.observe_llm_call(
                name="empathy_llm_judge",
                model="gemini-1.5-flash",
                input=prompt
            ) as llm_span:
                result = await self._call_llm(prompt)
                llm_span.update(output=result)
            
            # Parse and validate results
            parsed_result = self._parse_result(result)
            
            # Calculate aggregate score
            if "scores" in parsed_result and parsed_result["scores"]:
                avg_score = sum(parsed_result["scores"].values()) / len(parsed_result["scores"])
                parsed_result["average_empathy_score"] = avg_score
            else:
                parsed_result["average_empathy_score"] = 0.0
            
            # Add metadata
            parsed_result.update({
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluator": "empathy",
                "dimensions_evaluated": len(self.dimensions)
            })
            
            # Track output in LangFuse
            langfuse_context.update_current_observation(
                output=parsed_result,
                metadata={"success": True}
            )
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Empathy evaluation failed: {e}")
            
            # Track error in LangFuse
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=str(e),
                metadata={"success": False, "error": str(e)}
            )
            
            return {
                "error": str(e),
                "scores": {dim: 0.0 for dim in self.dimensions},
                "average_empathy_score": 0.0,
                "evaluation_timestamp": datetime.now().isoformat()
            }

    def _build_evaluation_prompt(self, user_message: str, bot_message: str) -> str:
        """Build the LLM evaluation prompt"""
        return f"""
You are a trauma-informed evaluator. Assess the empathy of the bot response.

Follow these steps:
1. Read the user message and chatbot response carefully
2. Identify phrases where the bot validates, acknowledges, or mirrors the user's feelings
3. Assign a score from 0.0 to 1.0 for each of these 5 empathy dimensions:
   - validation_of_feelings: Does the bot validate and accept the user's emotions?
   - perspective_taking: Does the bot show understanding of the user's viewpoint?
   - tone_sensitivity: Is the tone gentle, supportive, and appropriate?
   - politeness_soft_language: Does the bot use respectful and soft language?
   - encouragement_hopefulness: Does the bot provide appropriate encouragement?

4. Provide clear reasoning for each score

Return ONLY valid JSON in this exact format:
{{
    "scores": {{
        "validation_of_feelings": <score 0.0-1.0>,
        "perspective_taking": <score 0.0-1.0>,
        "tone_sensitivity": <score 0.0-1.0>,
        "politeness_soft_language": <score 0.0-1.0>,
        "encouragement_hopefulness": <score 0.0-1.0>
    }},
    "reasoning": {{
        "validation_of_feelings": "<brief explanation>",
        "perspective_taking": "<brief explanation>",
        "tone_sensitivity": "<brief explanation>",
        "politeness_soft_language": "<brief explanation>",
        "encouragement_hopefulness": "<brief explanation>"
    }}
}}

User Message: "{user_message}"
Bot Response: "{bot_message}"
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for evaluation"""
        if not self.llm_client:
            return json.dumps({"error": "No LLM client available"})
        
        try:
            # Use the LLM client (Gemini or OpenAI)
            # This should be adapted to work with the actual provider
            if hasattr(self.llm_client, 'generate_content_async'):
                # Gemini API
                response = await self.llm_client.generate_content_async(prompt)
                return response.text
            elif hasattr(self.llm_client, 'chat'):
                # OpenAI API
                response = await self.llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            else:
                return json.dumps({"error": "Unsupported LLM client"})
                
        except Exception as e:
            return json.dumps({"error": f"LLM call failed: {str(e)}"})

    def _parse_result(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Try to parse as JSON
            if isinstance(llm_response, str):
                # Find JSON in response
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = llm_response[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    
                    # Validate scores are in range
                    if "scores" in parsed:
                        for key, value in parsed["scores"].items():
                            parsed["scores"][key] = max(0.0, min(1.0, float(value)))
                    
                    return parsed
            
            # If already a dict
            if isinstance(llm_response, dict):
                return llm_response
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse empathy evaluation: {e}")
        
        # Return fallback
        return {
            "scores": {dim: 0.0 for dim in self.dimensions},
            "reasoning": {dim: "Failed to parse LLM response" for dim in self.dimensions}
        }

