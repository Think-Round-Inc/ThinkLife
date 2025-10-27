import json
import asyncio
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from datetime import datetime
from brain.brain_core import ThinkxLifeBrain

class LLMEvaluator:
    def __init__(self, brain_instance: ThinkxLifeBrain = None):
        self.brain = brain_instance or ThinkxLifeBrain()
        self.llm = self.brain.providers.get("gemini")

    async def _send_prompt(self, prompt: str):
        """
        Send a prompt to the Gemini LLM safely with timeout handling.
        Returns the response text or JSON error string.
        """
        try:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=self.llm.max_tokens,
                temperature=self.llm.temperature,
                response_mime_type="application/json"
            )
            # Call Gemini API
            try:
                response = await asyncio.wait_for(
                    self.llm.model.generate_content_async(
                        contents=prompt,
                        generation_config=generation_config
                    ),
                    timeout=self.llm.timeout
                )
                return response.text
            except asyncio.TimeoutError:
                return json.dumps({"error": "Gemini request timed out"})
        except Exception as e:
            return json.dumps({"error": f"Gemini generation failed: {str(e)}"})