"""
Document Summarizer Tool - Summarizes text documents
"""

import logging
from typing import Dict, Any, Optional

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class DocumentSummarizerTool(BaseTool):
    """
    Document Summarizer Tool - Summarizes text documents
    
    Best for:
    - Condensing long documents
    - Extracting key points
    - Creating executive summaries
    - Quick document reviews
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_length = config.get("max_length", 500) if config else 500
        self.provider = None
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default summarizer configuration"""
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_length": 500,
            "summary_types": ["brief", "detailed", "bullet_points"]
        }
    
    async def initialize(self) -> bool:
        """Initialize summarizer"""
        try:
            # Import provider for summarization
            from ..providers import create_provider
            
            # Use OpenAI by default for summarization
            provider_type = self.config.get("provider", "openai")
            self.provider = create_provider(provider_type, {
                "model": self.config.get("model", "gpt-4o-mini"),
                "temperature": 0.3,
                "max_tokens": self.max_length
            })
            
            await self.provider.initialize()
            self._initialized = True
            logger.info("Document Summarizer Tool initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Document Summarizer: {str(e)}")
            return False
    
    async def execute(
        self,
        text: str,
        summary_type: str = "brief",
        focus: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Summarize text document
        
        Args:
            text: Text to summarize
            summary_type: Type of summary - "brief", "detailed", or "bullet_points"
            focus: Optional focus area for summary
            
        Returns:
            ToolResult with summary
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.provider:
            return ToolResult(
                tool_name="DocumentSummarizer",
                success=False,
                content=None,
                metadata={},
                error="Summarizer provider not initialized"
            )
        
        try:
            logger.info(f"Summarizing document ({len(text)} chars, type: {summary_type})")
            
            # Build prompt based on summary type
            prompts = {
                "brief": "Provide a brief summary (2-3 sentences) of the following text:",
                "detailed": "Provide a comprehensive summary of the following text, covering all key points:",
                "bullet_points": "Summarize the following text as a list of key bullet points:"
            }
            
            prompt = prompts.get(summary_type, prompts["brief"])
            
            if focus:
                prompt += f"\nFocus on: {focus}"
            
            prompt += f"\n\nText:\n{text}"
            
            # Generate summary
            messages = [
                {"role": "system", "content": "You are a professional document summarizer."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.provider.generate_response(messages=messages)
            
            if response.get("success") and response.get("content"):
                summary = response.get("content", "")
                
                return ToolResult(
                    tool_name="DocumentSummarizer",
                    success=True,
                    content=summary,
                    metadata={
                        "original_length": len(text),
                        "summary_length": len(summary),
                        "summary_type": summary_type,
                        "compression_ratio": round(len(summary) / len(text), 2)
                    }
                )
            else:
                return ToolResult(
                    tool_name="DocumentSummarizer",
                    success=False,
                    content=None,
                    metadata={},
                    error="Failed to generate summary"
                )
                
        except Exception as e:
            logger.error(f"Document summarization error: {str(e)}")
            return ToolResult(
                tool_name="DocumentSummarizer",
                success=False,
                content=None,
                metadata={},
                error=str(e)
            )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get tool parameters schema"""
        return {
            "text": {
                "type": "string",
                "description": "Text to summarize",
                "required": True
            },
            "summary_type": {
                "type": "string",
                "description": "Type of summary",
                "default": "brief",
                "enum": ["brief", "detailed", "bullet_points"]
            },
            "focus": {
                "type": "string",
                "description": "Optional focus area for summary",
                "required": False
            }
        }
    
    async def close(self) -> None:
        """Close provider and cleanup"""
        if self.provider:
            await self.provider.close()
        self.provider = None
        self._initialized = False
        logger.debug("Document Summarizer Tool closed")


def create_document_summarizer_tool(config: Dict[str, Any] = None) -> DocumentSummarizerTool:
    """Factory function to create Document Summarizer Tool"""
    return DocumentSummarizerTool(config)

