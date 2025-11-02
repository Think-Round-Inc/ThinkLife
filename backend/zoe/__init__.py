"""
Zoe - AI Companion for ThinkLife
Integrates with the Brain module for LLM calls and responses with full conversation management
"""

from .zoe_core import ZoeCore
from .conversation_manager import ZoeConversationManager
from .personality import ZoePersonality

__all__ = ["ZoeCore", "ZoeConversationManager", "ZoePersonality"] 