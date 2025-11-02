"""
ThinkLife Brain - Backend AI Orchestration System

This module contains the centralized AI Brain that manages all AI operations
across the ThinkLife platform from the backend.
"""

from .brain_core import ThinkLifeBrain
from .types import BrainRequest, BrainResponse, BrainConfig
# Providers are imported dynamically in brain_core.py

__version__ = "1.0.0"
__all__ = [
    "ThinkLifeBrain",
    "BrainRequest", 
    "BrainResponse",
    "BrainConfig"
] 