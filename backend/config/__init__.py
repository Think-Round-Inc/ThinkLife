"""
Centralized configuration package for the ThinkLife backend.

Import `settings` (a singleton `Settings` instance) instead of calling
`os.getenv()` directly anywhere in the codebase.

    from config import settings
    settings.openai_api_key
    settings.confidence_threshold
"""

from .settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
