"""
Embedding model factory, chosen by EMBEDDING_PROVIDER / EMBEDDING_MODEL
(config/settings.py). Used by vector search and the knowledge index builder so
both always embed with the same model.
"""

import logging
from typing import Any, List, Optional

from config import settings

logger = logging.getLogger(__name__)

# provider -> (OpenAI-compatible base_url or None, default model)
_REMOTE = {
    "openai": (None, "text-embedding-3-small"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai/text-embedding-3-small"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-embedding-001"),
}


class LocalEmbeddings:
    """Chroma's bundled ONNX MiniLM-L6-v2 behind the LangChain Embeddings interface.
    No API key; downloads ~80 MB once to ~/.cache/chroma on first use."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def resolve_embedding_provider() -> str:
    """'auto' -> openai if its key is set, otherwise local."""
    p = settings.embedding_provider.lower()
    if p == "auto":
        return "openai" if settings.openai_api_key else "local"
    return p


def get_embeddings(model_override: Optional[str] = None) -> Optional[Any]:
    """Return an embeddings object, or None if the chosen provider can't be used
    (unknown name, missing key, missing library). Never raises."""
    provider = resolve_embedding_provider()
    try:
        if provider == "local":
            logger.info("Embeddings: local MiniLM-L6-v2 (no API key)")
            return LocalEmbeddings()
        if provider not in _REMOTE:
            logger.warning(f"Unknown EMBEDDING_PROVIDER '{provider}'. Use auto|local|{'|'.join(_REMOTE)}")
            return None
        key = settings.provider_api_key(provider)
        if not key:
            logger.warning(f"Embeddings disabled: EMBEDDING_PROVIDER={provider} needs {provider.upper()}_API_KEY")
            return None
        from langchain_openai import OpenAIEmbeddings
        base_url, default_model = _REMOTE[provider]
        model = model_override or settings.embedding_model or default_model
        logger.info(f"Embeddings: {provider} / {model}")
        kwargs: dict = {"model": model, "api_key": key}
        if base_url:
            # non-OpenAI endpoints expect raw strings, not OpenAI token ids
            kwargs.update(base_url=base_url, check_embedding_ctx_length=False)
        return OpenAIEmbeddings(**kwargs)
    except Exception:
        logger.exception("Failed to load embeddings")
        return None
