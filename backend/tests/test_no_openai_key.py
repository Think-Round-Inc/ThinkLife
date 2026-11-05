"""Optional OpenAI-only features switch off quietly when OPENAI_API_KEY is unset."""
import asyncio
from config import settings
from brain.data_sources.vector_db import VectorDataSource
from brain.data_sources.base_data_source import DataSourceConfig
from brain.specs import DataSourceType


def test_vector_search_returns_empty_when_embeddings_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    monkeypatch.setattr(settings, "embedding_provider", "openrouter")  # selected but no key -> disabled
    cfg = DataSourceConfig("v", DataSourceType.VECTOR_DB, {"collection_name": "c", "db_path": str(tmp_path / "x.sqlite3")})
    ds = VectorDataSource(cfg)
    asyncio.run(ds.initialize())
    assert asyncio.run(ds.query("hello")) == []


def test_tts_disabled_without_key(monkeypatch):
    from agents.zoe.tts_service import TTSService
    monkeypatch.setattr(settings, "openai_api_key", None)
    assert asyncio.run(TTSService().generate_speech("hi")) is None


def test_local_embeddings_used_without_key(monkeypatch):
    from brain.data_sources import embeddings as emb
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(emb, "LocalEmbeddings", lambda: "local")
    assert emb.get_embeddings() == "local"


def test_openai_embeddings_used_with_key(monkeypatch):
    from brain.data_sources import embeddings as emb
    monkeypatch.setattr(settings, "openai_api_key", "k")
    assert type(emb.get_embeddings()).__name__ == "OpenAIEmbeddings"


def _emb(monkeypatch, provider, model=None, **keys):
    from brain.data_sources import embeddings as emb
    monkeypatch.setattr(settings, "embedding_provider", provider)
    monkeypatch.setattr(settings, "embedding_model", model)
    for k in ("openai_api_key", "openrouter_api_key", "gemini_api_key"):
        monkeypatch.setattr(settings, k, keys.get(k))
    return emb.get_embeddings()


def test_openrouter_embeddings_use_openrouter_endpoint(monkeypatch):
    e = _emb(monkeypatch, "openrouter", openrouter_api_key="k")
    assert e.model == "openai/text-embedding-3-small" and "openrouter.ai" in str(e.openai_api_base)


def test_gemini_embeddings_use_gemini_endpoint(monkeypatch):
    e = _emb(monkeypatch, "gemini", "gemini-embedding-001", gemini_api_key="k")
    assert "generativelanguage.googleapis.com" in str(e.openai_api_base)


def test_selected_embedding_provider_without_key_is_disabled(monkeypatch):
    assert _emb(monkeypatch, "openrouter") is None
    assert _emb(monkeypatch, "nonsense") is None
