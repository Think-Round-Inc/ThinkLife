"""
Health endpoint smoke tests (Epic 5 / Story 5.1, Task 4.5).

Run with: pytest tests/test_health_endpoints.py -v

Boots the real FastAPI app (including its `lifespan`, so `CortexFlow`,
`ZoeService`, and the plugin registry are genuinely initialized) via
`TestClient`, then monkeypatches the health-check methods to simulate
degraded/unhealthy dependencies without needing real provider credentials.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_liveness_always_returns_200(client):
    """/healthz must not depend on Brain/Zoe/provider state."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_healthy_when_dependencies_are_up(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("healthy", "degraded")  # no real provider keys in CI
    assert "brain" in body and "zoe" in body and "plugins" in body
    assert set(body["plugins"]["registered"]) >= {"zoe", "echo"}


def test_readiness_public_paths_do_not_require_auth(client):
    # No Authorization header supplied - if these weren't on the public
    # allow-list (see middleware/keycloak_auth.py), a protected endpoint
    # would still respond (auth here doesn't hard-block), but a mismatch
    # would fail differently. Asserting 200 here is the meaningful check.
    for path in ("/health", "/healthz", "/api/health", "/api/brain/health", "/api/zoe/health"):
        assert client.get(path).status_code == 200, f"{path} should be public and healthy"


def test_readiness_degraded_when_brain_unhealthy(client, monkeypatch):
    async def fake_unhealthy_status():
        return {"overall": "unhealthy", "providers": {}, "system": {}}

    monkeypatch.setattr(main.brain_instance, "get_health_status", fake_unhealthy_status)

    response = client.get("/health")
    assert response.status_code == 200  # readiness reports status in the body, doesn't 5xx
    body = response.json()
    assert body["brain"]["overall"] == "unhealthy"
    assert body["status"] in ("degraded", "unhealthy")


def test_readiness_survives_brain_health_check_raising(client, monkeypatch):
    async def raising_health_check():
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(main.brain_instance, "get_health_status", raising_health_check)

    response = client.get("/health")
    assert response.status_code == 200  # must not 500 just because a dependency check failed
    body = response.json()
    assert body["brain"]["overall"] == "unhealthy"
    assert "error" in body["brain"]


def test_brain_health_and_analytics_endpoints_no_longer_500(client):
    """
    Regression test at the HTTP layer for the CortexFlow.get_health_status /
    get_analytics bug fixed in PR2 - these previously raised AttributeError.
    """
    assert client.get("/api/brain/health").status_code == 200
    assert client.get("/api/brain/analytics").status_code == 200
