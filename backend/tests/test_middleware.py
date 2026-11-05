"""
Middleware tests (Epic 5 / Story 5.1, Task 5.5).

Run with: pytest tests/test_middleware.py -v
"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

import main
from middleware.logging_middleware import RequestLoggingMiddleware, correlation_id_var
from middleware.validation_middleware import PayloadValidationMiddleware


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


# --- RequestLoggingMiddleware ---------------------------------------------

def test_correlation_id_header_is_generated_when_absent(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID")  # present and non-empty


def test_incoming_correlation_id_is_honored(client):
    response = client.get("/healthz", headers={"X-Correlation-ID": "test-fixed-id"})
    assert response.headers.get("X-Correlation-ID") == "test-fixed-id"


def test_correlation_id_appears_in_logs_across_modules(client, caplog):
    """
    Regression test for Task 5.4: the correlation ID set by
    RequestLoggingMiddleware must be visible on log records emitted by
    completely different modules (not just the middleware itself) during
    the same request.
    """
    with caplog.at_level(logging.INFO):
        response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200

    correlation_id = response.headers["X-Correlation-ID"]
    # At least one record from a module other than the middleware itself
    # should carry the same correlation_id attribute.
    other_module_records = [
        r for r in caplog.records
        if r.name != "middleware.logging_middleware" and hasattr(r, "correlation_id")
    ]
    assert other_module_records, "expected other modules' log records to carry correlation_id"
    assert any(r.correlation_id == correlation_id for r in other_module_records)


def test_logging_middleware_does_not_swallow_exceptions():
    """
    A route that raises must still propagate past RequestLoggingMiddleware -
    the middleware logs the failure but re-raises rather than converting it
    into a response itself.
    """
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/boom")
    async def boom():
        raise ValueError("kaboom")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")
    # Starlette's own ExceptionMiddleware turns the unhandled error into a
    # 500 - the important thing is it's a 500 (server error, exception
    # surfaced), not a silently-swallowed 200.
    assert response.status_code == 500


def test_correlation_id_contextvar_reset_after_request(client):
    assert correlation_id_var.get() == "-"
    client.get("/healthz")
    # Context var must not leak between requests/outside request handling.
    assert correlation_id_var.get() == "-"


# --- PayloadValidationMiddleware ------------------------------------------

def test_malformed_json_body_returns_400_not_500(client):
    response = client.post(
        "/api/chat", data=b"{not valid json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_oversized_body_returns_413_not_500(client):
    huge_payload = b'{"message": "' + b"a" * 2_000_000 + b'"}'
    response = client.post(
        "/api/chat", data=huge_payload, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_valid_json_body_passes_through(client):
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200


def test_validation_middleware_only_inspects_body_methods():
    """GET requests are never body-inspected, even with a JSON content-type header."""
    app = FastAPI()
    app.add_middleware(PayloadValidationMiddleware)

    @app.get("/ok")
    async def ok():
        return PlainTextResponse("fine")

    client = TestClient(app)
    response = client.get("/ok", headers={"Content-Type": "application/json"})
    assert response.status_code == 200


# --- Field-level validation (ChatMessageRequest) --------------------------

def test_blank_message_returns_422_not_500(client):
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422


def test_missing_message_field_returns_422_not_500(client):
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_oversized_message_field_returns_422(client):
    response = client.post("/api/chat", json={"message": "a" * 10001})
    assert response.status_code == 422
