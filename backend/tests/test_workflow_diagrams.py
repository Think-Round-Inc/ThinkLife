"""
US 1.1 Task 5 - workflow diagrams for release traceability. The committed
diagram and profile table must match what the code generates, so a graph
change cannot ship with stale docs.

Run with: pytest tests/test_workflow_diagrams.py -v
Fix a failure with: python scripts/export_workflow_diagrams.py
"""

import importlib.util
from pathlib import Path

import pytest

from brain.cortex import workflow_engine as we
from brain.cortex.workflow_profiles import BUILTIN_PROFILES

BACKEND_DIR = Path(__file__).resolve().parents[1]


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_workflow_diagrams",
                                                  BACKEND_DIR / "scripts" / "export_workflow_diagrams.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rendered():
    return load_exporter().render_all()


def test_committed_workflow_docs_match_the_code(rendered):
    stale = [path.name for path, content in rendered.items()
             if not path.exists() or path.read_text(encoding="utf-8") != content]
    assert not stale, f"Out of date: {stale}. Run: python scripts/export_workflow_diagrams.py"


def test_diagram_shows_every_node_and_failure_route(rendered):
    diagram = rendered[BACKEND_DIR / "docs" / "workflows" / "brain_workflow.md"]
    for name, value in vars(we).items():
        if name.startswith("NODE_"):
            assert f"{value}[{value}]" in diagram, f"{value} missing from diagram"
    for node, targets in we.ERROR_ROUTES.items():
        for target in targets:
            assert f"{node}_error -.-> {target}" in diagram
    # Routes out of fallback_provider (reached only through the error handler)
    assert "fallback_provider -. recovered .-> check_safety" in diagram
    assert "fallback_provider -. escalate .-> escalate" in diagram
    assert f"v{we.WORKFLOW_VERSION}" in diagram


def test_profiles_table_lists_builtin_profiles(rendered):
    table = rendered[BACKEND_DIR / "docs" / "workflows" / "profiles.md"]
    for name in BUILTIN_PROFILES:
        assert f"| `{name}` |" in table
