"""
Local prompt version registry.

Prompt templates live as versioned files in this directory (e.g. zoe_system.v1.md,
zoe_system.v2.md, ...). This registry resolves a prompt name to its highest
available version, so bumping a prompt is just adding a new .vN.md file and
committing it -- git history becomes the prompt's changelog.

This is the offline/self-host fallback described in Epic 4 (works without
Langfuse Prompt Management). Every resolved prompt carries its version + a
content hash so traces can record exactly which prompt version produced a
given response.
"""

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent

_VERSION_PATTERN = re.compile(r"^(?P<name>.+)\.v(?P<version>\d+)\.md$")


class PromptVersion(TypedDict):
    name: str
    version: int
    sha256: str
    text: str


def _discover_latest_version(name: str) -> Path:
    candidates = []
    for path in PROMPTS_DIR.glob(f"{name}.v*.md"):
        match = _VERSION_PATTERN.match(path.name)
        if match and match.group("name") == name:
            candidates.append((int(match.group("version")), path))

    if not candidates:
        raise FileNotFoundError(f"No versioned prompt files found for '{name}' in {PROMPTS_DIR}")

    candidates.sort(key=lambda c: c[0])
    return candidates[-1][1]


@lru_cache(maxsize=32)
def get_prompt(name: str) -> PromptVersion:
    """Resolve `name` to its highest-numbered .vN.md file. Cached per process --
    restart the process to pick up a newly bumped version (matches the "commit
    the file so git = version history" model: a version bump is a deploy)."""
    path = _discover_latest_version(name)
    match = _VERSION_PATTERN.match(path.name)
    version = int(match.group("version"))
    text = path.read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return {"name": name, "version": version, "sha256": digest, "text": text}
