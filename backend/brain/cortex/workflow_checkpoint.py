"""
Workflow checkpointing (US 1.1)

LangGraph saves the workflow state after every step to a checkpointer, keyed
by thread_id (= the execution id). If a run is interrupted (process restart,
crash in a node) it can be resumed from its last finished step instead of
starting over.

Choose the backend with WORKFLOW_CHECKPOINTER:
    sqlite  (default)  backend/data/workflow_checkpoints.sqlite3, survives restarts
    memory             in-process only; used by tests
    none               checkpointing off

WORKFLOW_CHECKPOINT_PATH overrides the SQLite file location.

Privacy: Zoe's prompt carries user context (e.g. ACE score) and `messages` is
part of the state, so checkpoints of finished runs are deleted by default.
Set WORKFLOW_KEEP_COMPLETED_CHECKPOINTS=true to keep them for debugging.
The file is git-ignored (*.sqlite3).
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "workflow_checkpoints.sqlite3"


def keep_completed_checkpoints() -> bool:
    return os.getenv("WORKFLOW_KEEP_COMPLETED_CHECKPOINTS", "").strip().lower() in {"1", "true", "yes", "on"}


async def create_checkpointer(kind: Optional[str] = None, path: Optional[str] = None) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Returns (checkpointer, closeable). `closeable` is the open DB connection
    for SQLite (close it on shutdown) and None otherwise.
    Falls back to the in-memory saver if the SQLite saver is not installed.
    """
    kind = (kind or os.getenv("WORKFLOW_CHECKPOINTER") or "sqlite").strip().lower()

    if kind in {"none", "off", "false", "0"}:
        logger.info("Workflow checkpointing disabled")
        return None, None

    if kind == "sqlite":
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            db_path = Path(path or os.getenv("WORKFLOW_CHECKPOINT_PATH") or _DEFAULT_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(db_path))
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            logger.info("Workflow checkpoints: SQLite at %s", db_path)
            return saver, conn
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite / aiosqlite not installed; "
                "using in-memory workflow checkpoints (lost on restart)"
            )
        except Exception:
            logger.exception("Could not open SQLite checkpoint store; using in-memory checkpoints")

    if kind not in {"sqlite", "memory"}:
        logger.warning("Unknown WORKFLOW_CHECKPOINTER=%r; using in-memory checkpoints", kind)

    from langgraph.checkpoint.memory import InMemorySaver
    return InMemorySaver(), None
