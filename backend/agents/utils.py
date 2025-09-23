import os
from dotenv import load_dotenv


load_dotenv()


def enable_tracing():
    """Start tracing for agents
    monitor all the details of the agent including LLM metadata, LLM calls, tool calls and args, intermediate steps
    have feature to visualize the agent through the UI => ( good for debugging )
    """

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")


def LLMLogs(session_id: str, limit: int = 200):
    """Fetch token count + cost usage for all LLM calls in the entire agent session.
    Use it once per session after the session ends.
    beautiful"""
    from langsmith import Client

    client = Client()

    filter_string = f'has(tags, "session:{session_id}")'
    runs_iter = client.list_runs(
        project_name=os.getenv("LANGSMITH_PROJECT"),
        limit=limit,
        filter=filter_string,
    )

    logs = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost": 0.0,
        "llm_calls": 0,
    }

    for run in runs_iter:
        logs["total_tokens"] += getattr(run, "total_tokens", 0) or 0
        logs["prompt_tokens"] += getattr(run, "prompt_tokens", 0) or 0
        logs["completion_tokens"] += getattr(run, "completion_tokens", 0) or 0
        logs["total_cost"] += float(getattr(run, "total_cost", 0) or 0)
        logs["llm_calls"] += 1

    return logs
