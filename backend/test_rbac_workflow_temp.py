import asyncio
from types import SimpleNamespace
from brain.cortex.workflow_engine import WorkflowEngine

async def main():
    engine = WorkflowEngine()

    request = SimpleNamespace(
        user_context=SimpleNamespace(
            user_id="test_user_003",
            roles=["thinklife_user"]
        )
    )

    specs = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="restricted_tool",
                enabled=True
            )
        ]
    )

    state = {
        "request": request,
        "specs": specs,
        "execution_steps": [],
        "tool_results": [],
        "errors": []
    }

    result = await engine._execute_tools_node(state)

    print("Tool Results:", result["tool_results"])
    print("Errors:", result["errors"])

    assert result["tool_results"][0]["status"] == "denied"
    assert result["tool_results"][0]["tool"] == "restricted_tool"
    assert result["errors"] == []

    print("PASS: Unauthorized tool was blocked by workflow RBAC")
if __name__ == "__main__":
    asyncio.run(main())


