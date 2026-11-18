
"""
User Story 2.2 - Task 1
Temporary Zoe Metadata Security Test

Checks whether ZoeService forwards raw agent metadata
to the public response or stores raw agent errors.

This is a static source-code check, not a runtime
integration test.
"""

import ast
from pathlib import Path


SERVICE_FILE = (
    Path(__file__).resolve().parent
    / "agents"
    / "zoe"
    / "zoe_service.py"
)


def get_return_dictionaries(tree):
    """Find return statements containing dictionaries."""
    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(
            node.value, ast.Dict
        ):
            results.append(node)

    return results


def get_dictionary_value(dictionary, key):
    """Retrieve a dictionary value by its string key."""
    for dict_key, value in zip(
        dictionary.keys, dictionary.values
    ):
        if (
            isinstance(dict_key, ast.Constant)
            and dict_key.value == key
        ):
            return value

    return None


def contains_raw_agent_metadata(node):
    """Detect **agent_response.metadata unpacking."""
    if not isinstance(node, ast.Dict):
        return False

    for key, value in zip(node.keys, node.values):
        if key is not None:
            continue

        if isinstance(value, ast.Attribute):
            if (
                value.attr == "metadata"
                and isinstance(value.value, ast.Name)
                and value.value.id == "agent_response"
            ):
                return True

    return False


def check_public_metadata():
    """Check public response dictionaries."""
    source = SERVICE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    findings = []

    for statement in get_return_dictionaries(tree):
        response_dict = statement.value

        metadata = get_dictionary_value(
            response_dict, "metadata"
        )

        if contains_raw_agent_metadata(metadata):
            findings.append(statement.lineno)

    if findings:
        print("[FAIL] Raw agent metadata forwarding detected.")

        for line in findings:
            print(f"       Return statement at line {line}")

        return False

    print("[PASS] No direct raw agent metadata unpacking found.")
    return True


def check_history_error():
    """Detect raw agent error stored in history metadata."""
    source = SERVICE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    findings = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if not isinstance(node.func, ast.Attribute):
            continue

        if node.func.attr != "add_message":
            continue

        for keyword in node.keywords:
            if keyword.arg != "metadata":
                continue

            metadata = keyword.value

            if not isinstance(metadata, ast.Dict):
                continue

            error_value = get_dictionary_value(
                metadata, "error"
            )

            if error_value is None:
                continue

            for child in ast.walk(error_value):
                if not isinstance(child, ast.Attribute):
                    continue

                if (
                    child.attr == "metadata"
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "agent_response"
                ):
                    findings.append(node.lineno)
                    break

    if findings:
        print("[FAIL] Raw agent error stored in history.")

        for line in findings:
            print(f"       add_message call at line {line}")

        return False

    print("[PASS] No direct raw agent error storage found.")
    return True


# Pytest-compatible tests.
# Assertions ensure a failed security check fails Pytest.

def test_public_metadata():
    assert SERVICE_FILE.exists(), (
        f"ZoeService file not found: {SERVICE_FILE}"
    )
    assert check_public_metadata(), (
        "Raw agent metadata forwarding detected."
    )


def test_history_error():
    assert SERVICE_FILE.exists(), (
        f"ZoeService file not found: {SERVICE_FILE}"
    )
    assert check_history_error(), (
        "Raw agent error storage detected."
    )


def main():
    print("=" * 60)
    print("USER STORY 2.2 - TASK 1")
    print("ZOE METADATA SECURITY CHECK")
    print("=" * 60)

    if not SERVICE_FILE.exists():
        print("[ERROR] zoe_service.py was not found.")
        print(f"Expected location: {SERVICE_FILE}")
        raise SystemExit(2)

    print(f"Checking: {SERVICE_FILE.name}")
    print()

    results = [
        check_public_metadata(),
        check_history_error(),
    ]

    print()
    print("=" * 60)

    if all(results):
        print("RESULT: STATIC CHECKS PASSED")
        print("Runtime verification is still required.")
    else:
        print("RESULT: SECURITY REVIEW REQUIRED")
        print("Do not modify code until findings are reviewed.")

    print("=" * 60)

    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

