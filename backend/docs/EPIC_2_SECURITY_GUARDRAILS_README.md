# Epic 2: Security & Guardrails Enhancement

**User Stories:** 2.1 — Stronger Security Guardrails; 2.2 — Safe and Context-Aware Responses  
**Epic:** Epic 2 — Security & Guardrails Enhancement  
**Project:** ThinkLife — Generative AI Backend  
**Component:** ThinkLife Brain / Zoe AI Companion 
**Author:** Bhargavi
**Branch:** To be confirmed  
**Pull Request:** To be created

## 1. User Stories

### User Story 2.1 — Stronger Security Guardrails

Implement stronger security controls for the ThinkLife AI system, including input validation, protection of sensitive information, role-based access controls for agent tools, and security architecture updates.

### User Story 2.2 — Safe and Context-Aware Responses

Improve the safety and contextual appropriateness of AI-generated responses by introducing output policy validation, fallback responses for policy violations, guardrail testing scenarios, and hallucination observation tracking.
## Summary of changes

### User Story 2.1 — Stronger Security Guardrails

- Implemented Pydantic request validation for API and chat endpoints, including required nonempty messages, whitespace handling, and safe dictionary defaults.
- Added configurable phrase-based content filtering, input sanitization, and moderation checkpoints for incoming messages and generated responses.
- Implemented role-based access controls (RBAC) for agent tools, including authorization checks in the workflow engine, authenticated role handling, and strengthened JWT validation.
- Added sensitive-data redaction using Microsoft Presidio, spaCy, and regex-based detection, integrating redaction into Zoe message processing, conversation history, and Cortex input/output handling.

### User Story 2.2 — Safe and Context-Aware Responses

- Implemented final-response policy validation with configurable safety rules covering medical safety, crisis safety, harmful instructions, professional boundaries, unsupported certainty, and privacy.
- Added category-specific fallback responses when generated content violates blocking safety policies.
- Implemented guardrail testing scenarios covering policy rules, fallback handling, edge cases, regression cases, and hallucination observation tracking.
- Added privacy-conscious hallucination observation tracking with structured JSONL records, response-path monitoring, and a rate calculation for observations classified as flagged or unflagged.
## Files touched

### User Story 2.1 — Stronger Security Guardrails

| **File** | **Change** | **Why** |
| --- | --- | --- |
| `backend/main.py` | Changed | Added Pydantic request validation and protected authentication-related context fields. |
| `backend/agents/zoe/helpers.py` | Changed | Restricted role extraction to authenticated user contexts. |
| `backend/brain/specs/core_specs.py` | Changed | Added role and token fields to the user context structure. |
| `backend/brain/guardrails/session_manager.py` | Changed | Strengthened JWT validation and rejected tokens when signature verification is unavailable. |
| `backend/brain/guardrails/security_manager.py` | Changed | Added content filtering, input sanitization, and sensitive-data redaction. |
| `backend/brain/guardrails/rbac.py` | New | Implemented role-based authorization for agent tools. |
| `backend/brain/guardrails/pii_detector.py` | New | Added Presidio-based PII detection and redaction. |
| `backend/brain/guardrails/__init__.py` | Changed | Exported RBAC components through the guardrails package. |
| `backend/brain/cortex/workflow_engine.py` | Changed | Added RBAC authorization checks before tool handling. |
| `backend/brain/cortex/cortex.py` | Changed | Integrated input and output security checkpoints. |
| `backend/config/blocked_terms.txt` | New | Added configurable phrases for content filtering. |
| `LOCAL_SETUP_GUIDE.md` | Changed | Documented installation of the spaCy English language model. |

### User Story 2.2 — Safe and Context-Aware Responses

| **File** | **Change** | **Why** |
| --- | --- | --- |
| `backend/brain/guardrails/policy_validator.py` | New | Implemented final-response safety policy validation. |
| `backend/brain/guardrails/fallback_handler.py` | New | Added category-specific safe fallback messages. |
| `backend/brain/guardrails/hallucination_tracker.py` | New | Implemented structured hallucination observation tracking and rate calculation. |
| `backend/tests/guardrails_safety_tests/test_safety_policy_rules.py` | New | Added safety policy rule tests. |
| `backend/tests/guardrails_safety_tests/test_safety_fallbacks.py` | New | Added fallback message tests. |
| `backend/tests/guardrails_safety_tests/test_safety_edge_cases.py` | New | Added safety edge-case tests. |
| `backend/tests/guardrails_safety_tests/test_policy_gap_regression.py` | New | Added regression tests for previously identified policy gaps. |
| `backend/tests/guardrails_safety_tests/test_hallucination_tracker.py` | New | Added observation tracking and metric tests. |
| `backend/tests/guardrails_safety_tests/assess_policy_gaps.py` | New | Added a script for assessing safety policy gaps. |

### Shared files — User Stories 2.1 and 2.2

| **File** | **Change** | **Why** |
| --- | --- | --- |
| `backend/agents/zoe/zoe_service.py` | Changed | Integrated redaction, output policy validation, fallback handling, and hallucination observation tracking into Zoe response processing. |
| `backend/requirements.txt` | Changed | Added Presidio, spaCy, the English language model, and pytest dependencies. |
| `backend/docs/EPIC_2_SECURITY_GUARDRAILS_README.md` | New | Documents implementation, setup, testing, and acceptance criteria for both user stories. |
## Pseudocode

### User Story 2.1 — Stronger Security Guardrails

**1. Input validation using Pydantic**

```text
FUNCTION validate_api_request(request):
    VALIDATE request using the appropriate Pydantic model
    STRIP surrounding whitespace from string fields

    IF message is missing or empty after validation:
        RETURN request validation error

    INITIALIZE user_context with an empty dictionary if not provided
    RETURN validated request
```

**2. Content filtering and moderation**

```text
FUNCTION filter_content(content):
    LOAD configured blocked phrases

    CHECK content against blocked phrases
    REPLACE matched blocked phrases with masked characters
    RECORD matching content-filter flags

    CHECK for configured trauma-related indicators
    RECORD trauma-related flags separately

    IF blocked phrases were detected:
        RETURN filtered content with safe = False
    ELSE:
        RETURN content with safe = True
```

**3. Authentication and role-based access control**

```text
FUNCTION validate_session(token):
    RETRIEVE Keycloak public key

    IF public key is unavailable:
        RETURN invalid token result

    VERIFY token signature, issuer, audience, and expiration

    IF token verification fails:
        RETURN invalid token result

    RETURN validated session information
```

```text
FUNCTION authorize_agent_tool(user_context, tool):
    EXTRACT authenticated user roles from trusted context

    CHECK roles against configured tool permissions

    IF no role authorizes the requested tool:
        LOG RBAC_DENY event
        RETURN tool result with status = "denied"

    LOG RBAC_ALLOW event

    IF tool registry is available:
        RETURN permitted tool-handling result

    RETURN the applicable workflow result
```

**4. Sensitive-data detection and redaction**

```text
FUNCTION redact_sensitive_data(text):
    DETECT sensitive patterns using regular expressions
    MASK matching sensitive values

    RUN Presidio-based PII detection
    IDENTIFY supported sensitive entity types
    REPLACE detected entities with redaction placeholders

    IF Presidio detection fails:
        LOG warning
        RETAIN regex-based masking where available

    RETURN redacted text and redaction metadata
```

**5. Input and output security checkpoints**

```text
FUNCTION process_secured_message(message):
    SANITIZE incoming message

    IF sanitized message is empty:
        RETURN input error

    REDACT sensitive information
    APPLY content filtering

    IF input is blocked:
        RETURN safe error response

    CONTINUE message processing

    SANITIZE generated response
    REDACT sensitive information
    APPLY output content filtering

    IF generated response is blocked:
        REPLACE it with a safe response

    RETURN processed response
```

### User Story 2.2 — Safe and Context-Aware Responses

**1. Final-response policy validation**

```text
FUNCTION validate_response(response, context):
    IF response is invalid or empty:
        RETURN blocked validation result

    CHECK response against configured safety policy rules

    FOR each matching rule:
        RECORD rule identifier and policy category

        IF severity is BLOCK:
            RECORD blocking violation

        ELSE IF severity is WARN:
            RECORD warning

    IF blocking violations exist:
        RETURN allowed = False with violation details

    RETURN allowed = True with warnings and matched rules

    IF unexpected validation error occurs:
        RETURN blocked validation result
```

**2. Safe fallback response handling**

```text
FUNCTION get_fallback_message(policy_result):
    IF policy_result is invalid:
        RETURN default fallback message

    READ blocking violations

    FOR each category in configured priority order:
        IF category matches a blocking violation:
            RETURN category-specific fallback message

    RETURN default fallback message
```

**3. Final-response moderation in Zoe**

```text
FUNCTION moderate_final_response(generated_response):
    SANITIZE generated response
    REDACT sensitive information
    APPLY content filtering

    IF content filtering blocks the response:
        RETURN safe replacement response

    VALIDATE response against output safety policies

    IF policy validation blocks the response:
        RETURN category-specific fallback message

    RETURN approved response

    IF moderation raises an unexpected error:
        RETURN safe replacement response
```

**4. Guardrail testing scenarios**

```text
FUNCTION run_guardrail_tests():
    TEST configured safety policy rules
    TEST safe and unsafe response examples
    TEST category-specific fallback messages
    TEST fallback priority and default handling
    TEST policy edge cases and regression scenarios
    TEST hallucination observation recording and metrics

    ASSERT expected results for each test
    REPORT passed and failed tests
```

**5. Hallucination observation tracking**

```text
FUNCTION record_hallucination_observation(reason):
    CREATE observation with:
        status = "unevaluated"
        agent_name = "zoe"
        groundedness_score = None
        reason = provided reason

    ATTEMPT to append observation to JSONL log

    IF persistence fails:
        LOG warning without interrupting message processing

    RETURN
```

```text
FUNCTION calculate_hallucination_rate(agent_name, since_timestamp):
    LOAD stored observations
    FILTER by agent and timestamp when provided

    COUNT flagged observations
    COUNT unflagged observations
    COUNT unevaluated observations

    evaluated = flagged + unflagged

    IF evaluated equals zero:
        rate = None
    ELSE:
        rate = flagged / evaluated

    RETURN observation counts and calculated rate
```
## Setup

### Packages added

The following dependencies were added to `backend/requirements.txt`:

- `presidio-analyzer==2.2.364` — Sensitive-data detection.
- `spacy==3.8.15` — Natural language processing for PII detection.
- `en-core-web-lg==3.8.0` — spaCy English language model, installed from its pinned wheel URL.
- `pytest==9.1.1` — Automated guardrail testing.

### Environment variables

- `LANGFUSE_PUBLIC_KEY` — Optional for the tested guardrail workflows; required to enable Langfuse client functionality.

Existing Keycloak authentication configuration must also be available for authenticated workflows. No new Keycloak environment variable names have been verified specifically for these user stories.

### Config flags

The security manager uses the following defaults:

- `rate_limiting.enabled`: `True`
- `rate_limiting.requests_per_minute`: `60`
- `rate_limiting.requests_per_hour`: `1000`
- `content_filtering.enabled`: `True`
- `content_filtering.blocked_words`: `[]`
- `content_filtering.trauma_safe_mode`: `True`
- `user_validation.require_auth`: `True`
- `user_validation.allow_anonymous`: `False`

Additional configuration:

- `backend/config/blocked_terms.txt` — Configurable blocked phrases used by the content filtering layer.
- `logs/hallucination_observations.jsonl` — Default output path for hallucination observation records.

### Migrations or seed data

- Database migrations: None.
- Database seed data: None.
- The blocked-terms configuration file must be included in the deployment.
- The spaCy English language model must be installed with the backend dependencies.
## How to run

Run the following commands from the `ThinkLife/backend` directory.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

This installs the backend dependencies, including Presidio, spaCy, the English language model, and pytest.

### 2. Start the backend

```bash
python main.py
```

The FastAPI application starts using Uvicorn at `http://localhost:8000`.

### 3. Run automated guardrail tests

```bash
python -m pytest tests/guardrails_safety_tests/ -v
```

**Verification:** All 55 automated guardrail tests passed during local testing.

### 4. Run additional security verification scripts

**RBAC workflow verification:**

```bash
python test_rbac_workflow_temp.py
```

**Sensitive-data redaction verification:**

```bash
python test_task4_message_flow.py
```

Both scripts passed during local verification.

**Note:** Authenticated workflows require valid Keycloak configuration. Langfuse monitoring requires its applicable configuration; without the public key, the Langfuse client may be disabled.
## Acceptance criteria mapping

### User Story 2.1 — Stronger Security Guardrails

| **Task or criterion (from the epic)** | **Implemented in** | **Proof** | **Status** |
| --- | --- | --- | --- |
| 1. Implement input validation layer using Pydantic schemas | `backend/main.py` | Pydantic request models and input validation logic reviewed | Done |
| 2. Add content filtering & moderation checkpoints | `backend/brain/guardrails/security_manager.py`, `backend/brain/cortex/cortex.py`, `backend/agents/zoe/zoe_service.py` | Input and output moderation checkpoints implemented | Done |
| 3. Define role-based access controls (RBAC) for agent tools | `backend/brain/guardrails/rbac.py`, `backend/brain/cortex/workflow_engine.py`, `backend/brain/guardrails/session_manager.py` | `test_rbac_workflow_temp.py` — Unauthorized tool access blocked successfully | Done |
| 4. Add sensitive data redaction layer | `backend/brain/guardrails/pii_detector.py`, `backend/brain/guardrails/security_manager.py`, `backend/agents/zoe/zoe_service.py` | `test_task4_message_flow.py` — All 3 redaction checks passed | Done |
| 5. Document security architecture update in release document | `backend/docs/EPIC_2_SECURITY_GUARDRAILS_README.md` | Security guardrail implementation and architecture updates documented in the combined Epic 2 README | Done |

### User Story 2.2 — Safe and Context-Aware Responses

| **Task or criterion (from the epic)** | **Implemented in** | **Proof** | **Status** |
| --- | --- | --- | --- |
| 1. Introduce policy validation before final output | `backend/brain/guardrails/policy_validator.py`, `backend/agents/zoe/zoe_service.py` | Automated safety policy validation tests passed | Done |
| 2. Add fallback message when policy violation detected | `backend/brain/guardrails/fallback_handler.py`, `backend/agents/zoe/zoe_service.py` | Automated fallback response tests passed | Done |
| 3. Implement guardrail testing scenarios | `backend/tests/guardrails_safety_tests/` | `python -m pytest tests/guardrails_safety_tests/ -v` — 55 tests passed | Done |
| 4. Define hallucination observation tracking metric | `backend/brain/guardrails/hallucination_tracker.py`, `backend/agents/zoe/zoe_service.py` | `test_hallucination_tracker.py` — Automated tests passed | Done |
## Dependencies and conflicts

- **Depends on:** Existing ThinkLife backend infrastructure, including Zoe AI Companion, Cortex workflow engine, Keycloak authentication, and the shared security manager. User Stories 2.1 and 2.2 are implemented together as part of Epic 2.

- **Blocks:** No specific dependent user story has been confirmed. Integration testing should include the updated security and response-processing flows.

- **Shared files also edited by other stories:** `backend/agents/zoe/zoe_service.py`, `backend/brain/cortex/cortex.py`, `backend/brain/cortex/workflow_engine.py`, `backend/main.py`, and `backend/requirements.txt` are shared integration points. Potential conflicts should be reviewed against changes in the staging branch.

- **Suggested merge order:** Merge the combined Epic 2 implementation into `staging` after reviewing shared-file conflicts and completing the required integration checks. The designated integrator will handle the merge.
## Known limitations

- **Integration testing:** The security guardrails and response safety components have passed local tests. Full end-to-end validation with the integrated LLM provider and other user stories remains pending team integration testing.

- **Hallucination observation tracking:** Zoe currently records observations with the status `unevaluated`. The tracker supports calculating a hallucination rate from observations classified as `flagged` or `unflagged`, but automatic hallucination classification and groundedness scoring are not implemented.

- **RBAC tool execution:** Role-based authorization checks are implemented and tested. The workflow engine's permitted-tool execution path currently uses a placeholder and requires integration with the actual tool execution mechanism.

- **Content filtering configuration:** The blocked-terms configuration contains temporary testing phrases that should be removed before production deployment.

- **Follow-up story:** No separate follow-up story ID has been assigned. Remaining integration and production-readiness activities will be coordinated during team integration.