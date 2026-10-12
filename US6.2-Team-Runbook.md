# User Story 6.2: Release Governance Runbook


## 1. Requirements and Deliverables

| User Story 6.2 task | Implementation | Evidence |
| --- | --- | --- |
| Document model versions and prompt versions | `scripts/generate_release_manifest.py`, `scripts/release_common.py` | `.release/manifest.json`, `.release/release.md` |
| Update the dependency list in the release document | The generator reads three Python requirements files, frontend package and lock files, and Docker Compose | Dependency tables, actual backend package inventory, and deployed container digests |
| Validate environment variable changes | `scripts/validate_release_env.py`, `docs/release/environment-contract.json` | `.release/env-validation.json`; added, removed, and changed names when prior configuration is supplied |
| Prepare a deployment checklist | `docs/release/DEPLOYMENT_CHECKLIST.md` | Completed release record for the target environment |
| Prepare a rollback plan | `docs/release/ROLLBACK_PLAN.md` | Previous successful release, retained artifacts, configuration and backup references, recovery steps, and rehearsal evidence |
| Enable post-release smoke tests | `scripts/post_release_smoke.py`, `.github/workflows/post-release-smoke.yml` | `.release/smoke-results.json`, exit code, and Actions results when executed |


## 2. Run the Offline Governance Tests

```bat
python -m unittest discover -s tests/release -v
```

Expected ending; execution time varies:

```text
Ran 17 tests in ...s

OK
```

The tests cover missing and placeholder settings, numeric validation, secret-safe reports, environment comparisons, prompt hashes, dependency records, simulated HTTP errors, retries, incorrect release identities, and failure exit behavior. They use a local mock HTTP server and do not require the real application, Keycloak, or an OpenAI key.

| Output | Meaning and action |
| --- | --- |
| `OK` | Governance tests passed; this does not certify application health |
| `FAILED (failures=...)` | An assertion failed; inspect the corresponding traceback |
| `FAILED (errors=...)` | A test raised an exception; inspect the traceback |
| `Start directory is not importable: 'tests/release'` | Usually the wrong working directory, or missing test files |
| `ModuleNotFoundError` | Check module files and directory placement |
| Python command not found | Use an available interpreter; `backend\.venv\Scripts\python.exe` may be used if that environment exists |


## 3. Check Environment Contract Coverage

```bat
python scripts/validate_release_env.py --check-contract
```

Expected:

```text
PASS environment contract-only
```

This checks whether environment names identified by static scanning are documented in the contract. It does not validate deployment credentials. Static scanning is not a complete proof of every possible dynamic lookup; the current provider `key_map` credentials are explicitly included. Review and update coverage when adding new dynamic configuration sources.

If a name is reported as `consumed but missing from contract`, review its purpose, type, and required status, update `docs/release/environment-contract.json`, and rerun the check.

## 4. Validate Current Configuration and Historical Changes

### 4.1 Validate Current Files

```bat
python scripts/validate_release_env.py
```

By default, this reads `backend/.env` and `frontend/.env.local`. Keep the full files in protected locations. Reports do not include secret values or hashes of secrets.

A passing result can include these warnings:

```text
WARN backend.CORS_ORIGINS: not in release contract; verify whether the application uses it
WARN backend.DEBUG: not in release contract; verify whether the application uses it
WARN backend.PORT: not in release contract; verify whether the application uses it
WARN backend: no previous environment supplied; change comparison not performed
WARN frontend.NEXT_PUBLIC_APP_URL: not in release contract; verify whether the application uses it
WARN frontend: no previous environment supplied; change comparison not performed
PASS environment deployment
```

The four legacy settings are not read by the corresponding application code in the inspected baseline. Do not assume changing them changes runtime behavior. If prior configuration is unavailable, explicitly record that no historical comparison was performed. Do not modify application logic simply to remove these warnings.

`PASS environment deployment` means the configured validation rules passed. It does **not** authenticate an API key, check account quota, prove URL reachability, or perform a deployment.

Output: `.release/env-validation.json`. Each run overwrites that default report. Use `--report` with a different filename when retaining separate attempts.

### 4.2 Compare Only When Real Prior Configuration Exists

These are example paths, not files created by the tool:

```bat
python scripts/validate_release_env.py --previous-backend-env "C:\release-secrets\previous-backend.env" --previous-frontend-env "C:\release-secrets\previous-frontend.env" --strict --report .release\env-comparison.json
```

The report lists only variable names under `added`, `removed`, and `changed`. With `--strict`, warnings also fail the gate. Missing historical files produce `Environment file is missing`. Do not copy the current files and label them as a previous successful release.

If services use process environment overrides, add `--process-env` and run with the same configuration as those services. This does not automatically read a remote hosting platform's settings or every Next.js layered environment file. The operator must supply values that match the actual deployment.

### 4.3 Common Validation Failures

| Output fragment | Meaning and action |
| --- | --- |
| `required or explicitly configured value is empty` | Supply the required value or correct an invalid explicit empty setting |
| `placeholder value` | Replace example configuration with the intended real value |
| `invalid int` / `invalid float` / `out of range` | Correct the numeric format or allowed range |
| `requires an HTTP(S) URL` | Correct the URL; embedded credentials, query strings, and fragments are not accepted |
| `configure both or neither` | A paired setting, such as Langfuse or Google OAuth credentials, is incomplete |
| `Keycloak ... mismatch` | Check frontend/backend URL, realm, and client ID |
| `secret must not be public` | Do not expose a secret through a `NEXT_PUBLIC_` variable |
| `Duplicate variable` | Remove duplicate assignments in the same file |
| `Resolve dotenv expansion` | The tool does not resolve `${...}`; supply resolved configuration |
| `Environment file is missing` | Determine whether the missing file is a current file or an explicitly supplied prior file |

## 5. Generate Candidate Model, Prompt, and Dependency Evidence

```bat
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini
```

Typical candidate output:

```text
Generated manifest.json and release.md; dirty=True
```

`dirty=True` is acceptable during candidate review. It indicates uncommitted or untracked, non-ignored files. Review `.release/release.md` and `.release/manifest.json` for:

- Release ID and full source commit.
- Separate records for the Brain configuration, Zoe, FAQ, judge, TTS, and embedding models. The operator must supply the actual Brain configuration using `--openai-model`; the generator does not read `.env` to choose it.
- Zoe's independently hardcoded `gpt-4o-mini` model in the baseline. `OPENAI_MODEL` does not control every model in the project.
- Prompt versions based on entire source-file SHA-256 hashes. Non-prompt edits and line-ending changes can also change these hashes. Dynamic conversations and retrieval content are outside this source inventory.
- The distinction between model aliases and immutable provider snapshots. Record a provider-returned model identifier from an approved functional test when available.
- All three Python requirements files, frontend direct and transitive lockfile data, and container tags.

`BASELINE_RELEASE.md` is an initial preview, not a replacement for the current release evidence.

Capture installed packages from the **actual backend environment**:

```bat
backend\.venv\Scripts\python.exe -m pip list --format=json > .release\backend-installed.json
```

A successful run normally has no terminal output. Inspect the resulting JSON. If the path does not exist, locate the actual backend virtual environment; do not substitute an unrelated Anaconda environment. If the standalone FAQ service is enabled, capture and validate its environment separately.

`pip list` is an installed-package inventory, not a hash-locked installer. Python requirements are not fully pinned, and container tags can change. Retain tested artifacts, a wheelhouse or runtime environment, and actual image digests to support reliable recovery.


## 6. Generate Final Metadata Before Building

From the final, clean release commit:

```bat
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini --require-clean --publish-frontend
```

Expected:

```text
Generated manifest.json and release.md; dirty=False
```

This creates `frontend/public/release.json`. Generate it **before** the frontend build and include it in the deployed artifact. It and `.release/` are gitignored and do not need another commit. Do not manually change a dirty flag to false.

If the command returns `FAIL: Release requires a committed, clean working tree`, inspect the remaining changes with Git. If source or a lockfile changes after generation, commit the new candidate, regenerate metadata, rebuild, and retest.


## 7. Run Live Smoke Tests

### 10.1 Local Execution

In Terminal D, from the repository root:

```bat
cd /d "C:\path\to\ThinkLife"
for /f %i in ('git rev-parse HEAD') do set RELEASE_SHA=%i
echo %RELEASE_SHA%
python scripts/post_release_smoke.py --release-id 0.7.0-rc1 --expected-commit %RELEASE_SHA% --frontend-url http://localhost:3000 --backend-url http://localhost:8000 --keycloak-url http://localhost:8080 --realm thinklife
echo %ERRORLEVEL%
```

Use `%i` in interactive CMD and `%%i` inside a `.bat` file. Check the exit code immediately after the smoke command. Confirm that HEAD is the commit actually built. When checking a remote release, supply that deployment's SHA explicitly instead of assuming the local HEAD is correct.

Defaults are three attempts per check, a 10-second request timeout, and a two-second retry delay. `--attempts`, `--timeout`, and `--delay` can adjust these bounded settings. Repeated retries do not repair a deterministic application defect.

### 7.2 The Seven Checks

| Check | Endpoint | Pass criteria |
| --- | --- | --- |
| frontend_health | Frontend `/api/health` | `status=healthy`, `service=ThinkLife Frontend` |
| frontend_release | Frontend `/release.json` | Release ID and full commit match; dirty flag is false |
| backend_identity | Backend `/` | `message=ThinkLife Backend`, baseline root version `0.7.0` |
| backend_health | Backend `/health` | `status=healthy`, both `brain` and `zoe` are true |
| brain_health | Backend `/api/brain/health` | `status=healthy`, `brain_status.overall=healthy` |
| zoe_health | Backend `/api/zoe/health` | Service is Zoe, initialization/component flags are true, and agent status is healthy |
| keycloak_oidc | `/realms/thinklife/.well-known/openid-configuration` | Correct issuer and matching authorization, token, and JWKS endpoint prefixes |

The root endpoint's `0.7.0` is not a Git commit and is separate from the FastAPI metadata version `2.0.0`. If the application intentionally changes the root version, review and update the smoke expectation; do not confuse this value with the release ID.

Illustrative all-pass output:

```text
PASS frontend_health: OK
PASS frontend_release: OK
PASS backend_identity: OK
PASS backend_health: OK
PASS brain_health: OK
PASS zoe_health: OK
PASS keycloak_oidc: OK
```

Exit codes: **0** means every check passed; **1** means at least one failed; **2** commonly indicates invalid CLI arguments and no completed test run.

### 7.3 Report and Failure Interpretation

`.release/smoke-results.json` contains the release ID, expected commit, UTC timestamp, target URLs, per-check results and attempt counts, and overall pass status. Response bodies are omitted. The default report is overwritten on reruns; use separate `--report` paths to retain failed and recovery attempts.

| Failure | Investigation |
| --- | --- |
| `HTTP 404` | Wrong URL, missing endpoint, realm, or release metadata |
| `HTTP 500` | Server-side exception; inspect that service's logs rather than assuming a cause from the status alone |
| `HTTP 401` / `403` | Authentication, authorization, or proxy restrictions |
| `HTTP 301/302/307/308` | Redirects are not accepted; check the intended final service URL |
| `Connection or TLS failure` | Service stopped, wrong port, timeout, network, or certificate issue |
| `Response contract mismatch` | Invalid JSON, incorrect fields, unhealthy state, old SHA, dirty metadata, or another semantic mismatch |
| `usage: ... error: ...` | Invalid arguments, such as a value other than a full 40-character lowercase SHA |

This table does not enumerate every operating-system or network exception. Preserve unlisted errors and investigate them; do not treat them as a pass.


## 8. Manual Functional Verification and Sign-off

These smoke tests do not perform real LLM generation or fully validate sign-in, authorization, response quality, or FAQ retrieval. After automated checks, also verify:

- [ ] Sign-in, sign-out, and protected pages behave as intended.
- [ ] A representative chat succeeds using approved test data.
- [ ] If the standalone FAQ service is enabled, its API, retrieval, and answers receive separate validation; the main smoke suite does not cover it.
- [ ] The deployed backend artifact digest is verified independently of frontend metadata.
- [ ] Error rates, latency, and authentication failures are observed over a window agreed before deployment; this script does not monitor those thresholds automatically.
- [ ] A failed required check stops release approval or promotion, with evidence retained and a rollback decision made.

Only if the current release record does not already exist:

```bat
copy docs\release\RELEASE_RECORD.md .release\release-record.md
```

Do not overwrite completed entries. Record owners, approvers, environment, candidate and previous successful SHAs, artifact and backup references, reports, manual checks, unresolved issues, and the final decision. Mark unperformed steps as pending; do not invent results.

## 9. Rollback Preparation and Recovery Verification

Before deployment, complete the preparation in `DEPLOYMENT_CHECKLIST.md` and `ROLLBACK_PLAN.md`: prior successful frontend/backend artifacts and digests, valid configuration references, container digests, Keycloak configuration, database/knowledge backups, schema compatibility, restore permissions, and rehearsed host-specific steps.

If recovery is needed:

1. The release owner records the trigger and stops promotion. Coordinate a write pause if restoring data is necessary.
2. Decide whether a compatible application rollback is sufficient or whether data restoration or forward repair is required. Restoring a backup may lose later writes and must follow the approved recovery decision.
3. Restore the retained previous frontend/backend artifacts and compatible configuration through the actual hosting platform. Do not resolve fresh unpinned dependencies and assume they reproduce the old release.
4. Preserve Postgres volumes. Do not use `docker compose down -v` as an ordinary rollback step.
5. Rerun smoke tests against the **previous successful release ID and full SHA**, using `--report .release\rollback-smoke-results.json`. Do not retain the failed candidate's expected SHA.
6. Verify sign-in, chat, data, and operational metrics. Restore traffic/writes only after approval, and record recovery time and any data loss.

The repository does not contain complete production hosting or migration automation. The deployment operator must supply and rehearse concrete platform commands. If there is no known-good previous release, document that limitation and the recovery approach; do not claim a rollback rehearsal occurred.

## 10. GitHub Actions Integration

- `release-governance.yml` runs offline tests, contract coverage checks, and candidate evidence generation on PRs, pushes to Feifan/main, or manual invocation. It uploads artifacts but does not deploy the application.
- `post-release-smoke.yml` supports `workflow_dispatch` and `workflow_call`. The inspected branch had no complete existing deployment pipeline to connect automatically; the release owner must invoke it or wire it into the real deployment workflow after deployment succeeds.
- Supply `frontend_url`, `backend_url`, `keycloak_url`, `realm`, `expected_commit`, and `release_id` for the **actual deployed version**.
- A GitHub-hosted runner's localhost is not a developer's computer. Use the CLI for local services and runner-accessible URLs for hosted services.
- Manual workflow visibility can depend on the workflow being present on the default branch. Follow team branch policy and use the CLI until the workflow is available.
- Configure the real release pipeline to require a successful smoke job. Do not use `continue-on-error` to approve a failed required gate.

## 11. Evidence Retention and Completion Criteria

| Evidence | Required content |
| --- | --- |
| Unit-test output | Governance test result associated with the candidate SHA |
| manifest.json / release.md | Source SHA, model declarations, prompt hashes, dependencies, and limitations |
| backend-installed.json | Package versions from the actual backend virtual environment |
| env-validation.json | Current configuration checks and warnings |
| env-comparison.json, when applicable | Genuine prior/current variable-name differences without values |
| smoke-results.json | Target URLs, expected SHA, per-check results, and UTC timestamp |
| release-record.md | People, artifact/backup references, manual checks, blockers, and decision |
| rollback-smoke-results.json, when applicable | Validation after restoring the previous version |

`.release/` is gitignored, but its evidence must still be retained as approved release attachments or controlled artifacts. Preserve distinct attempts instead of overwriting the only failure evidence. Do not attach complete environment files or secrets. GitHub workflow artifacts do not automatically include unuploaded local reports.

Implementation completion means the governance tooling and documents exist and their checks have been verified. Release approval additionally requires the actual deployment's required checks and operational acceptance criteria to pass, with a completed release record.

## 12. Known Progress for This Execution

Based on terminal output supplied by the operator:

- The 17 governance unit tests completed with `OK`.
- Environment contract coverage returned `PASS environment contract-only`.
- Current configuration returned `PASS environment deployment`, with warnings for four legacy settings and unavailable historical comparisons.
- Generation with `--require-clean --publish-frontend` returned `dirty=False`.
- This runbook does not certify that the full application build, GitHub push, all seven live smoke checks, manual sign-in/chat, or a rollback rehearsal succeeded. Update the release record using actual subsequent evidence.
- The missing Brain health method is confirmed in the inspected baseline source. Confirm the exact cause of any 500 on the running instance using curl and backend logs.
