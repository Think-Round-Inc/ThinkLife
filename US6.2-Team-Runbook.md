# User Story 6.2: Release Governance Runbook

This runbook provides operator commands for the [US6.2 README](docs/release/README.md). It covers the existing release tooling on branch `Feifan`; it does not deploy the app, repair its business logic or certify a production release.

## 1. Requirements and Deliverables

| Requirement | Tool / document | Output |
| --- | --- | --- |
| Model and prompt versions | `generate_release_manifest.py` | `.release/manifest.json`, `.release/release.md` |
| Dependency list | Same generator | Python, frontend and container inventory in release evidence |
| ENV changes | `validate_release_env.py` | `.release/env-validation.json` / optional comparison report |
| Deployment checklist | `docs/release/DEPLOYMENT_CHECKLIST.md` | Reviewed release steps |
| Rollback plan | `docs/release/ROLLBACK_PLAN.md` | Recovery procedure and release record |
| Post-release smoke | `post_release_smoke.py` | `.release/smoke-results.json` |

**Prerequisites:** Windows CMD commands below assume the repository root, Python 3.11 and Git. The release scripts use the standard library. The running application and Keycloak are required only for live smoke tests. Do not commit `.env` files or secrets.

## 2. Run the Offline Governance Tests

```bat
cd /d "C:\path\to\ThinkLife"
python -m unittest discover -s tests/release -v
```

Expected: `Ran 17 tests` followed by `OK`. These tests use mocks; `OK` does **not** indicate that the deployed application is healthy. If imports or paths fail, confirm that CMD is in the repository root and Python can find the release scripts.

## 3. Check Environment Contract Coverage

```bat
python scripts/validate_release_env.py --check-contract
```

Expected: `PASS environment contract-only`. This checks that detected source ENV names are covered by `docs/release/environment-contract.json`. It **does not** inspect actual ENV values or compare deployments. Review the contract when adding new application configuration names.

## 4. Validate Current Configuration and Historical Changes

Validate the local backend/frontend files (`backend/.env`, `frontend/.env.local`):

```bat
python scripts/validate_release_env.py
```

The output goes to `.release/env-validation.json`. A `PASS` may still include warnings. In the inspected baseline, `CORS_ORIGINS`, `DEBUG`, `PORT` and `NEXT_PUBLIC_APP_URL` may warn as unconsumed settings; warnings do not prove the application uses those values. Without prior files, the tool warns that a historical comparison was not performed.

Use genuine copies of the previous deployed configuration to compare variable **names**, never secret values:

```bat
python scripts/validate_release_env.py --previous-backend-env "C:\release-secrets\previous-backend.env" --previous-frontend-env "C:\release-secrets\previous-frontend.env" --strict --report .release\env-comparison.json
```

`--strict` makes warnings fail too. The paths above are examples and must exist. If the service uses process environment overrides, add `--process-env` when invoking with the matching variables; this does not automatically read a remote host's settings. Never fabricate previous ENV files by copying the current ones.

Common errors: missing file, placeholder/empty value, invalid number/URL, incomplete credential pair, cross-service Keycloak mismatch or a `NEXT_PUBLIC_` secret. Fix the configuration, not application business logic, then rerun. Reports must not contain secret values.

## 5. Generate Candidate Model, Prompt, and Dependency Evidence

```bat
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini
```

Check `.release/manifest.json` and `.release/release.md` for the commit, models, SHA-256 prompt-source versions and dependencies. `dirty=True` is acceptable for **review only**. `--openai-model` documents the intended Brain model; it does not set application environment variables or control Zoe/FAQ models. Source hashes may change due to non-prompt edits, and model names may be aliases rather than immutable provider versions.

For stronger rollback evidence, record installed packages from the **tested backend environment** after `.release/` exists:

```bat
backend\.venv\Scripts\python.exe -m pip list --format=json > .release\backend-installed.json
```

Keep tested runtime artifacts and container digests; requirement declarations alone do not guarantee a reproducible rebuild.

## 6. Generate Final Metadata Before Building

Commit all intended changes first and confirm `git status --short` has no pending non-ignored files:

```bat
git status --short
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini --require-clean --publish-frontend
cd frontend
npm ci
npm run build
cd ..
```

Expected: `dirty=False`. This creates `frontend/public/release.json` **before** the frontend build. The generated `.release/` directory and frontend metadata are gitignored; preserve them with the release artifacts. If the committed source changes, regenerate and rebuild. `0.7.0-rc1` is an example release ID; use the actual release settings.

## 7. Run Live Smoke Tests

First deploy/start the matching frontend (`:3000`), backend (`:8000`) and Keycloak (`:8080`) using the existing project/hosting instructions. Then in Windows CMD:

```bat
for /f %i in ('git rev-parse HEAD') do set RELEASE_SHA=%i
python scripts/post_release_smoke.py --release-id 0.7.0-rc1 --expected-commit %RELEASE_SHA% --frontend-url http://localhost:3000 --backend-url http://localhost:8000 --keycloak-url http://localhost:8080 --realm thinklife
echo %ERRORLEVEL%
```

Use `%%i` in a `.bat` file. Supply the SHA of the **deployed artifact**, not local HEAD, if they differ. Seven checks are `frontend_health`, `frontend_release`, `backend_identity`, `backend_health`, `brain_health`, `zoe_health` and `keycloak_oidc`. All require HTTP 200 and matching JSON semantics. The current backend identity check expects root version `0.7.0`.

Exit `0`: all pass. Exit `1`: at least one fails. Exit `2`: invalid CLI arguments (a report may not exist). The default report is `.release/smoke-results.json`, overwritten on reruns; use `--report` with different filenames to preserve separate attempts. Defaults are 3 attempts, 10 seconds per request, and 2 seconds between retries.

**Failures:** HTTP 500 indicates a server-side error; check service logs. 404 suggests a missing URL/endpoint; 401/403 may indicate access restrictions; redirects fail; connection/TLS errors suggest reachability issues; contract mismatch means JSON or release fields are wrong. Never turn a required failure into a pass.

The committed [historical smoke report](docs/release/smoke-results.json) recorded six passes and `brain_health` HTTP 500 on October 11, 2026. It does not describe the latest deployed service. The missing `CortexFlow.get_health_status()` in the inspected baseline belongs to the application owner, outside US6.2.

## 8. Manual Functional Verification and Sign-off

After the seven checks pass, verify sign-in/protected pages, a representative chat with approved test data, backend artifact identity and agreed error/latency thresholds. The smoke script does not test actual LLM generation, provider quota or full login. Test a separately deployed FAQ service independently if applicable.

Copy the release record template for an actual release (do not overwrite a completed record):

```bat
copy docs\release\RELEASE_RECORD.md .release\release-record.md
```

Fill in owner/approver, deployment SHA, previous release, artifacts, config and backup references, environment diff, evidence links and final decision. Mark unperformed steps **pending**, not done.

## 9. Rollback Preparation and Recovery Verification

Follow [Deployment Checklist](docs/release/DEPLOYMENT_CHECKLIST.md) and [Rollback Plan](docs/release/ROLLBACK_PLAN.md). Before deployment, retain the last known-good frontend/backend build, full commit SHA, compatible configuration, database/knowledge backups, and restore access.

If a required check fails: stop promotion, preserve the failed report, and let the release owner decide whether to roll back. Restore approved known-good artifacts and compatible configuration through the actual hosting platform; assess data/schema compatibility before restoring backups. **Do not run `docker compose down -v`** as a routine rollback step because it destroys the database volume. Rerun smoke tests with the **previous release ID and SHA**, then manually verify login, chat and data recovery. Record the outcome and any data loss. There is no automatic rollback or documented completed rehearsal in US6.2.

## 10. GitHub Actions Integration

`release-governance.yml` runs the 17 unit tests, contract-only validation, manifest generation and artifact upload. The [November 8 successful run](https://github.com/Think-Round-Inc/ThinkLife/actions/runs/42542535840) contains an Artifact with `manifest.json`, `release.md` and **contract-only** `env-validation.json`. It does not attest to live deployment.

`post-release-smoke.yml` is manually invoked or called by a deployment workflow, with deployed service URLs, full SHA and release ID. A GitHub-hosted runner cannot access your laptop's `localhost`. Pushing README changes does not automatically run live smoke checks; the smoke workflow must be invoked with reachable targets. Do not bypass a failing release gate.

## 11. Evidence Retention and Completion Criteria

Keep each release's manifest, release Markdown, ENV comparison (when genuine prior files exist), test result, smoke report, tested dependency versions, artifacts/backups and signed release record. `.release/` is gitignored and GitHub Actions artifacts expire, so archive approval evidence in the team's controlled release location. Never upload secrets.

**Implementation complete** means US6.2 tooling and documents exist and offline checks pass. **Release approved** requires the actual deployment's smoke and manual checks to pass, with release-owner sign-off. These are different statuses.

## 12. Known Progress for This Execution

- Verified offline governance run: [Success, commit `8ba9d0e`](https://github.com/Think-Round-Inc/ThinkLife/actions/runs/42542535840), including uploaded candidate evidence.
- Historical local smoke snapshot: `brain_health` HTTP 500; not a passed deployment gate.
- A new live smoke run, manual verification, production deployment and rollback rehearsal are **not evidenced** by the offline CI success.
