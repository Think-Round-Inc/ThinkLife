# US6.2: Structured release governance

**Epic:** 6.2 — structured release governance so that deployments are traceable and reversible  
**Author:** Feifan Li  
**Branch:** `Feifan`  
**PR:**

## User story

As a release owner, I want structured release governance so that deployments are traceable and reversible.

## Summary of changes

Implemented six release-governance tasks: model/prompt version records, dependency inventories, environment-change validation, a deployment checklist, a rollback plan, and post-release smoke tests. The tools record evidence and report failures; they do not deploy, fix application defects, or roll back services automatically.

## Files touched

Compared with the agreed `arvind` baseline (`f488f9ff18b459fff267d3e080a0e34fc769aeb4`):

| File | Change | Purpose |
| --- | --- | --- |
| `.gitignore` | Changed | Ignore generated release evidence and frontend metadata. |
| `README.md` | Changed | Link to US6.2 documentation. |
| `scripts/release_common.py` | New | Shared Git, hashing, inventory and dotenv utilities. |
| `scripts/generate_release_manifest.py` | New | Generate model, prompt, dependency and release records. |
| `scripts/validate_release_env.py` | New | Validate ENV configuration and compare previous/current names. |
| `scripts/post_release_smoke.py` | New | Verify deployed HTTP endpoints and release identity. |
| `tests/release/test_release.py` | New | Unit tests for release tooling. |
| `docs/release/environment-contract.json` | New | Reviewed ENV rules and source references. |
| `docs/release/BASELINE_RELEASE.md` | New | Historical preview (not release approval). |
| `docs/release/README.md` | New | US6.2 overview and acceptance evidence. |
| `docs/release/DEPLOYMENT_CHECKLIST.md` | New | Required pre-/post-deployment steps. |
| `docs/release/RELEASE_RECORD.md` | New | Template for approval and evidence links. |
| `docs/release/ROLLBACK_PLAN.md` | New | Operator-controlled recovery procedure. |
| `docs/release/smoke-results.json` | New | Historical local smoke-test snapshot. |
| `US6.2-Team-Runbook.md` | New | Operator commands and troubleshooting. |
| `.github/workflows/release-governance.yml` | New | Offline CI checks and candidate artifact upload. |
| `.github/workflows/post-release-smoke.yml` | New | Manual/reusable deployed-service smoke workflow. |
| `frontend/next-env.d.ts` | Changed | Existing branch difference; not required for US6.2. |

The `frontend/next-env.d.ts` difference came from a later branch commit, not the initial US6.2 implementation. Review it separately when integrating.

## Pseudocode

**Manifest and dependency inventory** (`generate_release_manifest.py`):

```text
Read release ID and source commit; check working-tree state.
Collect configured/model-source references, prompt-source SHA-256 hashes,
Python requirements, frontend package/lock data and container tags.
If --require-clean and the tree is dirty: fail.
Write .release/manifest.json and .release/release.md.
If requested, write frontend/public/release.json before the build.
```

**Environment validation** (`validate_release_env.py`):

```text
Check source ENV names against environment-contract.json.
Unless --check-contract: read current backend/frontend ENV files,
validate required/configured values, and compare genuine prior files if given.
Report added/removed/changed variable NAMES, not secret values.
Write JSON findings; fail on errors (or warnings with --strict).
```

**Post-release checks** (`post_release_smoke.py`):

```text
Validate target URLs, release ID and full expected commit SHA.
Check seven frontend/backend/Brain/Zoe/Keycloak endpoints with bounded retries.
Require valid HTTP 200 JSON and expected healthy/release fields.
Write .release/smoke-results.json; return failure if any check fails.
```

**Approval and rollback** (checklist and rollback plan):

```text
Record candidate and previous known-good artifacts, configuration and backups.
Deploy using the team's hosting process; run smoke and manual checks.
Approve only when required checks pass and the release owner signs off.
On failure, preserve evidence and follow an operator-approved rollback;
verify the restored release with its own ID and commit SHA.
```

## Setup

- **Requirements:** Git and Python 3.11, run from the repository root. US6.2 scripts need no new Python packages; live deployment/build uses the application's existing dependencies.
- **Configuration:** No new environment variables. See [environment-contract.json](environment-contract.json); current files default to `backend/.env` and `frontend/.env.local`. Never commit secrets.
- **Services:** Offline tools need no running application. Live smoke tests require reachable frontend, backend and Keycloak. No database migrations are introduced by US6.2.

## How to run

Run from the **repository root**. See [US6.2-Team-Runbook.md](../../US6.2-Team-Runbook.md) for Windows CMD examples, troubleshooting and release sign-off.

```bash
python -m unittest discover -s tests/release -v
python scripts/validate_release_env.py --check-contract
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini
python scripts/validate_release_env.py
```

The example release ID and model must be replaced with the actual deployment values. The first command should finish with `Ran 17 tests` and `OK`. Contract-only validation checks source coverage, **not** deployed settings. For historical ENV changes, supply genuine previous backend/frontend ENV files via `--previous-backend-env` and `--previous-frontend-env`; use `--strict` only when warnings must also fail the gate.

After committing the final candidate, generate clean metadata **before** building the frontend:

```bash
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini --require-clean --publish-frontend
```

A successful clean run reports `dirty=False`. Do not edit the dirty flag or treat a dirty candidate as deployment-ready. Deploy the matching artifact, then run `post_release_smoke.py` with `--release-id`, `--expected-commit` (the deployed 40-character SHA), and the correct service URLs. Exit `0` means all seven checks passed; exit `1` means at least one failed.

**Evidence locations:** Local outputs are in ignored `.release/`. The [successful Release governance workflow](https://github.com/Think-Round-Inc/ThinkLife/actions/runs/42542535840) uploaded candidate `manifest.json`, `release.md` and **contract-only** `env-validation.json` as an Actions Artifact. This is not a live-deployment pass. A separate manually invoked/reusable smoke workflow uploads smoke results when produced. The committed [smoke snapshot](smoke-results.json) is historical and records `brain_health` HTTP 500, not a passing release.

## Acceptance criteria mapping

| US6.2 task | Implementation | Evidence / status |
| --- | --- | --- |
| Document model & prompt versions | Manifest generator and prompt hashes | `manifest.json`, `release.md` — implemented |
| Update dependency list | Requirements, frontend lock and container inventory | Generated `release.md` — implemented |
| Validate environment changes | ENV validator and contract | `env-validation.json`; comparison requires genuine prior ENV files — implemented |
| Prepare deployment checklist | `DEPLOYMENT_CHECKLIST.md` and `RELEASE_RECORD.md` | Templates available; release-specific sign-off pending |
| Prepare rollback plan | `ROLLBACK_PLAN.md` | Procedure documented; actual rehearsal not evidenced |
| Enable post-release smoke tests | Smoke CLI and Actions workflow | Seven checks implemented; historical live report has a failure |

## Dependencies and conflicts

Depends on the `arvind` baseline, Git/Python and the existing application interfaces. Live tests depend on running services and correct deployment metadata. No other story IDs were supplied. Application-owned health defects, deployment hosting decisions and merge conflicts must be resolved by their respective owners; US6.2 does not change core application behavior.

## Known limitations

- The historical `/api/brain/health` failure (`HTTP 500`, missing `CortexFlow.get_health_status()` in the inspected baseline) remains a release blocker until the application owner verifies a fix. Never mark it PASS without a passing live check.
- The successful offline CI run does not prove actual ENV differences, working credentials, healthy deployed services, login/chat behavior or a successful rollback.
- Model aliases and source-file hashes are not immutable provider snapshot IDs; unpinned Python requirements and mutable container tags also limit reproducibility. Retain tested build artifacts and actual dependency/image versions.
- Smoke checks use the current backend identity expectation (`0.7.0`); update the check with an intentional app version change. Deployment, rollback and post-release workflow invocation remain operator-controlled.
