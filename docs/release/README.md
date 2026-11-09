# US6.2: Structured release governance

**Epic:** 6.2 — structured release governance so that deployments are traceable and reversible
**Author:** Feifan Li
**Branch:** Feifan
**PR:** Not opened; add the PR link when available

## User story

Structured release governance so that deployments are traceable and reversible.

## Summary of changes

* Added release manifests that record the source commit, working-tree state, model references, SHA-256 prompt-source versions, and dependency inventories.
* Added environment-contract validation and previous/current configuration comparison, together with deployment, release-record, and rollback documents.
* Added seven post-release smoke checks, 17 release-tool unit tests, and GitHub Actions workflows that retain reviewable JSON and Markdown evidence.

## Files touched

The file list is relative to the `arvind` baseline, `f488f9ff18b459fff267d3e080a0e34fc769aeb4`. New/Changed refers to that comparison.

| File | Change | Why |
| --- | --- | --- |
| `.gitignore` | Changed | Exclude generated `.release/` evidence and `frontend/public/release.json` from source commits. |
| `README.md` | Changed | Link the main project documentation to US6.2 release guidance. |
| `.github/workflows/release-governance.yml` | New | Run offline release gates and upload candidate evidence on supported events. |
| `.github/workflows/post-release-smoke.yml` | New | Support manual/reusable live smoke checks and report uploads. |
| `scripts/release_common.py` | New | Share Git, SHA-256, source inventory, dotenv parsing, and JSON-writing helpers. |
| `scripts/generate_release_manifest.py` | New | Generate model, prompt, dependency, and frontend release metadata. |
| `scripts/validate_release_env.py` | New | Validate environment contracts and report configuration changes by name only. |
| `scripts/post_release_smoke.py` | New | Check deployed services and release identity with bounded retries. |
| `tests/release/test_release.py` | New | Test manifest generation, environment validation, and smoke failure handling. |
| `docs/release/environment-contract.json` | New | Define reviewed variable names, requirements, types, defaults, and source references. |
| `docs/release/BASELINE_RELEASE.md` | New | Provide an initial model, prompt, and dependency preview; not release approval. |
| `docs/release/README.md` | New | Explain implementation, execution, evidence locations, and acceptance mapping. |
| `docs/release/DEPLOYMENT_CHECKLIST.md` | New | Define preparation, deployment, and post-deployment approval checks. |
| `docs/release/RELEASE_RECORD.md` | New | Capture per-release owners, commits, artifact/config references, evidence, and decisions. |
| `docs/release/ROLLBACK_PLAN.md` | New | Define recovery authority, retained material, restoration, and re-verification. |
| `docs/release/smoke-results.json` | New | Preserve a reviewed local smoke-test snapshot for teammates to read on GitHub. |
| `US6.2-Team-Runbook.md` | New | Provide the team's longer operational walkthrough. |
| `frontend/next-env.d.ts` | Changed | Existing branch difference changes the generated route-type import from `.next/dev/types` to `.next/types`; not required by US6.2. |

The `next-env.d.ts` difference was introduced in later commit `4f36cd0`, not the initial US6.2 commit `b7dd584`. It is listed for integration review, not as a release-governance feature.

## Pseudocode

Manifest generation — `scripts/generate_release_manifest.py`:

```text
FUNCTION generate_release_manifest(arguments):
    Validate the release ID and declared Brain model identifier
    Read Git HEAD and whether tracked/untracked, non-ignored files are dirty
    Read application versions and model references from source
    Keep Brain configuration separate from Zoe, FAQ, judge, TTS, and embedding defaults
    Hash each selected prompt source file with SHA-256
    Inventory Python requirements, frontend package/lock data, and container image tags
    IF require-clean is enabled AND the working tree is dirty:
        RETURN failure without writing new release metadata
    WRITE manifest.json and release.md to the output directory
    IF publish-frontend is enabled:
        WRITE release ID, source commit, schema version, and dirty flag to public/release.json
    RETURN exit code 0; return 1 for handled generation failures
```

Environment validation — `scripts/validate_release_env.py`:

```text
FUNCTION validate_release_environment(arguments):
    Load the reviewed environment contract
    Scan supported source references and flag consumed names missing from the contract
    IF check-contract is disabled:
        Read backend and frontend environment files
        IF process-env is enabled:
            Overlay known contract variables from the current process
        Validate required values, placeholders, types, ranges, URLs, and credential pairs
        Check frontend public-secret names and cross-service Keycloak settings
        FOR each scope:
            IF a previous environment file was supplied:
                Compare added, removed, and changed variable names
            ELSE:
                Add a warning that comparison was not performed
    SET passed = (no errors) AND (strict is off OR no warnings)
    WRITE the JSON report with names and findings, excluding secret values and hashes
    PRINT PASS/WARN/FAIL messages
    RETURN 0 when passed, otherwise 1
```

Live smoke testing — `scripts/post_release_smoke.py`:

```text
FUNCTION run_post_release_smoke(arguments):
    Validate base URLs, full lowercase commit SHA, and retry/time limits
    IF CLI arguments are invalid:
        RETURN exit code 2 before running checks
    Build the seven frontend, backend, Brain, Zoe, and Keycloak checks
    FOR each check:
        Retry up to the configured attempt limit
        Require HTTP 200 without following redirects
        Parse a size-limited JSON object and evaluate the response predicate
        For frontend metadata, require the expected commit, release ID, and dirty=false
        Record pass/fail, attempt count, and a safe failure reason
        PRINT one PASS or FAIL line
    SET overall passed to true only when all seven checks pass
    WRITE the JSON report, including failed checks but excluding response bodies
    RETURN 0 when all checks pass, otherwise 1
```

GitHub Actions — the two workflow files:

```text
FUNCTION run_release_workflows(event, inputs):
    IF the governance workflow is triggered:
        Checkout source and set up Python 3.11
        Run release unit tests and source-contract coverage validation
        Generate clean candidate evidence for the checked-out commit
        Upload manifest.json, release.md, and env-validation.json
    IF the smoke workflow is manually invoked or called by another workflow:
        Checkout source and set up Python 3.11
        Run smoke tests against the supplied deployed URLs, commit, and release ID
        Always attempt to upload smoke-results.json, even when checks fail
        Fail the upload step if the report does not exist
    RETURN workflow status and any successfully uploaded artifacts
```

Operator-controlled deployment and rollback — the checklist and rollback plan:

```text
FUNCTION operate_release(candidate, previous_known_good_release):
    Record owners, artifact digests, protected config references, backups, and thresholds
    Complete environment review and generate clean metadata before building
    Retain the previous tested artifacts and verify recovery access
    Deploy the recorded candidate artifacts using the hosting platform
    Run smoke tests plus manual login/chat checks and observe the agreed metrics
    IF required checks pass AND the release owner approves:
        Record approval
    ELSE:
        Hold promotion and preserve the failed report
        Let the release owner decide whether to roll back
        IF rollback is approved:
            Restore prior artifacts and compatible configuration
            Restore data only according to the approved compatibility/recovery decision
            Re-run smoke tests with the previous release ID and commit
            Record recovery checks and the final decision
    RETURN the completed release record; these steps are not automated by the scripts
```

## Setup

* Packages added: None. The release tools and tests use only the Python standard library. Use Python 3.11 to match CI, and make Git available on PATH. Application build/runtime dependencies remain in the existing project dependency files.
* Environment variables: None added by US6.2. Existing contract names are `ADMIN_EMAILS`, `ANTHROPIC_API_KEY`, `CHROMA_DB_DIR`, `CHUNK_SIZE`, `CONTEXT_TXT_PATH`, `EMPATHY_CSV_PATH`, `GEMINI_API_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `KEYCLOAK_CLIENT_ID`, `KEYCLOAK_REALM`, `KEYCLOAK_URL`, `KNOWLEDGE_JSON_PATH`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `MAX_EMPATHY_EXAMPLES`, `NEXTAUTH_SECRET`, `NEXT_PUBLIC_ADMIN_EMAILS`, `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_KEYCLOAK_CLIENT_ID`, `NEXT_PUBLIC_KEYCLOAK_REALM`, `NEXT_PUBLIC_KEYCLOAK_URL`, `NODE_ENV`, `OPENAI_API_KEY`, `OPENAI_MAX_TOKENS`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`, `TAVILY_API_KEY`, and `ZOE_VECTOR_DB_PATH`. Requirements and scope are defined in [environment-contract.json](environment-contract.json); optional credentials are needed when their provider is actually used.
* Config flags: Manifest: `--release-id` required; `--openai-model` defaults to `gpt-4o-mini`; `--output-dir` defaults to the repository's `.release/`; `--require-clean` and `--publish-frontend` default to off. Validator: `--backend-env` defaults to `backend/.env`, `--frontend-env` to `frontend/.env.local`, and `--report` to `.release/env-validation.json`; previous-file arguments are optional; `--check-contract`, `--process-env`, and `--strict` default to off. Smoke: `--release-id` and `--expected-commit` are required; frontend/backend/Keycloak URLs default to `http://localhost:3000`, `http://localhost:8000`, and `http://localhost:8080`; `--realm` defaults to `thinklife`; `--attempts` to 3, `--timeout` to 10 seconds, `--delay` to 2 seconds, and `--report` to `.release/smoke-results.json`.
* Migrations or seed data: None. US6.2 does not alter application schemas, seed databases, or start services. Live smoke tests require the existing frontend, backend, and Keycloak deployment to be running.

## How to run

Use the repository root as the working directory, not `scripts/`. On a new machine, obtain the branch with the first two commands below; skip them if the repository is already checked out. No additional `pip install` is needed for US6.2. The tools are CLI commands, so there is no separate US6.2 server to start.

```bash
git clone --branch Feifan --single-branch https://github.com/Think-Round-Inc/ThinkLife.git ThinkLife
cd ThinkLife
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini
python -m unittest discover -s tests/release -v
python scripts/validate_release_env.py --check-contract
```

A successful unit-test run ends with `Ran 17 tests` and `OK`. Contract coverage prints `PASS environment contract-only`. Candidate generation writes `.release/manifest.json` and `.release/release.md`; `dirty=True` is permitted for review, but does not qualify the candidate for deployment.

Validate actual configuration separately. The following reads `backend/.env` and `frontend/.env.local`; missing required files or values fail:

```bash
python scripts/validate_release_env.py
```

Compare with genuine protected copies of the previous deployment's configuration. The Windows paths below are examples and must identify existing files; do not create a supposed previous version by copying today's configuration.

```bash
python scripts/validate_release_env.py --previous-backend-env "C:/release-secrets/previous-backend.env" --previous-frontend-env "C:/release-secrets/previous-frontend.env" --strict --report .release/env-comparison.json
```

`--strict` fails on warnings as well as errors. Without previous files, comparison is not performed and warnings are recorded. For a first deployment, record that limitation and review warnings in the release record. Use `--process-env` only when the invoking process has the same overrides as the deployment. Contract-only validation does not inspect actual deployment values or test credentials.

After the final source and documentation changes are committed, generate clean release metadata before building:

```bash
git status --short
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini --require-clean --publish-frontend
cd frontend
npm ci
npm run build
cd ..
```

`git status --short` should show no pending, non-ignored changes. Successful generation prints `Generated manifest.json and release.md; dirty=False`. Include the generated `frontend/public/release.json` in the deployed frontend artifact. If source or dependency files change, commit the new candidate, regenerate metadata, rebuild, and retest. Never edit a dirty flag to force a pass.

`0.7.0-rc1` is an example release label. `--openai-model` records the actual Brain model selected for the deployment; it does not set environment variables or change the application model. These arguments do not select two environments.

Follow the [deployment checklist](DEPLOYMENT_CHECKLIST.md) and existing [project setup](../../README.md) to start or deploy the application services. Then run smoke tests from the checkout of the exact deployed release. The following works in Bash/Git Bash or PowerShell; in Windows CMD, run `git rev-parse HEAD` first and replace `"$(git rev-parse HEAD)"` with that full SHA.

```bash
python scripts/post_release_smoke.py --release-id 0.7.0-rc1 --expected-commit "$(git rev-parse HEAD)" --frontend-url http://localhost:3000 --backend-url http://localhost:8000 --keycloak-url http://localhost:8080 --realm thinklife
```

Use the actual deployed URLs and release ID. If the checkout differs from the deployed artifact, supply the artifact's recorded full SHA instead of the current HEAD. After rollback, supply the previous release's ID and SHA.

The seven checks are `frontend_health`, `frontend_release`, `backend_identity`, `backend_health`, `brain_health`, `zoe_health`, and `keycloak_oidc`. Exit code 0 means all seven passed; 1 means at least one failed; 2 means invalid CLI arguments. A completed check run writes the report even when checks fail. Argument errors, cancellation, or failures before report writing can leave no report.

**Where teammates can read results:**

* Local results: `.release/smoke-results.json` contains `release_id`, `expected_commit`, `checked_at`, each check's result/attempts/reason, and the overall `passed` flag. The default path is overwritten on a later run; use `--report` with a distinct filename when preserving evidence.
* GitHub Code: [saved local smoke report](smoke-results.json). This is a deliberately committed snapshot, not an automatically refreshed result. Copy a reviewed report here, commit it, and push to `Feifan` when sharing a new snapshot. Preserve its original timestamp and tested SHA; the new evidence commit does not mean the new HEAD was tested.
* GitHub Actions: open the matching run, select **Summary**, then **Artifacts**. `release-governance-<SHA>` contains the candidate manifest, release Markdown, and contract-only environment report. A completed smoke workflow uploads `post-release-smoke-<run_id>-<run_attempt>` with `smoke-results.json`, including failure reports when they were produced. Artifacts have retention limits; archive approved release evidence with the release record.

The `.release/` directory is gitignored, so local reports do not appear in GitHub Code automatically. The current [successful governance run](https://github.com/Think-Round-Inc/ThinkLife/actions/runs/38834373199) verifies release tooling and source-contract coverage; it does not certify a live deployment or contain a live smoke result.

The committed smoke snapshot was checked at `2026-10-11T20:20:16.364546+00:00`, for release `0.7.0-rc1` and commit `b7dd584d4c21e0133b3ab920752d747e97fd7d1c`. Six checks passed; `brain_health` failed with `HTTP 500` after three attempts, so overall `passed` is `false`. This historical report does not attest to the current branch HEAD.

`Release governance` runs on pull requests, pushes to `Feifan` or `main`, and manual dispatch. `Post-release smoke tests` runs only through manual dispatch or `workflow_call`; pushing code does not automatically test live services. Its inputs are `frontend_url`, `backend_url`, `keycloak_url`, `realm`, `expected_commit`, and `release_id`. The deployment owner must invoke it after deployment or connect it to the hosting pipeline. GitHub-hosted runners cannot reach a developer's localhost; use the CLI for local services. Manual dispatch requires the workflow on the repository's default branch; follow team integration policy.

On a required check failure, hold release promotion, retain the report, and follow the [rollback plan](ROLLBACK_PLAN.md). Complete a copy of [RELEASE_RECORD.md](RELEASE_RECORD.md) for the actual release, including evidence links, reviewed environment changes, previous artifacts, backup references, owner decisions, and recovery verification.

## Acceptance criteria mapping

`Done` below means the requested US6.2 tooling or document has been delivered. It does not mean a deployment has passed its release gate or a rollback rehearsal has been completed.

| Task or criterion (from the epic) | Implemented in | Proof | Status |
| --- | --- | --- | --- |
| Document model versions & prompt versions | `generate_release_manifest.py:build`; `release_common.py:model_inventory`, `prompt_inventory`, `digest` | `ManifestTests.test_models_separate_and_dependency_coverage`; `ManifestTests.test_prompt_versions_use_actual_content`; generated manifest and [baseline preview](BASELINE_RELEASE.md). | Done |
| Update dependency list in release doc | `generate_release_manifest.py:dependency_inventory` and Markdown generation | `ManifestTests.test_models_separate_and_dependency_coverage`; Python/frontend/container lists in generated `release.md`. | Done |
| Validate environment variable changes | `environment-contract.json`; `validate_release_env.py:validate`, `changes`, `main` | `EnvironmentTests.test_source_contract_complete`, `test_diff_reports_names_only`, and `test_cli_matching_env_and_changed_secret`; deployment comparison still requires real prior files. | Done |
| Prepare deployment checklist | `docs/release/DEPLOYMENT_CHECKLIST.md`; `docs/release/RELEASE_RECORD.md` | [Checklist](DEPLOYMENT_CHECKLIST.md) covers pre-deployment, deployment, and post-deployment gates; [record template](RELEASE_RECORD.md) captures owner approval and evidence. | Done |
| Prepare rollback plan | `docs/release/ROLLBACK_PLAN.md` | [Plan](ROLLBACK_PLAN.md) defines triggers, prior artifacts/configuration, data compatibility, recovery steps, and verification; a real rehearsal remains operational work. | Done |
| Enable post-release smoke tests | `post_release_smoke.py:checks`, `check`, `main`; `.github/workflows/post-release-smoke.yml` | `SmokeTests.test_cli_all_pass_and_brain_failure_exit`, `test_version_mismatch_and_dirty_fail`, and `test_transient_failure_retries`; [saved live report](smoke-results.json) correctly records the unresolved Brain failure. | Done |

All named tests are in [tests/release/test_release.py](../../tests/release/test_release.py). Smoke unit tests use a local mock HTTP server; their success is separate from live-service health.

## Dependencies and conflicts

* Depends on: The agreed `arvind` baseline and its current source/configuration interfaces. Offline release gates require Python and Git; live smoke checks additionally require reachable frontend, backend, and Keycloak services. No other prerequisite story ID was supplied.

## Known limitations

* The inspected baseline's `/api/brain/health` calls the missing `CortexFlow.get_health_status()` method. The saved report records HTTP 500. This remains a release blocker for the application owner; US6.2 does not skip it or change it to PASS.
* No live smoke run, production deployment, or rollback rehearsal was performed as part of this README rewrite. `RELEASE_RECORD.md` remains a template to complete for each real release.
* Verify backend artifact digests independently and perform the required manual checks. The backend identity predicate currently expects root version `0.7.0`.
* The environment contract is reviewed, maintained configuration, not a fully automatic schema generator. Source coverage recognizes supported literal access patterns plus documented provider-key mappings. The dotenv parser rejects duplicates, multiline values, and unresolved expansions. Unknown names warn; strict mode fails on warnings. Validation does not prove credentials work.
* The smoke workflow is not yet wired to an automatic deployment pipeline.

