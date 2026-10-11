# User Story 6.2 — Release governance

Scope: release tooling and documentation only. No changes to application behavior,
model choices, prompt text, authentication, or CortexFlow. Python tools require
Python 3.9+ and only the standard library. Run from the repository root with
`python scripts/...`; no `backend` imports and no Unix shebang requirement.

| Task | Implementation | Evidence |
| --- | --- | --- |
| Model and prompt versions | `generate_release_manifest.py` | Source commit, configured Brain model, separate Zoe/FAQ/judge/TTS/embedding defaults, model references and SHA-256 prompt source versions |
| Dependency list | Same generator | `release.md` contains all three Python requirement lists, frontend direct dependencies with lock resolutions and container tags; manifest includes frontend transitive versions and hashes |
| Environment changes | `validate_release_env.py` + environment contract | Missing/placeholder/type checks, consumed-variable coverage, cross-service Keycloak checks, previous/current added/removed/changed names |
| Deployment checklist | `DEPLOYMENT_CHECKLIST.md` | Completed release record and evidence links |
| Rollback plan | `ROLLBACK_PLAN.md` | Prior immutable artifacts, protected configuration, data backups, rehearsed recovery and verification |
| Post-release smoke | `post_release_smoke.py` + Actions workflow | Strict HTTP and semantic checks, bounded retries, JSON report and failing exit code |

Initial generated dependency/model/prompt inventory: [BASELINE_RELEASE.md](BASELINE_RELEASE.md).
This is a preview, not release approval; regenerate for each release.

## Current baseline findings

Inspected Feifan at `f488f9ff18b459fff267d3e080a0e34fc769aeb4`.

- Brain config reads `OPENAI_MODEL` (default `gpt-4o-mini`), but Zoe's
  `ProviderSpec` independently hardcodes `gpt-4o-mini`. Setting the environment
  variable does **not** change Zoe. Manifest preserves that distinction.
- Model aliases are not immutable provider snapshot versions. Record the actual
  model returned by a representative provider request in the release record if
  available. Optional provider references are not evidence those providers ran.
- Prompt versions hash whole Python source containers, including assembly logic,
  evaluators, personality and fallback text. Dynamic user/history/retrieval data
  are not prompts stored in this repository; separately version deployed knowledge
  snapshots. Whole-file hashing can change on non-prompt edits too.
- Python dependency files are unpinned; `postgres:15` is mutable. Do not claim an
  exactly reproducible rollback from these declarations. Archive the tested venv
  package versions, wheels/build artifact and actual container image digests.
- `/api/brain/health` calls `CortexFlow.get_health_status()`, which is absent in
  this baseline. Expect a 500 after initialization. This is a **release blocker**,
  not a passed smoke check. The application owner must repair it in a separate
  scope before release approval; US6.2 does not patch core behavior.
- `PORT`, `DEBUG`, `CORS_ORIGINS`, `NEXT_PUBLIC_APP_URL` in the old setup guide are
  not consumed in the corresponding app source. Validator warns if present.
  Current backend entry point uses port 8000 and reload=True; CORS is hardcoded.
- Root backend version `0.7.0`, FastAPI API version `2.0.0` and frontend package
  version `0.7.0` have different meanings. Root version is not a Git commit.
- Docker Compose provisions only Keycloak/Postgres and uses development mode and
  default credentials. It is not a complete production deployment specification.

## Quick start (Windows CMD or PowerShell)

Use `python`, or replace it with `backend\.venv\Scripts\python.exe`.

```text
python -m unittest discover -s tests/release -v
python scripts/validate_release_env.py --check-contract
python scripts/validate_release_env.py
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1
```

`--check-contract` checks source coverage only. It is explicitly **not** deployment
validation. Ordinary validation reads `backend/.env` and `frontend/.env.local`;
missing files/required values fail. Optional variables use existing app defaults.
No environment file is created, printed, copied into reports or uploaded.
For deployment process overrides use `--process-env`; run with the same process
variables as the deployed services. Without that flag, only the files are checked.
Simple quoted/unquoted assignments, comments and `export` are supported; duplicate
keys, multiline values and `${...}` expansions fail rather than guessing.

To compare against protected copies of the preceding configuration:

```text
python scripts/validate_release_env.py --previous-backend-env "C:\release-secrets\previous-backend.env" --previous-frontend-env "C:\release-secrets\previous-frontend.env" --strict
```

Only variable **names** are reported for additions/removals/value changes; even
secret hashes are excluded. First release has no previous environment; review
warnings explicitly in the release record instead of claiming comparison occurred.
Keycloak URL/realm/client must agree in this baseline. If a future deployment
requires distinct internal/external issuers, review the auth design and contract.
Langfuse and Google OAuth credentials must be configured in pairs when enabled.
The optional standalone FAQ contract is documented; main validator and smoke scope
are backend/frontend/Keycloak. A release enabling FAQ needs its own API/RAG checks.

## Build metadata and release evidence

After committing changes, build from a clean checkout of the exact release commit:

```text
python scripts/generate_release_manifest.py --release-id 0.7.0-rc1 --openai-model gpt-4o-mini --require-clean --publish-frontend
```

Supply the actual deployed `OPENAI_MODEL` value explicitly if non-default.
This writes `.release/manifest.json`, `.release/release.md` and
`frontend/public/release.json`. Run it **before** `npm run build` and include
`public/` in the deployed frontend artifact. Generated output is gitignored;
retain it with the build artifact. A dirty candidate manifest is allowed for local
review, but the deployed metadata check rejects dirty builds.
This script does not change prompts, set application environment variables or
claim to resolve provider model aliases. Frontend metadata includes no secrets.

Capture tested backend installed versions using the backend venv:

```text
backend\.venv\Scripts\python.exe -m pip list --format=json > .release\backend-installed.json
```

If FAQ is included, capture its venv separately. `pip list` supplies inventory,
not a hash-locked installer. Keep the tested artifact/wheelhouse for rollback.
Use `npm ci` with the committed lockfile. Record actual Python, Node and npm
versions, container digests and knowledge/database backup identifiers.

## Run after deployment, and again after rollback

Replace the SHA with the full deployed commit (not a moving branch name):

```text
python scripts/post_release_smoke.py --release-id 0.7.0-rc1 --expected-commit FULL_40_CHARACTER_SHA --frontend-url http://localhost:3000 --backend-url http://localhost:8000 --keycloak-url http://localhost:8080 --realm thinklife
```

Checks frontend health and build SHA/release ID, backend identity and initialized
components, Brain health, Zoe readiness and Keycloak OIDC discovery. An HTML login
page, redirect, HTTP 500, invalid JSON, false readiness or stale metadata fails.
No LLM requests are made, so these checks do not prove response quality, provider
billing/quota availability or interactive login. The baseline has no backend
commit endpoint: independently verify deployed backend artifact identity.
Reports omit response bodies. Failure exits 1; malformed CLI arguments exit 2.

## GitHub Actions

`Release governance` runs offline checks and uploads candidate evidence on pushes
or pull requests. It requires no API keys. It does not deploy the application.
`Post-release smoke tests` is enabled as a reusable workflow (`workflow_call`) and
manual workflow. There was no existing deployment pipeline in this branch to hook
into. The release owner must invoke it after deployment, or add this job to the
actual deployment workflow (replace example URLs and release ID):

```yaml
post-release:
  needs: deploy
  uses: ./.github/workflows/post-release-smoke.yml
  with:
    frontend_url: https://frontend.example.com
    backend_url: https://backend.example.com
    keycloak_url: https://auth.example.com
    realm: thinklife
    expected_commit: ${{ github.sha }}
    release_id: 0.7.0-rc1
```

GitHub-hosted runners cannot reach your Windows localhost. Run the CLI locally
for local services. GitHub manual dispatch visibility can require the workflow
on the default branch; respect team branch policy and use CLI until merged.
Do not approve/promote a deployment on smoke failure. Preserve the report and
follow the rollback plan; rollback is deliberately operator-controlled.
