# Deployment checklist

Complete a copy of RELEASE_RECORD.md for every environment/release. An unchecked
required item prevents release approval.

## Before deployment

- [ ] Name release owner, approver, environment, UTC window and rollback operator.
- [ ] Record exact committed candidate SHA and previous successful release SHA.
- [ ] Run release unit tests and source environment-contract validation.
- [ ] Validate actual backend/frontend environment including process overrides;
      compare with protected previous config; review each added/changed/removed
      name. Store values only in the approved secret manager, not in evidence.
- [ ] Record secret version identifiers; verify previous credentials remain valid.
- [ ] Review model inventory and prompt hashes; record non-default OPENAI_MODEL
      explicitly and knowledge/index snapshot identifiers.
- [ ] Generate clean manifest plus frontend metadata **before** frontend build.
- [ ] Install/test dependencies in isolated build environments; `npm ci`, then
      `npm run build`. Archive Python installed versions, tested artifacts,
      lockfile, OS/runtime versions and container image digests.
- [ ] Resolve application blockers separately: baseline Brain health defect,
      production auth/config choices and any build/dependency errors. Governance
      tests do not certify the whole application's build or production readiness.
- [ ] Archive prior frontend/backend artifacts, protected env versions, Keycloak
      config and database/knowledge backups; verify restore access and rehearse.
- [ ] Review schema/index compatibility; name backup restore or forward-repair
      choice, expected downtime and possible writes lost if a restore is required.
- [ ] Approver confirms deployment/rollback steps appropriate to the actual host.

## Deployment

- [ ] Retain current service until candidate is ready when host supports this.
- [ ] Deploy the exact recorded backend/frontend artifacts and matching env.
- [ ] Confirm frontend public/release.json is included and caches refreshed.
- [ ] Verify backend artifact digest via host control plane (no backend SHA API).
- [ ] Verify Keycloak realm/client and database/index version compatibility.
- [ ] Record host deployment IDs and UTC start/end times.

## Post-deployment gate

- [ ] Run post_release_smoke.py or reusable Actions against deployed URLs and SHA.
- [ ] All seven checks pass; attach immutable JSON report to release record.
- [ ] Run manual sign-in, protected page and representative chat/RAG checks with
      approved test data; record provider-returned model version when available.
- [ ] Observe errors, latency and authentication failures over an agreed window;
      compare with thresholds recorded **before** deployment.
- [ ] On failure, hold promotion/traffic switch; release owner decides rollback
      using ROLLBACK_PLAN.md and preserves evidence. Never relabel a 500 as PASS.
- [ ] Record final approved / rejected / rolled-back decision, owner and time.
