# Rollback plan

## Authority and triggers

Release owner named in RELEASE_RECORD.md owns the decision; designated operator
executes recovery. Trigger on a failed required smoke check after bounded retries,
wrong release identity, sign-in/chat regression, incompatible state, or breach of
the release record's pre-agreed error/latency thresholds. Hold promotion first.
No script automatically destroys data or resets a branch.

## Required recovery material

Before deploying, retain last known good frontend/backend artifacts by digest,
full source SHA, generated metadata and dependency inventory; protected environment
and secret version references; Keycloak realm/client configuration; database and
knowledge/index snapshots. Confirm old credentials are still usable. An old Git
SHA alone does not restore unpinned dependencies, remote models or persisted state.
Do not roll back a revoked credential; issue an approved replacement instead.

## Recovery procedure

1. Record trigger, failed checks and incident timestamp. Stop promotion and pause
   writes if database recovery may be needed; preserve candidate artifacts/logs.
2. Confirm the previous release with the owner and assess schema/index compatibility.
   For code-only compatible changes, leave persisted state intact. For incompatible
   changes, obtain the pre-agreed restore decision, quiesce writers and restore the
   matching snapshot or use an approved forward repair. Record potential data loss.
3. Use the hosting platform's rollback/redeploy function to select the archived
   previous **backend and frontend** artifacts. Restore compatible protected config
   and Keycloak config as needed. Do not resolve fresh unpinned dependencies.
4. Restart/redeploy affected services using the host runbook. Confirm image/artifact
   digests and database/index version; ensure previous frontend release.json matches.
   Never run `docker compose down -v`: it removes the persistent Postgres volume.
5. Run smoke tests with the **previous** release ID and full previous SHA. Verify
   backend artifact identity independently. Then verify sign-in and a representative
   chat, and observe metrics for the agreed recovery window.
6. Reopen traffic/writes only after recovery approval. Attach smoke results and UTC
   recovery time to the record; document data loss, remaining problems and follow-up.

For local development, stop the current frontend/backend terminals and run the
archived known-good build with its matching venv/config, keeping Keycloak/Postgres
volumes. This branch contains no production hosting automation or DB migration
framework, so host-specific commands and backup IDs must be entered in the record
and rehearsed before approval. Never invent a successful rollback rehearsal.
