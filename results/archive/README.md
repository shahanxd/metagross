# results/archive: superseded evidence (kept for the record, not current)

Nothing in this folder describes the current stack. The claims ledger (`results/claims.csv`) does not read these
files, because it only scans `results/*.json` and `results/*/*.json`.

| Folder | What it is | Why it is archived |
|---|---|---|
| `verify_dev/`, `verify_dev2/` | DEV verification passes, tier-0 and rendered stereo, made on the development laptop (Windows, Chrome GPU renderer) before 2026-09-30 11:00 | Older stack: before the supervisor, localiser and safety-gate fixes (see `docs/BUILD_LOG.md`) |
| `runs_dev_tier0_old_stack/`, `runs_dev3_tier0/`, `runs_integration/`, `runs_dev_stereo/` | Earlier DEV closed-loop batches | Older stack. The current DEV results are in `results/runs_dev_tier0/` |
| `eval_run1/` | EVAL run 1 on commit 3cd62df (FULL 17/60) | Superseded by run 2 after a DEV-confirmed regression fix. Disclosed in `docs/EVAL_PREREGISTRATION.md` |
| `redteam_notes.md` | Red-team review of the older stack | Most findings were fixed afterwards (see the build log) |
