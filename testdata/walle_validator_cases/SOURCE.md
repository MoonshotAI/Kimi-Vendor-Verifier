# Walle Validator Cases

This directory is a snapshot of `MoonshotAI/walle` validator testdata.

- Source: https://github.com/MoonshotAI/walle/tree/main/testdata/validator_cases
- Commit: `e5c4ecf341c6af30e07856fdff3c49e47c2b3cfb`
- Tag: `v0.1.15`
- Synced on: 2026-08-31
- Previous snapshot: `cc1c6b7dab5496d5184677ecf4c3b95fc1bd1606` (`v0.1.10`, synced 2026-05-25)

The snapshot is committed so CI can run without fetching test cases from GitHub.
Only `valid.jsonl` cases are included in this repository for now; upstream `invalid.jsonl` cases are intentionally omitted.

Sync procedure (keeps this directory a faithful, conflict-free snapshot):

```bash
rsync -av --delete \
  --include='*/' --include='valid.jsonl' --include='FORMAT.txt' --exclude='*' \
  /path/to/walle/testdata/validator_cases/ \
  testdata/walle_validator_cases/validator_cases/
```

Always sync the whole directory as one unit and update the commit/tag/date
above in the same commit; do not hand-place individual cases, or the next
sync will produce duplicates or conflicts.

## Supplemental Go validator cases

`go_validator_cases.jsonl` covers schemas from upstream `validator_test.go`
that are absent from the JSONL snapshot. Each record contains the original
schema and its Canonical form generated with walle `v0.1.16`. Accepted cases
validate tool arguments against the Canonical form; rejected cases require the
endpoint to return HTTP 400.
