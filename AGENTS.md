# TrustDoc AI — persistent project rules

- Read the root `README.md` before making changes. The repository is complete through Part 5.1; the next scope is Part 6 only.
- Never rewrite or redesign completed modules just to fit a preferred architecture. Reuse actual APIs and inspect current source before calling interfaces.
- Maintain the central invariant: factual documentation claims should be evidence-grounded; no evidence is not proof of falsity. Keep generation and verification separate.
- Preserve Parts 1–5.1 compatibility, SQLite schema v2, migrations, repository-only SQL and transaction/history behavior. Avoid schema changes unless strictly necessary and justified.
- Constraints: CPU-first, free/open-source, no paid API dependency, no GPU-only path, no automatic model downloads. Do not make a real model a test dependency.
- Tests must run offline using fixtures, fake retrievers and scripted LLMs. Report exact pass/fail/skip counts and the reason for skips. Do not describe mocked tests as validation of real-model quality.
- Keep each phase focused. For Part 6 implement only documentation GENERATE plus its integration with existing VERIFY. Do not implement the UI or Part 7/8.
- Before edits, inspect `git status` and the intended diff. Prefer small, reversible changes; don't overwrite user data/source files by default.
- Run targeted tests first, then the complete available test suite and relevant demos. Inspect the final diff and update README instructions to match code exactly.
- Never claim success for commands that were not run. Clearly list model downloads or network-dependent checks not performed.
