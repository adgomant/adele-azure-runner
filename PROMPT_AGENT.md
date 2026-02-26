# Agent execution prompt (use in chat)

Read PROJECT_BRIEF.md and TODO.md. Implement the framework in this repo.

Rules:
- Do not hardcode secrets. Use env vars specified in config.
- Do not invent model IDs or endpoints: keep placeholders and read from YAML.
- Use `uv` for dependency management. Provide `pyproject.toml` and generate `uv.lock`.
- Work incrementally: complete Phase 0 then Phase 1, etc.
- After each phase, briefly summarize what changed and which files were added/modified.
- Prefer `azure-ai-inference` for Foundry inference and `typer` for CLI.
- Ensure idempotent runs: rerun should skip completed instance_id+model_id outputs.
- Use streaming IO (JSONL append). Avoid loading all outputs in memory.
- Add unit tests for parsing and resume/dedup.

Start now with:
1) Propose final architecture + data schemas + CLI commands (short).
2) Implement Phase 0 scaffold (pyproject, src layout, README skeleton, gitignore).
3) Then proceed to Phase 1.
