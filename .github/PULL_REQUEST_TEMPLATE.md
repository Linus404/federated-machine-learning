## Summary

<!-- What problem does this change solve, and how? -->

## Related work

<!-- Link an issue or identify the TODO.md item. Use "None" when not applicable. -->

## Verification

<!-- List the exact commands run and their outcomes. -->

## Checklist

- [ ] The change is focused and excludes unrelated edits.
- [ ] Tests cover new or changed behavior.
- [ ] `uv run --env-file .env.protocol python -m unittest discover -s tests -v` passes.
- [ ] `uv run --env-file .env.protocol ruff format --check .` passes.
- [ ] `uv run --env-file .env.protocol ruff check .` passes.
- [ ] Documentation reflects user-visible or configuration changes.
- [ ] No credentials, private data, datasets, or generated artifacts are committed.
- [ ] Scientific or security claims are supported by reproducible evidence.

## Risks and follow-up

<!-- Describe compatibility risks, limitations, or follow-up work. Use "None" when absent. -->
