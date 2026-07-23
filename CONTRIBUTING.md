# Contributing

Thank you for improving the project. Keep each contribution focused, tested,
and easy to review.

## Before opening an issue

- Use the bug report or feature request template so maintainers receive the
  information needed to reproduce or evaluate the change.
- Report suspected vulnerabilities through the private process in
  [`SECURITY.md`](SECURITY.md), not in a public issue.

## Development setup

The project supports Python 3.11 through 3.13 and uses
[`uv`](https://docs.astral.sh/uv/) for environments and dependencies.

```bash
git clone git@github.com:Linus404/federated-machine-learning.git
cd federated-machine-learning
uv sync
```

Create a branch from the latest `main` and keep unrelated changes separate.

## Make and verify changes

Follow the existing module structure and add or update tests for behavior
changes. Before committing, run:

```bash
uv run --env-file .env.protocol coverage run -m unittest discover -s tests -v
uv run --env-file .env.protocol coverage report
uv run --env-file .env.protocol ruff format --check .
uv run --env-file .env.protocol ruff check .
uv run --env-file .env.protocol mypy
```

Use `uv run --env-file .env.protocol ruff format .` to apply formatting. If a check cannot run in your
environment, explain why in the pull request rather than marking it complete.

Changes to experiment behavior or reported results must include enough
configuration, seed, dataset, and artifact information to reproduce the claim.
Do not describe illustrative privacy controls, local demos, or partial
experiments as production guarantees.

## Pull requests

- Open one pull request per coherent change.
- Explain the problem, the chosen solution, and any user-visible impact.
- Include the exact verification commands and their outcomes.
- Update documentation when commands, configuration, or behavior changes.
- Do not commit generated artifacts, datasets, credentials, or secrets.
- Keep commits small and use concise, informative commit messages.

Maintainers may ask for a smaller scope, additional evidence, or follow-up
changes before merging.

## Legal note

The repository is currently marked `UNLICENSED` and does not grant a general
license to use, copy, modify, or distribute its contents. Only contribute work
you have the right to submit. Accepting a contribution does not by itself
change the repository's licensing status.
