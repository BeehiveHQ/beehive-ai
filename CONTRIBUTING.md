# Contributing

Thank you for your interest in contributing to this project! Here’s a quick guide to get started:

## How to Contribute
- Contributions are made through Pull Requests (PRs).
- Please ensure your PR is associated with an existing issue, bug, or feature request. If none exist, feel free to create one first for discussion.

## Development Setup
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
make setup
```

This creates a Python 3.12 virtual environment in `.venv`, installs the locked
runtime and development dependencies, and installs Git hooks. On Windows, or
without Make, run `uv sync --locked` and `uv run --locked pre-commit install`.

Run commands with `uv run` (for example, `uv run pytest beehive/tests/ tests/`).
Use `uv add PACKAGE` for runtime dependencies and `uv add --dev PACKAGE`
for development tools. Commit both `pyproject.toml` and `uv.lock` after dependency
changes. The existing `dev` extra remains available for pip users.

## Submitting a PR
- Fork the repository and create your feature branch.
- Make your changes, ensuring you follow the project's coding standards. We use pre-commit hooks to ensure that consistency across contributors.
- Write tests to confirm the expected functionality of your changes. We rely heavily on mocking/patching via `unittest.mock`.
- Prior to submitting your PR, make sure that tests are passing:
  ```bash
  make tests
  ```
- Submit a PR with a clear description linking it to the related issue.

We appreciate your contribution!
