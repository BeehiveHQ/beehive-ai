# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
setup:
	uv sync --locked
	uv run --locked pre-commit install

# Run tests without reinstalling Git hooks.
tests:
	uv run --locked pytest beehive/tests/ tests/

.PHONY: setup tests
