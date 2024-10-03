PYTHON_VERSION:=3.12

# Set up dev environment
setup:
	brew install pyenv pipenv pre-commit
	pyenv install --skip-existing $(PYTHON_VERSION)
	pipenv install -e '.[dev]'
	pipenv run pre-commit install


# Run unit and integration tests
tests: setup
	pipenv run pytest beehive/tests/


.PHONY: setup tests
