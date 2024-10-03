# Contributing

Thank you for your interest in contributing to this project! Here’s a quick guide to get started:

## How to Contribute
- Contributions are made through Pull Requests (PRs).
- Please ensure your PR is associated with an existing issue, bug, or feature request. If none exist, feel free to create one first for discussion.

## Development Setup
You can set up the development environment using the provided Makefile. Simply run:

```bash
make setup
```

Note that this **only works on OSX**. Additional Makefile commands are incoming for folks using Windows and Linux machines.

This will install all necessary dependencies and prepare your environment for development.

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
