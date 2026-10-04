# Contributing to Turing

Thank you for your interest in contributing.

## Before You Start

- Check existing issues before opening a new one
- For large changes — open an issue first to discuss
- Small fixes (typos, docs) — PR directly

## Setup

See README.md

## Development Workflow

1. Fork the repository
2. Create a branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run tests: `make test` (or `npm test` for frontend / `pytest Turing_Test_Suite/backend` for backend)
5. Commit: follow commit format below
6. Push and open a PR

## Commit Format

feat:     new feature
fix:      bug fix
docs:     documentation only
style:    formatting, no logic change
refactor: code change, no feature/fix
test:     adding tests
chore:    build, deps, config

## PR Requirements

- [ ] Tests pass locally
- [ ] New features have tests
- [ ] Documentation updated
- [ ] No new console.logs
- [ ] No hardcoded values
- [ ] Follows existing code style

## What We Will NOT Merge

- Breaking changes without discussion
- New dependencies without justification
- Code without tests for new features
- PRs that change too many things at once


## Questions?

Open a discussion — not an issue.
