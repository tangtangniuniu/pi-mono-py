# Repository Guidelines

## Project Structure & Module Organization

Primary Python code lives in `src/pi_mono/` with five main packages: `ai/` for provider integrations and model types, `agent/` for the runtime loop, `coding_agent/` for the CLI, `server/` for FastAPI routes, and `config/` for settings loading. Tests mirror that layout under `tests/` (`tests/ai/`, `tests/agent/`, `tests/server/`, etc.). The `pi-mono/` directory is the upstream TypeScript reference implementation; use it for behavior parity, not as the place for Python changes. Specification notes live in `openspec/`.

## Build, Test, and Development Commands

Use `uv` for all local work:

```bash
uv sync --dev
uv run pytest
uv run pytest tests/ai/test_types.py
uv run pytest -m integration
uv run pytest --cov=pi_mono --cov-report=term
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
uv run pi
uv run pi-server
```

`uv sync --dev` installs runtime and dev dependencies. Run `pi` for the CLI assistant and `pi-server` for the HTTP/SSE API.

## Coding Style & Naming Conventions

Target Python 3.11+ with 4-space indentation, double quotes, and a 120-character line limit. Ruff handles formatting, import sorting, and linting; MyPy runs in strict mode. Follow existing naming: `snake_case` for modules/functions, `PascalCase` for classes, and `test_*.py` for test files. Prefer frozen dataclasses and `Literal`-tagged event types where those patterns already exist.

## Testing Guidelines

Pytest is the test framework, with `pytest-asyncio` enabled via `asyncio_mode = auto`. Keep tests adjacent to the matching package path and mirror filenames when practical, for example `src/pi_mono/agent/agent.py` -> `tests/agent/test_agent.py`. Coverage is enforced at 80%, so add or update tests with behavior changes.

## Commit & Pull Request Guidelines

Recent commits use short, direct subjects such as `add readme` and `openspec ts 2 python 90%`. Keep commit titles brief, imperative, and focused on one change; English is preferred for consistency. Pull requests should describe the user-visible impact, list validation commands run, link related specs/issues, and include screenshots or terminal captures when changing CLI or server behavior.

## Configuration & Reference Notes

Project settings load from `.pi/settings.yaml`; keep secrets out of the repository. When porting behavior, compare against `pi-mono/packages/*` and document intentional deviations in code comments or `openspec`.


## rules
always use chinese
