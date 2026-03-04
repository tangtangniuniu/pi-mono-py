# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python implementation of the [pi-mono](https://github.com/badlogic/pi-mono) TypeScript monorepo. Provides a unified LLM API, agent runtime, and CLI coding agent. The original TypeScript source lives in `pi-mono/` for reference.

## Commands

```bash
# Install dependencies (uses uv)
uv sync --dev

# Run tests
uv run pytest
uv run pytest tests/ai/test_types.py              # single file
uv run pytest tests/ai/test_types.py::test_name    # single test
uv run pytest -m integration                       # only integration tests
uv run pytest --cov=pi_mono --cov-report=term      # with coverage

# Lint and format
uv run ruff check src/ tests/                      # lint
uv run ruff check --fix src/ tests/                # lint + autofix
uv run ruff format src/ tests/                     # format

# Type check
uv run mypy src/

# Run the CLI coding agent
uv run pi

# Run the REST API server
uv run pi-server
```

## Architecture

Four layers, each depending only on the layer(s) below it:

```
coding_agent  ─┐
server        ─┤──▶  agent  ──▶  ai
               │
config ────────┘
```

### `src/pi_mono/ai/` — Unified LLM API
Provider-agnostic streaming interface for 20+ LLM providers. Core abstractions:
- `types.py` — All data types (`Model`, `Content` variants, `Message` types, `Usage`, `Cost`, streaming events). All frozen dataclasses.
- `providers/base.py` — `LLMProvider` protocol: `stream()` and `stream_simple()` returning `AsyncIterator[AssistantMessageEvent]`.
- `providers/openai_compat.py` — OpenAI-compatible provider implementation.
- `models.py` — Global model registry with cost calculation.
- `api_registry.py` — Provider registration/lookup by API identifier.
- Streaming event pattern: `StartEvent → TextDeltaEvent* → TextEndEvent → DoneEvent`.

### `src/pi_mono/agent/` — Agent Runtime
Stateful agentic loop with tool execution:
- `agent.py` — `Agent` class: state management, message queuing, event emission.
- `agent_loop.py` — Core loop: LLM call → parse tool calls → execute tools → repeat.
- `types.py` — `AgentTool` (JSON Schema params + async executor), `AgentContext`, `AgentLoopConfig`, event discriminated union via `Literal` type fields.

### `src/pi_mono/coding_agent/` — CLI Coding Agent
Interactive terminal coding assistant:
- `main.py` — Entry point, session setup, mode selection.
- `cli/args.py` — Click CLI argument parsing.
- `core/agent_session.py` — Session lifecycle, tool/extension loading.
- `core/tools/` — Built-in tools: `bash`, `read`, `write`, `edit`, `grep`, `find`, `ls`.
- `modes/interactive/` — Rich-based terminal UI.
- `modes/print_mode.py` — Non-interactive output mode.
- `core/extensions/` — Loads `.py` extensions from `.pi/extensions/`.

### `src/pi_mono/server/` — REST API Server
FastAPI server exposing the agent over HTTP:
- `app.py` — App factory, CORS, route registration.
- `routes/` — Endpoints for sessions, messages (SSE streaming), models, settings.
- `session_registry.py` — In-memory session management.

### `src/pi_mono/config/` — Configuration
- `settings.py` — Pydantic settings models (`ServerConfig`, `ModelsConfig`, `ExtensionsConfig`).
- `loader.py` — YAML config loading from `.pi/settings.yaml`.

## Key Patterns

- **Immutability**: Core data types use `@dataclass(frozen=True)`. Create new objects rather than mutating.
- **Async-first**: All tools, providers, and the agent loop are async. Tests use `asyncio_mode = "auto"`.
- **Provider protocol**: LLM providers implement `LLMProvider` (runtime-checkable Protocol) in `providers/base.py`.
- **Event-driven streaming**: Both LLM responses and agent execution emit typed event streams.
- **Discriminated unions**: Agent events use `Literal` type fields for discrimination (e.g., `type: Literal["turn_start"]`).

## Test Infrastructure

- Tests mirror source layout under `tests/`.
- `tests/conftest.py` provides: `MockLLMProvider`, `make_model()`, `make_assistant_message()`, `make_usage()`, `tmp_project_dir` (with `.pi/` structure), `mock_tool`.
- Coverage minimum: 80% (enforced in `pyproject.toml`).

## Configuration

- **Ruff**: line-length 120, double quotes, 4-space indent. See `ruff.toml` for enabled rule sets.
- **MyPy**: strict mode, `mypy_path = src`. See `mypy.ini`.
- **Python**: >= 3.11, targets 3.12/3.13.

## TypeScript Reference

The `pi-mono/` directory contains the original TypeScript monorepo. Module mapping:
- `packages/ai/` → `src/pi_mono/ai/`
- `packages/agent/` → `src/pi_mono/agent/`
- `packages/coding-agent/` → `src/pi_mono/coding_agent/`
- `packages/web-ui/` → `pi-mono/packages/web-ui/` (stays TypeScript, talks to `pi-server`)

## OpenSpec

Specification-driven development tracking in `openspec/`. Specs in `openspec/specs/`, archived changes in `openspec/changes/archive/`. Use `/opsx:*` commands to manage changes.
