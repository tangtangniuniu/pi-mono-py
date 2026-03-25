## Context

The current project loads committed configuration from `.pi/settings.yaml` plus `PI_*` environment overrides. That works for stable, non-secret settings, but it is a poor fit for local OpenAI-compatible model endpoints that require base URL, model ID, provider alias, and API key values that must not be committed. This change crosses configuration loading, model registration, coding-agent model resolution, documentation, and local validation workflow, so it benefits from an explicit design before implementation.

## Goals / Non-Goals

**Goals:**
- Support project-local `.env` as the source for runtime LLM connection values.
- Keep real credentials in untracked `.env` while providing a safe `.env.example`.
- Make the coding agent resolve a custom OpenAI-compatible model from `.env` without editing committed settings files.
- Add a repeatable `curl` validation path that confirms the configured endpoint is reachable before app use.

**Non-Goals:**
- Replacing `.pi/settings.yaml` for all configuration.
- Supporting multiple `.env` model profiles in the first iteration.
- Committing working secrets or auto-testing the user’s private endpoint inside CI.

## Decisions

Load `.env` during settings initialization, before applying existing `PI_*` overrides, so `.env` becomes the project-local baseline and explicit environment variables still remain highest priority. This preserves the current precedence model while adding a safer developer workflow.

Represent the runtime model as a derived custom model entry built from a fixed set of `.env` keys such as provider name, base URL, model ID, and API key env binding. This avoids scattering endpoint-specific logic across the CLI and keeps model registration aligned with the existing `ModelsConfig.custom` path.

Keep `.env.example` intentionally invalid, with a dummy model and placeholder key, to document required fields without enabling accidental production use. Alternative considered: storing example values in README only. Rejected because contributors benefit from a copyable template that matches the loader exactly.

Document `curl` validation as a manual preflight step rather than performing network checks in app startup. Alternative considered: eager connectivity testing during boot. Rejected because startup should not fail on optional external validation, and test environments may not have network access.

## Risks / Trade-offs

[Incorrect precedence between `.env` and `PI_*`] -> Preserve and test the order `.env` < explicit environment variables.

[Leaking secrets through examples or logs] -> Keep `.env` gitignored, use placeholder values in `.env.example`, and avoid echoing the configured key in error messages.

[Custom endpoint config diverges from existing settings schema] -> Reuse `ModelsConfig.custom`-compatible data structures and add focused tests around model registration and resolution.

[User config is syntactically valid but endpoint is unusable] -> Provide a documented `curl` smoke test and clear runtime errors for unreachable or unauthorized endpoints.
