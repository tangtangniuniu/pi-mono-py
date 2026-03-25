## Why

The repository currently documents model configuration in checked-in settings-oriented flows, which makes local setup for a custom OpenAI-compatible endpoint awkward and increases the risk of committing real credentials. We need a supported `.env`-based workflow so contributors can run the project against a local proxy model safely and predictably.

## What Changes

- Load runtime LLM connection settings from a local `.env` file for the project workflow, including provider name, base URL, model ID, and API key.
- Keep `.env` out of version control and provide `.env.example` with intentionally non-working placeholder values for user onboarding.
- Add a documented validation step that uses `curl` against the configured OpenAI-compatible endpoint before relying on the app runtime.
- Update model/settings resolution so the coding agent can use the `.env`-supplied model configuration without requiring committed secrets in repository files.
- Ensure error handling clearly distinguishes invalid `.env` values from missing credentials or unreachable endpoints.

## Capabilities

### New Capabilities
- `dotenv-runtime-config`: Define and validate project-local `.env` configuration for runtime model selection and OpenAI-compatible connection settings.

### Modified Capabilities
- `pi-extensions`: Extend configuration loading so project runtime settings can be sourced from `.env` in addition to `.pi/settings.yaml` and environment variables.
- `coding-agent`: Update model resolution behavior so the default runtime model can be supplied from `.env` without editing committed settings files.
- `ai-openai-compat`: Clarify how API key, base URL, and model configuration work for custom OpenAI-compatible providers used through local proxy endpoints.

## Impact

Affected areas include `src/pi_mono/config/`, `src/pi_mono/coding_agent/`, and the OpenAI-compatible provider path in `src/pi_mono/ai/`. User-facing docs and local setup files will change, especially `.env.example`, README guidance, and ignore rules verification for `.env`.
