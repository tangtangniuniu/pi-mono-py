## 1. Dotenv Configuration Plumbing

- [x] 1.1 Add project-local `.env` loading to the configuration path while preserving explicit `PI_*` environment variables as the highest-priority overrides.
- [x] 1.2 Define and validate the `.env` keys required for the runtime OpenAI-compatible model configuration.
- [x] 1.3 Synthesize a custom model entry from `.env` values and merge it into the effective settings/model configuration.

## 2. Runtime Model Resolution

- [x] 2.1 Update coding-agent model registry/bootstrap logic to register the `.env`-derived custom model.
- [x] 2.2 Update model resolution rules so the agent prefers the `.env` runtime model when no CLI model is provided.
- [x] 2.3 Ensure OpenAI-compatible provider code resolves the configured API key binding and reports clear errors for missing or invalid credentials.

## 3. Repository Safety and Developer Workflow

- [x] 3.1 Add or update `.env.example` with intentionally invalid placeholder provider, model, base URL, and key values.
- [x] 3.2 Verify `.env` remains ignored by git and that no committed config file needs real secrets for local runtime setup.
- [x] 3.3 Document the local setup flow, including copying `.env.example`, filling local values, and the intended precedence versus `.pi/settings.yaml`.

## 4. Validation and Tests

- [x] 4.1 Add unit tests for `.env` parsing, precedence ordering, and invalid configuration handling in the config loader.
- [x] 4.2 Add tests for registration and resolution of the `.env`-derived custom model in the coding-agent path.
- [x] 4.3 Add tests for custom OpenAI-compatible API key lookup behavior.
- [x] 4.4 Document and execute a `curl` smoke test against the configured endpoint using `.env` values, confirming the local proxy model is reachable before app runtime verification.
