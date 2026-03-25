## MODIFIED Requirements

### Requirement: Model registry and resolution
The system SHALL maintain a `ModelRegistry` of available models and a `ModelResolver` that resolves model identifiers to `Model` objects using the priority: CLI argument > settings default > `.env`-supplied runtime custom model > built-in default.

#### Scenario: Resolve model by ID
- **WHEN** `resolve("gpt-4o")` is called
- **THEN** the resolver SHALL look up the model in the registry and return the corresponding `Model` object with provider, base_url, and cost info

#### Scenario: Resolve with custom model
- **WHEN** a custom model is registered in settings with id "local-llama" and base_url
- **THEN** `resolve("local-llama")` SHALL return a Model with the custom configuration

#### Scenario: Resolve with `.env` runtime model
- **WHEN** no CLI model argument is provided and the project `.env` defines a runtime custom model
- **THEN** the resolver SHALL select that `.env`-supplied model for the coding agent session

#### Scenario: Fallback to default
- **WHEN** `resolve(None)` is called with no model argument and no `.env` runtime model is configured
- **THEN** the resolver SHALL use `settings.default_model` and then the built-in default if needed

