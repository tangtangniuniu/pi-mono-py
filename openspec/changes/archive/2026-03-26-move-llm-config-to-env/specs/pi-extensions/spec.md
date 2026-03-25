## MODIFIED Requirements

### Requirement: Settings YAML loading
The system SHALL load configuration from project-local `.env`, `.pi/settings.yaml` (project-level), and `~/.pi/settings.yaml` (user-level), with explicit environment variables taking highest priority, `.env` values taking priority over YAML files, and project-level YAML taking priority over user-level YAML.

#### Scenario: Load project settings
- **WHEN** `.pi/settings.yaml` exists with `default_model: "gpt-4o"` and no higher-priority overrides are present
- **THEN** `settings.default_model` SHALL be `"gpt-4o"`

#### Scenario: Load project-local dotenv values
- **WHEN** `.env` defines the runtime model configuration and `.pi/settings.yaml` does not override it through explicit environment variables
- **THEN** the loader SHALL incorporate the `.env` values into the effective project runtime configuration

#### Scenario: Environment variable override
- **WHEN** `.env` or `.pi/settings.yaml` sets a default model and environment variable `PI_DEFAULT_MODEL=gpt-4o-mini` is set
- **THEN** `settings.default_model` SHALL be `"gpt-4o-mini"`

#### Scenario: User-level fallback
- **WHEN** neither `.env` nor `.pi/settings.yaml` specifies the needed setting but `~/.pi/settings.yaml` does
- **THEN** the system SHALL use the user-level value

#### Scenario: Missing settings files
- **WHEN** `.env`, `.pi/settings.yaml`, and `~/.pi/settings.yaml` do not exist
- **THEN** the system SHALL use built-in default values for all settings

#### Scenario: Invalid dotenv or YAML
- **WHEN** `.env` contains malformed runtime model values or `.pi/settings.yaml` contains invalid YAML syntax
- **THEN** the system SHALL raise a clear error identifying the invalid source and the parse or validation failure

### Requirement: Custom model configuration in settings
The system SHALL support defining custom models from committed settings and from project-local `.env` runtime configuration, with each custom model including id, provider, and base_url and being registered in the ModelRegistry.

#### Scenario: Custom model from settings
- **WHEN** settings.yaml contains a custom model `{id: "local-llama", provider: "ollama", base_url: "http://localhost:11434/v1"}`
- **THEN** the ModelRegistry SHALL contain a model with id "local-llama" resolvable via ModelResolver

#### Scenario: Custom model from `.env`
- **WHEN** `.env` contains a valid runtime model definition for a custom OpenAI-compatible endpoint
- **THEN** the ModelRegistry SHALL contain a custom model generated from that `.env` configuration and resolvable via ModelResolver

