## ADDED Requirements

### Requirement: Python extension loading
The system SHALL dynamically load Python extension modules from `.pi/extensions/*.py` using `importlib`. Each extension module SHALL export a `create_extension(api: ExtensionAPI) -> None` function.

#### Scenario: Load extension on startup
- **WHEN** the coding agent starts and `.pi/extensions/my_ext.py` exists with a `create_extension` function
- **THEN** the system SHALL load the module and call `create_extension(api)` with an `ExtensionAPI` instance

#### Scenario: Extension load error
- **WHEN** an extension module has a syntax error or raises an exception during loading
- **THEN** the system SHALL log the error and continue loading other extensions without crashing

#### Scenario: No extensions directory
- **WHEN** the `.pi/extensions/` directory does not exist
- **THEN** the system SHALL skip extension loading without error

### Requirement: Extension tool registration
The system SHALL allow extensions to register custom tools via `api.register_tool(name, description, parameters, execute)`. Registered tools SHALL be available to the Agent alongside built-in tools.

#### Scenario: Register custom tool
- **WHEN** an extension calls `api.register_tool(name="my_tool", description="...", parameters={...}, execute=handler)`
- **THEN** the tool SHALL appear in the Agent's tool list and be callable by the LLM

#### Scenario: Tool name conflict
- **WHEN** an extension registers a tool with a name that conflicts with a built-in tool
- **THEN** the system SHALL log a warning and the extension tool SHALL NOT override the built-in tool

### Requirement: Extension event subscription
The system SHALL allow extensions to subscribe to Agent lifecycle events via the ExtensionAPI (e.g., `api.on_message_end(callback)`, `api.on_turn_end(callback)`).

#### Scenario: Subscribe to message_end
- **WHEN** an extension calls `api.on_message_end(callback)` and a message completes
- **THEN** the callback SHALL be invoked with the MessageEndEvent

#### Scenario: Async callback
- **WHEN** an extension registers an async callback
- **THEN** the system SHALL await the callback during event dispatch

### Requirement: Prompt template registration
The system SHALL allow extensions to register prompt templates from `.pi/prompts/*.md` files. Registered prompts SHALL be injectable into the system prompt.

#### Scenario: Register prompt template
- **WHEN** an extension calls `api.register_prompt("review", path=".pi/prompts/review.md")`
- **THEN** the prompt content SHALL be loadable by name and available for inclusion in the system prompt

#### Scenario: Prompt file not found
- **WHEN** `api.register_prompt("missing", path=".pi/prompts/missing.md")` is called with a non-existent file
- **THEN** the system SHALL raise a `FileNotFoundError` at registration time

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

### Requirement: Settings schema validation
The system SHALL validate the loaded settings against a Pydantic model, ensuring all values are of the correct type and within valid ranges.

#### Scenario: Valid settings
- **WHEN** settings.yaml contains valid values for all fields
- **THEN** the system SHALL return a validated `Settings` object

#### Scenario: Invalid field type
- **WHEN** settings.yaml has `max_turns: "fifty"` (string instead of int)
- **THEN** the system SHALL raise a validation error indicating the expected type

#### Scenario: Unknown fields
- **WHEN** settings.yaml contains fields not defined in the schema
- **THEN** the system SHALL ignore unknown fields without error

### Requirement: Settings persistence
The system SHALL support saving updated settings back to `.pi/settings.yaml` in YAML format.

#### Scenario: Save settings
- **WHEN** `settings_manager.save(updated_settings)` is called
- **THEN** the settings SHALL be written to `.pi/settings.yaml` in valid YAML format

#### Scenario: Partial update
- **WHEN** `settings_manager.update(default_model="gpt-4o-mini")` is called
- **THEN** only the specified fields SHALL be updated; all other fields SHALL retain their current values

### Requirement: Custom model configuration in settings
The system SHALL support defining custom models from committed settings and from project-local `.env` runtime configuration, with each custom model including id, provider, and base_url and being registered in the ModelRegistry.

#### Scenario: Custom model from settings
- **WHEN** settings.yaml contains a custom model `{id: "local-llama", provider: "ollama", base_url: "http://localhost:11434/v1"}`
- **THEN** the ModelRegistry SHALL contain a model with id "local-llama" resolvable via ModelResolver

#### Scenario: Custom model from `.env`
- **WHEN** `.env` contains a valid runtime model definition for a custom OpenAI-compatible endpoint
- **THEN** the ModelRegistry SHALL contain a custom model generated from that `.env` configuration and resolvable via ModelResolver

### Requirement: Extension unloading by source
The system SHALL support unloading all tools and event subscriptions registered by a specific extension, identified by its source path.

#### Scenario: Unload extension
- **WHEN** extension "my_ext" is unloaded
- **THEN** all tools registered by "my_ext" SHALL be removed from the Agent, and all event subscriptions from "my_ext" SHALL be cancelled
