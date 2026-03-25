## MODIFIED Requirements

### Requirement: API key management
The system SHALL resolve API keys from environment variables using configurable variable names per API provider, including custom OpenAI-compatible providers defined through project-local runtime configuration.

#### Scenario: Environment variable lookup
- **WHEN** a model's API requires authentication
- **THEN** the system SHALL look up the API key from the environment variable configured for that API (e.g., `OPENAI_API_KEY`)

#### Scenario: Custom provider key lookup
- **WHEN** a project-local runtime model is configured for a custom OpenAI-compatible provider
- **THEN** the system SHALL use the configured environment variable binding for that provider's API key when constructing requests

#### Scenario: Missing API key
- **WHEN** the required environment variable is not set
- **THEN** the system SHALL raise a clear error indicating which environment variable is expected
