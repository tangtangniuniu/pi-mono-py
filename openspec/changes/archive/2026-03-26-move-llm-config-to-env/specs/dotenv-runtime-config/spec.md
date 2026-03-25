## ADDED Requirements

### Requirement: Project-local dotenv runtime model configuration
The system SHALL support a project-local `.env` file that defines the runtime OpenAI-compatible model configuration used by the coding agent, including provider name, base URL, model identifier, and API key reference.

#### Scenario: Load runtime model from `.env`
- **WHEN** the project contains a `.env` file with valid runtime model fields
- **THEN** the system SHALL synthesize a custom model configuration from those values and make it available for model resolution

#### Scenario: Missing `.env` file
- **WHEN** the project does not contain a `.env` file
- **THEN** the system SHALL fall back to existing committed settings and explicit environment variables without error

### Requirement: Safe dotenv onboarding template
The repository SHALL provide a checked-in `.env.example` template containing the required runtime model keys with intentionally non-working placeholder values.

#### Scenario: Example file for onboarding
- **WHEN** a contributor clones the repository
- **THEN** they SHALL be able to copy `.env.example` to `.env` and replace placeholders with local values to configure the runtime model

#### Scenario: Placeholder values are non-production
- **WHEN** `.env.example` is committed
- **THEN** the file SHALL contain dummy model and key values that are unsuitable for real authenticated requests

