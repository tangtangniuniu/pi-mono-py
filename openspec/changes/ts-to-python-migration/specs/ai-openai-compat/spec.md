## ADDED Requirements

### Requirement: OpenAI-compatible streaming API
The system SHALL provide a unified streaming interface for LLM interactions using the OpenAI Chat Completions API protocol. The `stream()` function SHALL accept a `Model`, `Context` (messages + system_prompt + tools), and `StreamOptions`, returning an `AssistantMessageEventStream` that emits typed events (StartEvent, TextDeltaEvent, ThinkingDeltaEvent, ToolCallStartEvent, DoneEvent, ErrorEvent).

#### Scenario: Streaming text response
- **WHEN** `stream()` is called with a valid model, context containing user messages, and options
- **THEN** the returned event stream SHALL emit StartEvent, followed by one or more TextDeltaEvent with incremental text, followed by TextEndEvent and DoneEvent

#### Scenario: Streaming with tool calls
- **WHEN** the LLM response contains tool calls
- **THEN** the event stream SHALL emit ToolCallStartEvent (with tool name), ToolCallDeltaEvent (with argument fragments), and ToolCallEndEvent (with parsed arguments dict)

#### Scenario: Stream error handling
- **WHEN** the LLM API returns an error during streaming
- **THEN** the event stream SHALL emit an ErrorEvent containing the error message, and the stream SHALL terminate

### Requirement: Non-streaming completion API
The system SHALL provide a `complete()` function that returns a fully assembled `AssistantMessage` by consuming the stream internally.

#### Scenario: Successful completion
- **WHEN** `complete()` is called with a valid model and context
- **THEN** the system SHALL return an `AssistantMessage` containing all content blocks (text, thinking, tool calls), usage statistics, and cost information

### Requirement: Configurable base URL
The system SHALL allow users to configure the API base URL per model, enabling connection to any OpenAI-compatible service (OpenAI, Ollama, vLLM, LM Studio, etc.).

#### Scenario: Custom base URL
- **WHEN** a Model is configured with `base_url: "http://localhost:11434/v1"`
- **THEN** all API requests for that model SHALL be sent to the specified base URL

#### Scenario: Default base URL
- **WHEN** no base_url is specified on the Model
- **THEN** the system SHALL use `https://api.openai.com/v1` as the default

### Requirement: Model compatibility detection
The system SHALL detect and adapt to differences between OpenAI-compatible services using a `ModelCompat` configuration. This SHALL handle variations in supported features (developer role, reasoning effort, tool result naming, streaming options).

#### Scenario: Provider-specific compat
- **WHEN** a model has provider "mistral"
- **THEN** the system SHALL apply Mistral-specific compatibility rules (e.g., tool ID normalization, no developer role)

#### Scenario: Unknown provider defaults
- **WHEN** a model has an unrecognized provider
- **THEN** the system SHALL apply safe defaults that work with standard OpenAI Chat Completions API

### Requirement: Cost tracking
The system SHALL calculate token costs for each LLM interaction based on the model's cost configuration (input, output, cache_read, cache_write rates per million tokens).

#### Scenario: Cost calculation
- **WHEN** an LLM response completes with usage data (input_tokens=1000, output_tokens=500)
- **THEN** the system SHALL return a `Cost` object with accurate dollar amounts calculated from the model's per-million-token rates

### Requirement: API key management
The system SHALL resolve API keys from environment variables using configurable variable names per API provider.

#### Scenario: Environment variable lookup
- **WHEN** a model's API requires authentication
- **THEN** the system SHALL look up the API key from the environment variable configured for that API (e.g., `OPENAI_API_KEY`)

#### Scenario: Missing API key
- **WHEN** the required environment variable is not set
- **THEN** the system SHALL raise a clear error indicating which environment variable is expected

### Requirement: Provider registration
The system SHALL maintain a global provider registry where LLM providers can be registered and looked up by API name. Registration SHALL support source_id for batch unregistration.

#### Scenario: Register and lookup
- **WHEN** a provider is registered with api name "openai"
- **THEN** `get_api_provider("openai")` SHALL return that provider instance

#### Scenario: Batch unregister by source
- **WHEN** providers are registered with `source_id="extension-a"` and `unregister_api_providers("extension-a")` is called
- **THEN** all providers registered with that source_id SHALL be removed

### Requirement: Abort support
The system SHALL support aborting an in-progress stream via an `asyncio.Event` passed in `StreamOptions.abort_event`.

#### Scenario: Abort during streaming
- **WHEN** `abort_event.set()` is called while streaming
- **THEN** the stream SHALL terminate and emit an ErrorEvent or DoneEvent with abort indication
