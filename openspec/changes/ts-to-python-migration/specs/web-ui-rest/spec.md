## ADDED Requirements

### Requirement: REST Agent client
The web-ui SHALL provide a `RestAgentClient` class that implements the same interface as the in-process `Agent` (prompt, steer, abort, subscribe) but communicates with the Python backend via REST API and SSE.

#### Scenario: Send prompt via REST
- **WHEN** `restClient.prompt("hello")` is called
- **THEN** the client SHALL send `POST /api/sessions/{id}/messages` with `{"content": "hello"}` to the backend

#### Scenario: Subscribe to events via SSE
- **WHEN** `restClient.subscribe(callback)` is called
- **THEN** the client SHALL establish an SSE connection to `GET /api/sessions/{id}/events` and invoke the callback for each received event

#### Scenario: Abort via REST
- **WHEN** `restClient.abort()` is called
- **THEN** the client SHALL send `POST /api/sessions/{id}/abort` to the backend

### Requirement: Session management via REST
The web-ui SHALL manage sessions through the REST API instead of local IndexedDB storage for session data.

#### Scenario: Create session
- **WHEN** the user starts a new conversation
- **THEN** the web-ui SHALL call `POST /api/sessions` and use the returned session_id for subsequent interactions

#### Scenario: List sessions
- **WHEN** the user opens the session list
- **THEN** the web-ui SHALL call `GET /api/sessions` and display the returned session summaries

#### Scenario: Load session history
- **WHEN** the user selects an existing session
- **THEN** the web-ui SHALL call `GET /api/sessions/{id}/messages` to load the message history

#### Scenario: Delete session
- **WHEN** the user deletes a session
- **THEN** the web-ui SHALL call `DELETE /api/sessions/{id}`

### Requirement: AgentInterface REST integration
The `AgentInterface` web component SHALL accept a `RestAgentClient` instance instead of a direct `Agent` object. The component's rendering and event handling logic SHALL remain unchanged.

#### Scenario: Render streaming response
- **WHEN** the SSE stream delivers `message_update` events
- **THEN** the `AgentInterface` SHALL render the streaming text incrementally, identical to the direct-Agent behavior

#### Scenario: Display tool execution
- **WHEN** the SSE stream delivers `tool_execution_start` and `tool_execution_end` events
- **THEN** the `AgentInterface` SHALL display tool execution status and results

### Requirement: Model selection via REST
The web-ui model selector SHALL fetch available models from and switch models through the REST API.

#### Scenario: Fetch model list
- **WHEN** the model selector opens
- **THEN** the web-ui SHALL call `GET /api/models` and display the available models

#### Scenario: Switch model
- **WHEN** the user selects a different model
- **THEN** the web-ui SHALL call `PUT /api/sessions/{id}/model` with the selected model ID

### Requirement: Settings management via REST
The web-ui settings dialog SHALL read and write settings through the REST API.

#### Scenario: Load settings
- **WHEN** the settings dialog opens
- **THEN** the web-ui SHALL call `GET /api/settings` and populate the form

#### Scenario: Save settings
- **WHEN** the user saves settings changes
- **THEN** the web-ui SHALL call `PUT /api/settings` with the updated values

### Requirement: SSE connection management
The web-ui SHALL manage the SSE connection lifecycle: auto-connect when a session is active, reconnect on disconnect, and disconnect when the session is closed.

#### Scenario: Auto-reconnect
- **WHEN** the SSE connection drops unexpectedly
- **THEN** the web-ui SHALL attempt to reconnect with exponential backoff (1s, 2s, 4s, max 30s)

#### Scenario: Clean disconnect
- **WHEN** the user navigates away from a session
- **THEN** the web-ui SHALL close the SSE connection

### Requirement: Preserve existing UI components
The web-ui SHALL retain all existing UI components (ChatPanel, MessageList, StreamingMessageContainer, ThinkingBlock, ConsoleBlock, artifacts renderers) without modification. Only the data layer (Agent interaction) SHALL be replaced.

#### Scenario: Unchanged rendering
- **WHEN** the web-ui receives events through RestAgentClient
- **THEN** all UI components SHALL render identically to when using a direct Agent instance
