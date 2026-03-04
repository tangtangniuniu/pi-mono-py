## ADDED Requirements

### Requirement: Session CRUD via REST API
The system SHALL provide REST endpoints for creating, listing, retrieving, and deleting agent sessions.

#### Scenario: Create session
- **WHEN** `POST /api/sessions` is called with optional `{"model": "gpt-4o"}`
- **THEN** the server SHALL create a new AgentSession, return HTTP 201 with `{"session_id": "<id>", "model": "<resolved_model_id>"}`

#### Scenario: List sessions
- **WHEN** `GET /api/sessions` is called
- **THEN** the server SHALL return HTTP 200 with a JSON array of session summaries (id, title, created_at, message_count)

#### Scenario: Get session details
- **WHEN** `GET /api/sessions/{id}` is called with a valid session ID
- **THEN** the server SHALL return HTTP 200 with session state (model, thinking_level, is_streaming, message_count)

#### Scenario: Delete session
- **WHEN** `DELETE /api/sessions/{id}` is called with a valid session ID
- **THEN** the server SHALL clean up the AgentSession, remove it from the registry, and return HTTP 204

#### Scenario: Session not found
- **WHEN** any session endpoint is called with a non-existent session ID
- **THEN** the server SHALL return HTTP 404 with `{"error": "Session not found"}`

### Requirement: Message sending via REST API
The system SHALL provide a REST endpoint for sending messages to a session's agent. The response SHALL be immediate (HTTP 202) while processing occurs asynchronously. Results SHALL be delivered via the SSE event stream.

#### Scenario: Send message
- **WHEN** `POST /api/sessions/{id}/messages` is called with `{"content": "hello"}`
- **THEN** the server SHALL return HTTP 202 (Accepted), and the agent SHALL begin processing the message asynchronously

#### Scenario: Send message while busy
- **WHEN** a message is sent while the agent is already processing
- **THEN** the server SHALL return HTTP 409 (Conflict) with `{"error": "Agent is busy"}`

### Requirement: Steering message via REST API
The system SHALL provide a REST endpoint for sending steering messages that interrupt the current agent processing.

#### Scenario: Send steering message
- **WHEN** `POST /api/sessions/{id}/steer` is called with `{"content": "stop and do this instead"}`
- **THEN** the server SHALL enqueue the steering message and return HTTP 202

### Requirement: Abort via REST API
The system SHALL provide a REST endpoint for aborting the current agent processing.

#### Scenario: Abort active processing
- **WHEN** `POST /api/sessions/{id}/abort` is called while the agent is streaming
- **THEN** the server SHALL trigger `agent.abort()` and return HTTP 200

#### Scenario: Abort when idle
- **WHEN** `POST /api/sessions/{id}/abort` is called while the agent is idle
- **THEN** the server SHALL return HTTP 200 (idempotent, no error)

### Requirement: Message history via REST API
The system SHALL provide a REST endpoint for retrieving the message history of a session.

#### Scenario: Get message history
- **WHEN** `GET /api/sessions/{id}/messages` is called
- **THEN** the server SHALL return HTTP 200 with the ordered list of messages (user, assistant, tool_result) in the session

### Requirement: SSE event stream
The system SHALL provide an SSE endpoint that streams all AgentEvent instances for a given session in real-time. Events SHALL use the SSE standard format with `event:` type and `data:` JSON payload.

#### Scenario: Connect to event stream
- **WHEN** `GET /api/sessions/{id}/events` is called with `Accept: text/event-stream`
- **THEN** the server SHALL establish an SSE connection and begin streaming events

#### Scenario: Receive agent events
- **WHEN** the agent processes a message and emits events
- **THEN** the SSE stream SHALL deliver each event with `event: <type>` and `data: <json>` format

#### Scenario: SSE reconnection
- **WHEN** the SSE connection is dropped and the client reconnects
- **THEN** the server SHALL resume event delivery from the current state (missed events are not replayed)

#### Scenario: SSE heartbeat
- **WHEN** the SSE connection is idle (no agent activity)
- **THEN** the server SHALL send periodic heartbeat comments (`:heartbeat`) to keep the connection alive

### Requirement: Model management via REST API
The system SHALL provide REST endpoints for listing available models and switching a session's model.

#### Scenario: List models
- **WHEN** `GET /api/models` is called
- **THEN** the server SHALL return HTTP 200 with a list of available models (id, provider, display_name)

#### Scenario: Switch session model
- **WHEN** `PUT /api/sessions/{id}/model` is called with `{"model_id": "gpt-4o-mini"}`
- **THEN** the server SHALL update the session's model and return HTTP 200

### Requirement: Thinking level control via REST API
The system SHALL provide a REST endpoint for setting the thinking level of a session's agent.

#### Scenario: Set thinking level
- **WHEN** `PUT /api/sessions/{id}/thinking` is called with `{"level": "medium"}`
- **THEN** the server SHALL update the agent's thinking level and return HTTP 200

### Requirement: Settings management via REST API
The system SHALL provide REST endpoints for reading and updating global settings.

#### Scenario: Get settings
- **WHEN** `GET /api/settings` is called
- **THEN** the server SHALL return HTTP 200 with the current settings (excluding sensitive values like API keys)

#### Scenario: Update settings
- **WHEN** `PUT /api/settings` is called with partial settings JSON
- **THEN** the server SHALL merge the updates into existing settings, persist to settings.yaml, and return HTTP 200

### Requirement: Health check endpoint
The system SHALL provide a health check endpoint for monitoring.

#### Scenario: Health check
- **WHEN** `GET /api/health` is called
- **THEN** the server SHALL return HTTP 200 with `{"status": "ok", "version": "<version>"}`

### Requirement: CORS configuration
The system SHALL support configurable CORS origins to allow web-ui from different origins to access the API.

#### Scenario: Allowed origin
- **WHEN** a request comes from an origin listed in `settings.server.cors_origins`
- **THEN** the server SHALL include appropriate CORS headers in the response

#### Scenario: Disallowed origin
- **WHEN** a request comes from an origin not in the allowed list
- **THEN** the server SHALL reject the preflight request

### Requirement: Server startup
The system SHALL provide a CLI entry point to start the REST server with configurable host, port, and settings path.

#### Scenario: Start server
- **WHEN** `pi-server --host 0.0.0.0 --port 8080` is executed
- **THEN** the server SHALL start a uvicorn instance listening on the specified address and load settings from `.pi/settings.yaml`

#### Scenario: Default startup
- **WHEN** `pi-server` is executed without arguments
- **THEN** the server SHALL start on `127.0.0.1:8080` using default settings
