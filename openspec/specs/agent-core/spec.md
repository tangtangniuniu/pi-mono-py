## ADDED Requirements

### Requirement: Stateful Agent with message history
The system SHALL provide an `Agent` class that maintains conversation state including system prompt, model selection, thinking level, available tools, and an ordered list of messages. The message list SHALL be immutable — appending messages SHALL create a new list rather than mutating the existing one.

#### Scenario: Create agent and send prompt
- **WHEN** an Agent is created and `prompt(input)` is called with a user message
- **THEN** the Agent SHALL add the user message to its state, invoke the LLM, and append the assistant response to its message history

#### Scenario: Immutable message history
- **WHEN** a message is appended to the Agent state
- **THEN** a new list SHALL be created containing all previous messages plus the new one; the original list reference SHALL remain unchanged

### Requirement: Agent loop with tool execution
The system SHALL implement an `AgentLoop` that processes LLM responses in a loop: when the LLM returns tool calls, the loop SHALL execute each tool, collect results as `ToolResultMessage`, and feed them back to the LLM until no more tool calls are made.

#### Scenario: Single tool call cycle
- **WHEN** the LLM responds with a tool call for "bash" with arguments `{"command": "ls"}`
- **THEN** the loop SHALL execute the bash tool, collect the result, send a ToolResultMessage back to the LLM, and continue until the LLM responds with text only

#### Scenario: Multiple tool calls in one response
- **WHEN** the LLM responds with multiple tool calls in a single message
- **THEN** the loop SHALL execute all tools (potentially concurrently) and send all results back in one round

#### Scenario: Tool execution error
- **WHEN** a tool execution raises an exception
- **THEN** the loop SHALL create a ToolResultMessage with `is_error=True` and the error description, and continue the conversation

### Requirement: Event-driven architecture
The system SHALL emit typed `AgentEvent` instances during the agent loop lifecycle. Events SHALL include: `agent_start`, `agent_end`, `turn_start`, `turn_end`, `message_start`, `message_update`, `message_end`, `tool_execution_start`, `tool_execution_update`, `tool_execution_end`.

#### Scenario: Subscribe to events
- **WHEN** a callback is registered via `agent.subscribe(callback)`
- **THEN** the callback SHALL be invoked for every AgentEvent emitted during agent execution

#### Scenario: Event ordering
- **WHEN** the agent processes a prompt that triggers a tool call
- **THEN** events SHALL be emitted in order: agent_start → turn_start → message_start → message_update* → message_end → tool_execution_start → tool_execution_update* → tool_execution_end → turn_end → agent_end

### Requirement: Steering and follow-up message queues
The system SHALL support two message queues: a steering queue (high priority, interrupts current processing) and a follow-up queue (processed after current turn completes).

#### Scenario: Steering message
- **WHEN** `agent.steer(message)` is called during an active agent loop
- **THEN** the steering message SHALL be processed at the next loop iteration, before any follow-up messages

#### Scenario: Follow-up message
- **WHEN** `agent.follow_up(message)` is called during an active agent loop
- **THEN** the follow-up message SHALL be processed after the current turn completes

### Requirement: Abort mechanism
The system SHALL support aborting an in-progress agent loop via `agent.abort()`, which sets an `asyncio.Event` that propagates to the LLM stream and tool executions.

#### Scenario: Abort during LLM streaming
- **WHEN** `agent.abort()` is called while the LLM is streaming a response
- **THEN** the stream SHALL be cancelled, and an `agent_end` event SHALL be emitted

#### Scenario: Abort during tool execution
- **WHEN** `agent.abort()` is called while a tool is executing
- **THEN** the tool execution SHALL be cancelled (if it supports the abort signal), and the agent loop SHALL terminate

### Requirement: Context transformation hook
The system SHALL support a `transform_context` callback in `AgentLoopConfig` that allows modifying the message list before sending to the LLM (e.g., for context trimming or injection).

#### Scenario: Context transformation
- **WHEN** `transform_context` is configured and the agent sends messages to the LLM
- **THEN** the callback SHALL be invoked with the current message list, and the LLM SHALL receive the transformed result

### Requirement: Configurable LLM conversion
The system SHALL support a `convert_to_llm` callback in `AgentLoopConfig` that converts `AgentMessage` instances to LLM-compatible `Message` instances, enabling custom message type extensions.

#### Scenario: Custom message conversion
- **WHEN** the agent has custom `AgentMessage` types and `convert_to_llm` is configured
- **THEN** all messages SHALL be converted through this callback before being sent to the LLM provider
