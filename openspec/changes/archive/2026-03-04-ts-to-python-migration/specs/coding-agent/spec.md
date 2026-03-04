## ADDED Requirements

### Requirement: Session lifecycle management
The system SHALL provide a `SessionManager` that creates, loads, saves, lists, and deletes coding agent sessions. Sessions SHALL be persisted in JSONL format with one entry per line.

#### Scenario: Create new session
- **WHEN** `create_session()` is called
- **THEN** a new session SHALL be created with a unique ID (12-character hex from UUID), and an empty JSONL file SHALL be created in the sessions directory

#### Scenario: Load existing session
- **WHEN** `load_session(session_id)` is called with a valid session ID
- **THEN** the system SHALL read the JSONL file and return a list of `SessionEntry` objects (messages, tool uses, tool results)

#### Scenario: List all sessions
- **WHEN** `list_sessions()` is called
- **THEN** the system SHALL return a list of `SessionSummary` objects with id, title, timestamp, and message count

### Requirement: Agent session orchestration
The system SHALL provide an `AgentSession` class that orchestrates model resolution, tool creation, session management, and the Agent instance lifecycle.

#### Scenario: Start session with model
- **WHEN** `start(model="gpt-4o")` is called
- **THEN** the AgentSession SHALL resolve the model via ModelResolver, create all built-in tools, initialize the Agent, and return the session ID

#### Scenario: Send message and receive events
- **WHEN** `send_message(content)` is called on a started AgentSession
- **THEN** the system SHALL prompt the Agent and yield AgentEvent instances as an async generator

### Requirement: Built-in tool set
The system SHALL provide 7 built-in tools for file system operations and command execution. Each tool SHALL define a JSON Schema for its parameters and return an `AgentToolResult`.

#### Scenario: Bash tool execution
- **WHEN** the bash tool is invoked with `{"command": "echo hello"}`
- **THEN** the tool SHALL execute the command in a subprocess, capture stdout/stderr, and return the output as tool result content

#### Scenario: Read tool execution
- **WHEN** the read tool is invoked with `{"path": "/tmp/test.txt"}`
- **THEN** the tool SHALL read the file contents and return them as tool result content

#### Scenario: Write tool execution
- **WHEN** the write tool is invoked with `{"path": "/tmp/test.txt", "content": "hello"}`
- **THEN** the tool SHALL write the content to the specified file path

#### Scenario: Edit tool execution
- **WHEN** the edit tool is invoked with file path, old text, and new text
- **THEN** the tool SHALL replace the first occurrence of old text with new text in the file

#### Scenario: Grep tool execution
- **WHEN** the grep tool is invoked with `{"pattern": "TODO", "path": "."}`
- **THEN** the tool SHALL search for the pattern in files and return matching lines with file paths and line numbers

#### Scenario: Find tool execution
- **WHEN** the find tool is invoked with `{"pattern": "*.py", "path": "."}`
- **THEN** the tool SHALL find files matching the glob pattern and return the file paths

#### Scenario: Ls tool execution
- **WHEN** the ls tool is invoked with `{"path": "."}`
- **THEN** the tool SHALL list directory contents with file names and types

### Requirement: Model registry and resolution
The system SHALL maintain a `ModelRegistry` of available models and a `ModelResolver` that resolves model identifiers to `Model` objects using the priority: CLI argument > settings default > built-in default.

#### Scenario: Resolve model by ID
- **WHEN** `resolve("gpt-4o")` is called
- **THEN** the resolver SHALL look up the model in the registry and return the corresponding `Model` object with provider, base_url, and cost info

#### Scenario: Resolve with custom model
- **WHEN** a custom model is registered in settings with id "local-llama" and base_url
- **THEN** `resolve("local-llama")` SHALL return a Model with the custom configuration

#### Scenario: Fallback to default
- **WHEN** `resolve(None)` is called with no model argument
- **THEN** the resolver SHALL use `settings.default_model` as the model ID

### Requirement: System prompt generation
The system SHALL generate a system prompt for the coding agent that includes the working directory, available tools description, and any custom instructions from extensions or prompts.

#### Scenario: Default system prompt
- **WHEN** the agent session starts without custom instructions
- **THEN** the system prompt SHALL include the working directory path and descriptions of all available tools

#### Scenario: System prompt with extension prompts
- **WHEN** extensions register additional prompt content
- **THEN** the system prompt SHALL append the extension-provided content

### Requirement: Context compaction
The system SHALL support context compaction strategies to manage message history growth. The `SummaryCompaction` strategy SHALL use the LLM to summarize older messages when the context exceeds a configured threshold.

#### Scenario: Auto-compact on threshold
- **WHEN** `auto_compact` is enabled and the message history exceeds `compact_threshold` tokens
- **THEN** the system SHALL invoke the compaction strategy to summarize older messages, reducing the context size

#### Scenario: No compaction
- **WHEN** `NoCompaction` strategy is configured
- **THEN** the system SHALL never compact messages, allowing the context to grow unbounded

### Requirement: CLI entry point
The system SHALL provide a CLI entry point (`pi` command) that supports print mode (non-interactive, single prompt) for programmatic use.

#### Scenario: Print mode execution
- **WHEN** `pi --print "explain this code"` is executed
- **THEN** the system SHALL send the prompt to the agent, print the response to stdout, and exit
