/**
 * REST API client for the pi-mono Python backend.
 *
 * Implements the same interface as Agent (prompt, steer, abort, subscribe)
 * but routes calls through REST API + SSE instead of direct in-process calls.
 */
import type { AgentEvent, AgentMessage, AgentState, AgentTool, ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { Model } from "@mariozechner/pi-ai";

export interface RestAgentClientOptions {
	/** Base URL of the pi-mono REST server (e.g. "http://localhost:8080") */
	baseUrl: string;
	/** Session ID to connect to. If omitted, creates a new session. */
	sessionId?: string;
}

export interface SessionInfo {
	id: string;
	name: string;
	model: string | null;
	thinking_level: string;
	message_count: number;
	is_streaming: boolean;
	created_at: string;
}

export interface SessionListItem {
	id: string;
	name: string;
	model: string | null;
	message_count: number;
	created_at: string;
}

/**
 * REST-based Agent client that communicates with the Python backend
 * via HTTP calls and SSE event streams.
 */
export class RestAgentClient {
	private _baseUrl: string;
	private _sessionId: string | null = null;
	private _subscribers: Array<(e: AgentEvent) => void> = [];
	private _state: AgentState;
	private _sseManager: SseConnectionManager | null = null;

	constructor(options: RestAgentClientOptions) {
		this._baseUrl = options.baseUrl.replace(/\/$/, "");
		this._sessionId = options.sessionId ?? null;

		// Initialize with empty state
		this._state = {
			systemPrompt: "",
			model: null as any,
			thinkingLevel: "off" as ThinkingLevel,
			tools: [],
			messages: [],
			isStreaming: false,
			streamMessage: null,
			pendingToolCalls: new Set<string>(),
		};
	}

	get state(): AgentState {
		return this._state;
	}

	get sessionId(): string | null {
		return this._sessionId;
	}

	// ---------------------------------------------------------------------------
	// Session management
	// ---------------------------------------------------------------------------

	async createSession(model?: string, thinkingLevel?: string): Promise<string> {
		const body: Record<string, unknown> = {};
		if (model) body.model = model;
		if (thinkingLevel) body.thinking_level = thinkingLevel;

		const resp = await this._fetch("/api/sessions", {
			method: "POST",
			body: JSON.stringify(body),
		});
		const data = await resp.json();
		this._sessionId = data.id;
		return data.id;
	}

	async getSession(): Promise<SessionInfo> {
		this._requireSession();
		const resp = await this._fetch(`/api/sessions/${this._sessionId}`);
		return resp.json();
	}

	async listSessions(): Promise<SessionListItem[]> {
		const resp = await this._fetch("/api/sessions");
		const data = await resp.json();
		return data.sessions;
	}

	async deleteSession(sessionId?: string): Promise<void> {
		const id = sessionId ?? this._sessionId;
		if (!id) throw new Error("No session ID");
		await this._fetch(`/api/sessions/${id}`, { method: "DELETE" });
		if (id === this._sessionId) {
			this._sessionId = null;
		}
	}

	// ---------------------------------------------------------------------------
	// Agent operations (prompt, steer, abort)
	// ---------------------------------------------------------------------------

	async prompt(input: string | AgentMessage | AgentMessage[]): Promise<void> {
		this._requireSession();
		let body: Record<string, unknown>;

		if (typeof input === "string") {
			body = { content: input };
		} else if (Array.isArray(input)) {
			body = { messages: input };
		} else {
			body = { messages: [input] };
		}

		this._state = { ...this._state, isStreaming: true };
		this._emit({ type: "agent_start" });

		try {
			const resp = await this._fetch(`/api/sessions/${this._sessionId}/messages`, {
				method: "POST",
				body: JSON.stringify(body),
			});
			if (!resp.ok) {
				const err = await resp.json().catch(() => ({ detail: resp.statusText }));
				throw new Error(err.detail || `HTTP ${resp.status}`);
			}
		} catch (error) {
			this._state = { ...this._state, isStreaming: false, error: String(error) };
			this._emit({ type: "agent_end", messages: this._state.messages });
			throw error;
		}
	}

	steer(message: AgentMessage): void {
		if (!this._sessionId) return;
		// Fire-and-forget POST
		this._fetch(`/api/sessions/${this._sessionId}/steer`, {
			method: "POST",
			body: JSON.stringify({ message }),
		}).catch(console.error);
	}

	abort(): void {
		if (!this._sessionId) return;
		// Fire-and-forget POST
		this._fetch(`/api/sessions/${this._sessionId}/abort`, {
			method: "POST",
		}).catch(console.error);
	}

	// ---------------------------------------------------------------------------
	// Event subscription
	// ---------------------------------------------------------------------------

	subscribe(fn: (e: AgentEvent) => void): () => void {
		this._subscribers.push(fn);
		return () => {
			this._subscribers = this._subscribers.filter((s) => s !== fn);
		};
	}

	// ---------------------------------------------------------------------------
	// Model and settings
	// ---------------------------------------------------------------------------

	async setModel(model: Model<any>): Promise<void> {
		this._requireSession();
		await this._fetch(`/api/sessions/${this._sessionId}/model`, {
			method: "PUT",
			body: JSON.stringify({ model_id: model.id }),
		});
		this._state = { ...this._state, model };
	}

	async setThinkingLevel(level: ThinkingLevel): Promise<void> {
		this._requireSession();
		await this._fetch(`/api/sessions/${this._sessionId}/thinking`, {
			method: "PUT",
			body: JSON.stringify({ level }),
		});
		this._state = { ...this._state, thinkingLevel: level };
	}

	setTools(tools: AgentTool<any>[]): void {
		// Tools are managed server-side; store locally for UI reference
		this._state = { ...this._state, tools };
	}

	setSystemPrompt(prompt: string): void {
		this._state = { ...this._state, systemPrompt: prompt };
	}

	// ---------------------------------------------------------------------------
	// Message history
	// ---------------------------------------------------------------------------

	async getMessages(): Promise<AgentMessage[]> {
		this._requireSession();
		const resp = await this._fetch(`/api/sessions/${this._sessionId}/messages`);
		const data = await resp.json();
		return data.messages;
	}

	// ---------------------------------------------------------------------------
	// SSE connection management
	// ---------------------------------------------------------------------------

	connectEvents(): void {
		if (this._sseManager) return;
		this._requireSession();

		this._sseManager = new SseConnectionManager(
			`${this._baseUrl}/api/sessions/${this._sessionId}/events`,
			(event) => this._handleSseEvent(event),
			() => {
				// On disconnect
				this._state = { ...this._state, isStreaming: false };
			},
		);
		this._sseManager.connect();
	}

	disconnectEvents(): void {
		if (this._sseManager) {
			this._sseManager.disconnect();
			this._sseManager = null;
		}
	}

	// ---------------------------------------------------------------------------
	// Settings API
	// ---------------------------------------------------------------------------

	async getSettings(): Promise<Record<string, unknown>> {
		const resp = await this._fetch("/api/settings");
		return resp.json();
	}

	async updateSettings(settings: Record<string, unknown>): Promise<void> {
		await this._fetch("/api/settings", {
			method: "PUT",
			body: JSON.stringify(settings),
		});
	}

	// ---------------------------------------------------------------------------
	// Models API
	// ---------------------------------------------------------------------------

	async getModels(): Promise<Model<any>[]> {
		const resp = await this._fetch("/api/models");
		const data = await resp.json();
		return data.models;
	}

	// ---------------------------------------------------------------------------
	// Internal helpers
	// ---------------------------------------------------------------------------

	private _requireSession(): asserts this is { _sessionId: string } {
		if (!this._sessionId) {
			throw new Error("No active session. Call createSession() first.");
		}
	}

	private async _fetch(path: string, init?: RequestInit): Promise<Response> {
		const url = `${this._baseUrl}${path}`;
		const headers: Record<string, string> = {
			"Content-Type": "application/json",
			...(init?.headers as Record<string, string>),
		};

		return fetch(url, {
			...init,
			headers,
		});
	}

	private _emit(event: AgentEvent): void {
		for (const fn of this._subscribers) {
			try {
				fn(event);
			} catch (e) {
				console.error("Error in event subscriber:", e);
			}
		}
	}

	private _handleSseEvent(sseEvent: { type: string; data: string }): void {
		try {
			const data = JSON.parse(sseEvent.data);
			const agentEvent = this._mapSseToAgentEvent(sseEvent.type, data);
			if (agentEvent) {
				this._updateState(agentEvent);
				this._emit(agentEvent);
			}
		} catch (e) {
			console.error("Failed to parse SSE event:", e);
		}
	}

	private _mapSseToAgentEvent(type: string, data: any): AgentEvent | null {
		switch (type) {
			case "agent_start":
				return { type: "agent_start" };
			case "agent_end":
				return { type: "agent_end", messages: data.messages || [] };
			case "turn_start":
				return { type: "turn_start" };
			case "turn_end":
				return { type: "turn_end", message: data.message, toolResults: data.toolResults || [] };
			case "message_start":
				return { type: "message_start", message: data.message };
			case "message_update":
				return { type: "message_update", message: data.message, assistantMessageEvent: data.assistantMessageEvent };
			case "message_end":
				return { type: "message_end", message: data.message };
			case "tool_execution_start":
				return {
					type: "tool_execution_start",
					toolCallId: data.toolCallId,
					toolName: data.toolName,
					args: data.args,
				};
			case "tool_execution_update":
				return {
					type: "tool_execution_update",
					toolCallId: data.toolCallId,
					toolName: data.toolName,
					args: data.args,
					partialResult: data.partialResult,
				};
			case "tool_execution_end":
				return {
					type: "tool_execution_end",
					toolCallId: data.toolCallId,
					toolName: data.toolName,
					result: data.result,
					isError: data.isError,
				};
			case "heartbeat":
				return null;
			default:
				console.warn("Unknown SSE event type:", type);
				return null;
		}
	}

	private _updateState(event: AgentEvent): void {
		switch (event.type) {
			case "agent_start":
				this._state = { ...this._state, isStreaming: true, error: undefined };
				break;
			case "agent_end":
				this._state = {
					...this._state,
					isStreaming: false,
					streamMessage: null,
					messages: event.messages,
					pendingToolCalls: new Set(),
				};
				break;
			case "message_start":
				this._state = { ...this._state, streamMessage: event.message };
				break;
			case "message_update":
				this._state = { ...this._state, streamMessage: event.message };
				break;
			case "message_end":
				this._state = {
					...this._state,
					messages: [...this._state.messages, event.message],
					streamMessage: null,
				};
				break;
			case "tool_execution_start":
				this._state = {
					...this._state,
					pendingToolCalls: new Set([...this._state.pendingToolCalls, event.toolCallId]),
				};
				break;
			case "tool_execution_end":
				{
					const pending = new Set(this._state.pendingToolCalls);
					pending.delete(event.toolCallId);
					this._state = { ...this._state, pendingToolCalls: pending };
				}
				break;
		}
	}
}

// ---------------------------------------------------------------------------
// SSE Connection Manager (task 8.6)
// ---------------------------------------------------------------------------

export class SseConnectionManager {
	private _url: string;
	private _onEvent: (event: { type: string; data: string }) => void;
	private _onDisconnect: () => void;
	private _eventSource: EventSource | null = null;
	private _reconnectAttempts = 0;
	private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	private _connected = false;
	private _intentionalDisconnect = false;

	/** Maximum reconnection delay in milliseconds. */
	private static readonly MAX_RECONNECT_DELAY_MS = 30_000;
	/** Base delay for exponential backoff. */
	private static readonly BASE_DELAY_MS = 1_000;

	constructor(url: string, onEvent: (event: { type: string; data: string }) => void, onDisconnect: () => void) {
		this._url = url;
		this._onEvent = onEvent;
		this._onDisconnect = onDisconnect;
	}

	get connected(): boolean {
		return this._connected;
	}

	connect(): void {
		this._intentionalDisconnect = false;
		this._connectInternal();
	}

	disconnect(): void {
		this._intentionalDisconnect = true;
		this._cleanup();
		this._onDisconnect();
	}

	private _connectInternal(): void {
		this._cleanup();

		const es = new EventSource(this._url);
		this._eventSource = es;

		es.onopen = () => {
			this._connected = true;
			this._reconnectAttempts = 0;
		};

		es.onerror = () => {
			this._connected = false;
			if (!this._intentionalDisconnect) {
				this._scheduleReconnect();
			}
			this._onDisconnect();
		};

		// Listen for all named event types from the server
		const eventTypes = [
			"agent_start",
			"agent_end",
			"turn_start",
			"turn_end",
			"message_start",
			"message_update",
			"message_end",
			"tool_execution_start",
			"tool_execution_update",
			"tool_execution_end",
			"heartbeat",
		];

		for (const type of eventTypes) {
			es.addEventListener(type, (e: MessageEvent) => {
				this._onEvent({ type, data: e.data });
			});
		}
	}

	private _scheduleReconnect(): void {
		if (this._intentionalDisconnect) return;

		const delay = Math.min(
			SseConnectionManager.BASE_DELAY_MS * Math.pow(2, this._reconnectAttempts),
			SseConnectionManager.MAX_RECONNECT_DELAY_MS,
		);

		this._reconnectAttempts++;
		this._reconnectTimer = setTimeout(() => {
			this._connectInternal();
		}, delay);
	}

	private _cleanup(): void {
		if (this._eventSource) {
			this._eventSource.close();
			this._eventSource = null;
		}
		if (this._reconnectTimer) {
			clearTimeout(this._reconnectTimer);
			this._reconnectTimer = null;
		}
		this._connected = false;
	}
}
