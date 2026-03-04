/**
 * REST-backed data service for web-ui components.
 *
 * Provides the same data access patterns as the local storage stores
 * but routes calls through the RestAgentClient to the Python backend.
 */
import type { Model } from "@mariozechner/pi-ai";
import type { SessionMetadata } from "../storage/types.js";
import type { RestAgentClient, SessionListItem } from "./rest-agent-client.js";

/**
 * Adapts the REST API responses into the same shapes used by local storage stores.
 * This allows UI components to work with either data source.
 */
export class RestDataService {
	private _client: RestAgentClient;

	constructor(client: RestAgentClient) {
		this._client = client;
	}

	get client(): RestAgentClient {
		return this._client;
	}

	// ---------------------------------------------------------------------------
	// Sessions
	// ---------------------------------------------------------------------------

	async getAllSessionMetadata(): Promise<SessionMetadata[]> {
		const sessions = await this._client.listSessions();
		return sessions.map((s: SessionListItem) => ({
			id: s.id,
			title: s.name || "Untitled",
			createdAt: s.created_at,
			lastModified: s.created_at,
			messageCount: s.message_count,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			thinkingLevel: "off",
			preview: "",
		}));
	}

	async deleteSession(sessionId: string): Promise<void> {
		await this._client.deleteSession(sessionId);
	}

	// ---------------------------------------------------------------------------
	// Models
	// ---------------------------------------------------------------------------

	async getModels(): Promise<Model<any>[]> {
		return this._client.getModels();
	}

	// ---------------------------------------------------------------------------
	// Settings
	// ---------------------------------------------------------------------------

	async getSettings(): Promise<Record<string, unknown>> {
		return this._client.getSettings();
	}

	async updateSettings(settings: Record<string, unknown>): Promise<void> {
		return this._client.updateSettings(settings);
	}
}

/**
 * Global REST data service instance.
 * Set this when the app initializes with a REST backend.
 */
let _restDataService: RestDataService | null = null;

export function setRestDataService(service: RestDataService | null): void {
	_restDataService = service;
}

export function getRestDataService(): RestDataService | null {
	return _restDataService;
}

/** Check if the app is using REST mode. */
export function isRestMode(): boolean {
	return _restDataService !== null;
}
