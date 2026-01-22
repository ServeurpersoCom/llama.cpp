/**
 * agenticStore - Reactive State Store for Agentic Loop
 *
 * This store contains ONLY reactive state ($state).
 * All business logic is delegated to AgenticClient.
 *
 * **Architecture & Relationships:**
 * - **AgenticClient**: Business logic facade (loop orchestration, tool execution)
 * - **MCPClient**: Tool execution via MCP servers
 * - **agenticStore** (this): Reactive state for UI components
 *
 * **Responsibilities:**
 * - Hold per-conversation reactive state for UI binding
 * - Provide getters for computed values (scoped by conversationId)
 * - Expose setters for AgenticClient to update state
 * - Forward method calls to AgenticClient
 * - Track sampling requests for debugging
 *
 * **Per-Conversation Architecture:**
 * - Each conversation has its own AgenticSession
 * - Parallel agentic flows in different chats don't interfere
 * - Sessions are created on-demand and cleaned up when done
 *
 * @see AgenticClient in clients/agentic/ for business logic
 * @see MCPClient in clients/mcp/ for tool execution
 */

import { browser } from '$app/environment';
import type { AgenticFlowParams, AgenticFlowResult } from '$lib/clients';
import type { AgenticSession } from '$lib/types/agentic';

export type {
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	AgenticFlowParams,
	AgenticFlowResult
} from '$lib/clients';

/**
 * Creates a fresh agentic session with default values.
 */
function createDefaultSession(): AgenticSession {
	return {
		isRunning: false,
		currentTurn: 0,
		totalToolCalls: 0,
		lastError: null,
		streamingToolCall: null
	};
}

class AgenticStore {
	/**
	 * Per-conversation agentic sessions.
	 * Key is conversationId, value is the session state.
	 */
	private _sessions = $state<Map<string, AgenticSession>>(new Map());

	/** Reference to the client (lazy loaded to avoid circular dependency) */
	private _client: typeof import('$lib/clients/agentic.client').agenticClient | null = null;

	private get client() {
		return this._client;
	}

	/** Check if store is ready (client initialized) */
	get isReady(): boolean {
		return this._client !== null;
	}

	/**
	 * Initialize the store by wiring up to the client.
	 * Must be called once after app startup.
	 */
	async init(): Promise<void> {
		if (!browser) return;
		if (this._client) return; // Already initialized

		const { agenticClient } = await import('$lib/clients/agentic.client');
		this._client = agenticClient;

		agenticClient.setStoreCallbacks({
			setRunning: (convId, running) => this.updateSession(convId, { isRunning: running }),
			setCurrentTurn: (convId, turn) => this.updateSession(convId, { currentTurn: turn }),
			setTotalToolCalls: (convId, count) => this.updateSession(convId, { totalToolCalls: count }),
			setLastError: (convId, error) => this.updateSession(convId, { lastError: error }),
			setStreamingToolCall: (convId, tc) => this.updateSession(convId, { streamingToolCall: tc }),
			clearStreamingToolCall: (convId) => this.updateSession(convId, { streamingToolCall: null })
		});
	}

	/**
	 *
	 * Session Management
	 *
	 */

	/**
	 * Get session for a conversation, creating if needed.
	 */
	getSession(conversationId: string): AgenticSession {
		let session = this._sessions.get(conversationId);
		if (!session) {
			session = createDefaultSession();
			this._sessions.set(conversationId, session);
		}
		return session;
	}

	/**
	 * Update session state for a conversation.
	 */
	private updateSession(conversationId: string, update: Partial<AgenticSession>): void {
		const session = this.getSession(conversationId);
		const updated = { ...session, ...update };
		this._sessions.set(conversationId, updated);
	}

	/**
	 * Clear session for a conversation.
	 */
	clearSession(conversationId: string): void {
		this._sessions.delete(conversationId);
	}

	/**
	 * Get all active sessions (conversations with running agentic flows).
	 */
	getActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
		const active: Array<{ conversationId: string; session: AgenticSession }> = [];
		for (const [conversationId, session] of this._sessions.entries()) {
			if (session.isRunning) {
				active.push({ conversationId, session });
			}
		}
		return active;
	}

	/**
	 *
	 * Convenience Getters (for current/active conversation)
	 *
	 */

	/**
	 * Check if any agentic flow is running (global).
	 */
	get isAnyRunning(): boolean {
		for (const session of this._sessions.values()) {
			if (session.isRunning) return true;
		}
		return false;
	}

	/**
	 * Get running state for a specific conversation.
	 */
	isRunning(conversationId: string): boolean {
		return this.getSession(conversationId).isRunning;
	}

	/**
	 * Get current turn for a specific conversation.
	 */
	currentTurn(conversationId: string): number {
		return this.getSession(conversationId).currentTurn;
	}

	/**
	 * Get total tool calls for a specific conversation.
	 */
	totalToolCalls(conversationId: string): number {
		return this.getSession(conversationId).totalToolCalls;
	}

	/**
	 * Get last error for a specific conversation.
	 */
	lastError(conversationId: string): Error | null {
		return this.getSession(conversationId).lastError;
	}

	/**
	 * Get streaming tool call for a specific conversation.
	 */
	streamingToolCall(conversationId: string): { name: string; arguments: string } | null {
		return this.getSession(conversationId).streamingToolCall;
	}

	/**
	 *
	 * Agentic Flow Execution
	 *
	 */

	/**
	 * Run the agentic orchestration loop with MCP tools.
	 * Delegates to AgenticClient.
	 */
	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		if (!this.client) {
			throw new Error('AgenticStore not initialized. Call init() first.');
		}
		return this.client.runAgenticFlow(params);
	}

	/**
	 * Clear error state for a conversation.
	 */
	clearError(conversationId: string): void {
		this.updateSession(conversationId, { lastError: null });
	}
}

export const agenticStore = new AgenticStore();

// Auto-initialize in browser
if (browser) {
	agenticStore.init();
}

/**
 * Helper functions for reactive access in components.
 * These require conversationId parameter for per-conversation state.
 */
export function agenticIsRunning(conversationId: string) {
	return agenticStore.isRunning(conversationId);
}

export function agenticCurrentTurn(conversationId: string) {
	return agenticStore.currentTurn(conversationId);
}

export function agenticTotalToolCalls(conversationId: string) {
	return agenticStore.totalToolCalls(conversationId);
}

export function agenticLastError(conversationId: string) {
	return agenticStore.lastError(conversationId);
}

export function agenticStreamingToolCall(conversationId: string) {
	return agenticStore.streamingToolCall(conversationId);
}

export function agenticIsAnyRunning() {
	return agenticStore.isAnyRunning;
}
