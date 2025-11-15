import { MCP_CLIENT_INFO, MCP_PROTOCOL_VERSION, MCP_REQUEST_TIMEOUT_MS } from '$lib/constants/mcp';
import { browser } from '$app/environment';
import type { MCPJsonRpcRequest, MCPJsonRpcResponse, MCPToolDescription } from '$lib/types/mcp';

export type MCPTransportKind = 'sse' | 'websocket';

interface MCPClientOptions {
	endpointUrl: string;
	transport: MCPTransportKind;
	timeoutMs?: number;
	autoReconnect?: boolean;
}

interface MCPTransport {
	connect(): Promise<void>;
	disconnect(): Promise<void>;
	send(request: MCPJsonRpcRequest): Promise<MCPJsonRpcResponse | null>;
}

type PendingRequest = {
	resolve: (value: MCPJsonRpcResponse) => void;
	reject: (reason?: unknown) => void;
	timer: ReturnType<typeof setTimeout>;
};

class SSETransport implements MCPTransport {
	private pending = new Map<number | string, PendingRequest>();
	private connectPromise: Promise<void> | null = null;
	private streamAbort: AbortController | null = null;
	private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
	private buffer = '';
	private decoder = new TextDecoder();
	private sessionId: string | null = null;
	private postEndpoint: string;
	private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	private closed = false;

	constructor(
		private endpointUrl: string,
		private timeoutMs: number,
		private autoReconnect: boolean
	) {
		this.postEndpoint = endpointUrl;
	}

	async connect(): Promise<void> {
		if (!browser) {
			throw new Error('SSE transport is only available in the browser');
		}

		if (this.reader) {
			return;
		}

		if (this.connectPromise) {
			return this.connectPromise;
		}

		this.closed = false;
		this.connectPromise = this.openStream().finally(() => {
			this.connectPromise = null;
		});

		return this.connectPromise;
	}

	private async openStream(): Promise<void> {
		this.streamAbort?.abort();
		this.streamAbort = new AbortController();

		const response = await fetch(this.endpointUrl, {
			method: 'GET',
			headers: this.buildHeaders({ Accept: 'text/event-stream' }),
			signal: this.streamAbort.signal
		});

		if (!response.ok || !response.body) {
			throw new Error(`Failed to open MCP SSE stream (${response.status})`);
		}

		this.updateSessionMetadata(response);
		this.reader = response.body.getReader();
		this.buffer = '';
		void this.readLoop();
	}

	async disconnect(): Promise<void> {
		this.closed = true;
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer);
			this.reconnectTimer = null;
		}
		this.streamAbort?.abort();
		this.streamAbort = null;

		if (this.reader) {
			try {
				await this.reader.cancel();
			} catch (error) {
				console.warn('Failed to cancel MCP SSE reader:', error);
			}
			this.reader = null;
		}

		this.rejectAllPending(new Error('SSE transport disconnected'));
	}

	async send(request: MCPJsonRpcRequest): Promise<MCPJsonRpcResponse | null> {
		await this.connect();

		if (request.id === undefined || request.id === null) {
			await this.postMessage(request);
			return null;
		}

		const pending = this.registerPending(request.id);

		try {
			const immediate = await this.postMessage(request);
			if (immediate) {
				pending.cleanup();
				return immediate;
			}

			return await pending.promise;
		} catch (error) {
			pending.cleanup();
			throw error;
		}
	}

	private async postMessage(request: MCPJsonRpcRequest): Promise<MCPJsonRpcResponse | null> {
		const response = await fetch(this.postEndpoint, {
			method: 'POST',
			headers: this.buildHeaders({
				Accept: 'application/json, text/event-stream',
				'Content-Type': 'application/json'
			}),
			body: JSON.stringify(request)
		});

		this.updateSessionMetadata(response);

		if (response.status === 202 || response.status === 204) {
			return null;
		}

		if (!response.ok) {
			throw new Error(`MCP SSE request failed (${response.status})`);
		}

		const contentType = response.headers.get('content-type') || '';
		if (contentType.includes('application/json')) {
			return (await response.json()) as MCPJsonRpcResponse;
		}

		return null;
	}

	private buildHeaders(extra?: Record<string, string>): HeadersInit {
		const headers: Record<string, string> = {
			'MCP-Protocol-Version': MCP_PROTOCOL_VERSION
		};

		if (this.sessionId) {
			headers['Mcp-Session-Id'] = this.sessionId;
		}

		if (extra) {
			Object.assign(headers, extra);
		}

		return headers;
	}

	private updateSessionMetadata(response: Response) {
		const sessionHeader = response.headers.get('Mcp-Session-Id');
		if (sessionHeader) {
			this.sessionId = sessionHeader;
		}

		const postEndpoint = response.headers.get('Mcp-Post-Endpoint');
		if (postEndpoint) {
			this.postEndpoint = this.resolveEndpoint(postEndpoint);
		} else if (!this.postEndpoint) {
			this.postEndpoint = this.endpointUrl;
		}
	}

	private resolveEndpoint(endpoint: string): string {
		try {
			return new URL(endpoint, this.endpointUrl).toString();
		} catch (error) {
			console.warn('Failed to resolve MCP post endpoint:', error);
			return this.endpointUrl;
		}
	}

	private async readLoop() {
		const reader = this.reader;
		if (!reader) {
			return;
		}

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) {
					throw new Error('SSE stream closed');
				}

				if (value) {
					this.buffer += this.decoder.decode(value, { stream: true }).replace(/\r/g, '');
					this.processBuffer();
				}
			}
		} catch (error) {
			if (this.closed) {
				return;
			}

			this.handleStreamError(error instanceof Error ? error : new Error('SSE stream error'));
		}
	}

	private processBuffer() {
		let delimiterIndex = this.buffer.indexOf('\n\n');
		while (delimiterIndex !== -1) {
			const eventPayload = this.buffer.slice(0, delimiterIndex);
			this.buffer = this.buffer.slice(delimiterIndex + 2);

			const dataLines = eventPayload
				.split('\n')
				.filter((line) => line.startsWith('data:'))
				.map((line) => line.slice(5).trimStart());

			if (dataLines.length) {
				this.handleIncomingData(dataLines.join('\n'));
			}

			delimiterIndex = this.buffer.indexOf('\n\n');
		}
	}

	private handleIncomingData(data: string) {
		const trimmed = data.trim();
		if (!trimmed) {
			return;
		}

		try {
			const payload = JSON.parse(trimmed) as MCPJsonRpcResponse;
			if (payload.id === undefined || payload.id === null) {
				return;
			}

			const pending = this.pending.get(payload.id);
			if (!pending) {
				return;
			}

			clearTimeout(pending.timer);
			this.pending.delete(payload.id);
			pending.resolve(payload);
		} catch (error) {
			console.warn('Failed to parse MCP SSE payload:', error);
		}
	}

	private handleStreamError(error: Error) {
		this.reader = null;
		this.rejectAllPending(error);

		if (this.autoReconnect && !this.closed) {
			if (this.reconnectTimer) {
				return;
			}

			this.reconnectTimer = setTimeout(() => {
				this.reconnectTimer = null;
				this.connect().catch((err) => {
					console.warn('Failed to reconnect MCP SSE transport:', err);
				});
			}, 1000);
		}
	}

	private rejectAllPending(error: Error) {
		for (const [, pending] of this.pending) {
			clearTimeout(pending.timer);
			pending.reject(error);
		}

		this.pending.clear();
	}

	private registerPending(id: number | string): {
		promise: Promise<MCPJsonRpcResponse>;
		cleanup: () => void;
	} {
		let resolveFn!: (value: MCPJsonRpcResponse) => void;
		let rejectFn!: (reason?: unknown) => void;

		const promise = new Promise<MCPJsonRpcResponse>((resolve, reject) => {
			resolveFn = resolve;
			rejectFn = reject;
		});

		const timer = setTimeout(() => {
			this.pending.delete(id);
			rejectFn(new Error('MCP SSE request timed out'));
		}, this.timeoutMs);

		this.pending.set(id, { resolve: resolveFn!, reject: rejectFn!, timer });

		return {
			promise,
			cleanup: () => {
				const pending = this.pending.get(id);
				if (pending) {
					clearTimeout(pending.timer);
					this.pending.delete(id);
				}
			}
		};
	}
}

class WebSocketTransport implements MCPTransport {
	private socket: WebSocket | null = null;
	private pending = new Map<number | string, PendingRequest>();
	private connectPromise: Promise<void> | null = null;

	constructor(
		private endpointUrl: string,
		private timeoutMs: number,
		private autoReconnect: boolean
	) {}

	async connect(): Promise<void> {
		if (!browser) {
			throw new Error('WebSocket transport is only available in the browser');
		}

		if (this.socket && this.socket.readyState === WebSocket.OPEN) {
			return;
		}

		if (this.connectPromise) {
			return this.connectPromise;
		}

		this.connectPromise = new Promise((resolve, reject) => {
			const ws = new WebSocket(this.endpointUrl);

			const handleError = (event: Event) => {
				this.socket = null;
				ws.close();
				this.connectPromise = null;
				reject(new Error(`WebSocket error: ${event.type}`));
			};

			ws.onopen = () => {
				this.socket = ws;
				this.connectPromise = null;
				ws.onerror = (event) => {
					console.warn('MCP WebSocket transport error:', event);
				};
				ws.onmessage = (event) => {
					this.handleMessage(event.data);
				};
				ws.onclose = () => {
					this.handleClose();
				};
				resolve();
			};

			ws.onerror = handleError;
		});

		return this.connectPromise;
	}

	async disconnect(): Promise<void> {
		this.autoReconnect = false;
		this.socket?.close();
		this.socket = null;
		this.rejectAllPending(new Error('WebSocket transport disconnected'));
	}

	async send(request: MCPJsonRpcRequest): Promise<MCPJsonRpcResponse | null> {
		if (request.id === undefined || request.id === null) {
			await this.connect();
			this.socket?.send(JSON.stringify(request));
			return null;
		}

		await this.connect();

		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(request.id!);
				reject(new Error('MCP WebSocket request timed out'));
			}, this.timeoutMs);

			this.pending.set(request.id!, { resolve, reject, timer });
			this.socket?.send(JSON.stringify(request));
		});
	}

	private handleMessage(data: string) {
		try {
			const parsed = JSON.parse(data) as MCPJsonRpcResponse;
			if (parsed.id === undefined || parsed.id === null) {
				return;
			}

			const pending = this.pending.get(parsed.id);
			if (!pending) {
				return;
			}

			clearTimeout(pending.timer);
			this.pending.delete(parsed.id);
			pending.resolve(parsed);
		} catch (error) {
			console.warn('Failed to parse MCP WebSocket payload:', error);
		}
	}

	private handleClose() {
		this.socket = null;
		this.rejectAllPending(new Error('MCP WebSocket connection closed'));

		if (this.autoReconnect) {
			setTimeout(() => {
				this.connect().catch((error) => {
					console.warn('Failed to reconnect MCP WebSocket:', error);
				});
			}, 500);
		}
	}

	private rejectAllPending(error: Error) {
		for (const [, pending] of this.pending) {
			clearTimeout(pending.timer);
			pending.reject(error);
		}
		this.pending.clear();
	}
}

export class MCPClient {
	private transport: MCPTransport;
	private requestCounter = 0;
	private tools: MCPToolDescription[] = [];
	private initialized = false;
	private currentEndpointKey: string;

	constructor(private options: MCPClientOptions) {
		const timeout = options.timeoutMs ?? MCP_REQUEST_TIMEOUT_MS;
		this.currentEndpointKey = `${options.transport}:${options.endpointUrl}`;

		if (options.transport === 'sse') {
			this.transport = new SSETransport(
				options.endpointUrl,
				timeout,
				options.autoReconnect !== false
			);
		} else {
			this.transport = new WebSocketTransport(
				options.endpointUrl,
				timeout,
				options.autoReconnect !== false
			);
		}
	}

	getTools(): MCPToolDescription[] {
		return this.tools;
	}

	async ensureInitialized(): Promise<void> {
		if (this.initialized) {
			return;
		}

		await this.transport.connect();

		const initializeResponse = await this.sendRequest('initialize', {
			protocolVersion: MCP_PROTOCOL_VERSION,
			capabilities: {
				tools: { listChanged: true }
			},
			clientInfo: MCP_CLIENT_INFO
		});

		if (!initializeResponse || initializeResponse.error) {
			const message = initializeResponse?.error?.message || 'Failed to initialize MCP session';
			throw new Error(message);
		}

		await this.transport.send({
			jsonrpc: '2.0',
			method: 'notifications/initialized'
		});

		await this.refreshTools();
		this.initialized = true;
	}

	async refreshTools(): Promise<MCPToolDescription[]> {
		const response = await this.sendRequest('tools/list', {});
		const result = (response?.result as { tools?: MCPToolDescription[] }) ?? {};
		this.tools = Array.isArray(result.tools) ? result.tools : [];
		return this.tools;
	}

	async callTool(name: string, args: unknown): Promise<unknown> {
		await this.ensureInitialized();
		const response = await this.sendRequest('tools/call', { name, arguments: args });
		return response?.result ?? null;
	}

	private async sendRequest(
		method: string,
		params: Record<string, unknown> | undefined
	): Promise<MCPJsonRpcResponse | null> {
		const request: MCPJsonRpcRequest = {
			jsonrpc: '2.0',
			id: ++this.requestCounter,
			method,
			params
		};

		const response = await this.transport.send(request);
		if (!response) {
			return null;
		}

		if (response.error) {
			throw new Error(response.error.message || 'MCP request failed');
		}

		return response;
	}

	async disconnect(): Promise<void> {
		this.initialized = false;
		await this.transport.disconnect();
	}

	get connectionKey(): string {
		return this.currentEndpointKey;
	}
}
