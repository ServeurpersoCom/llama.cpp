import type {
	MCPClientCapabilities,
	MCPClientConfig,
	MCPClientInfo,
	MCPServerConfig,
	MCPTransportType
} from '../mcp/types';

/**
 * MCP server configuration entry.
 * Supports both WebSocket and Streamable HTTP transports.
 */
export type MCPServerEntry = {
	transport?: string;
	url?: string;
	headers?: Record<string, string>;
	protocols?: string | string[];
	credentials?: RequestCredentials;
	sessionId?: string;
	connectionTimeoutMs?: number;
};

/**
 * MCP client configuration.
 */
export type MCPConfig = {
	protocolVersion?: string;
	capabilities?: MCPClientCapabilities;
	clientInfo?: MCPClientInfo;
	requestTimeoutMs?: number;
	servers?: Record<string, MCPServerEntry>;
};

/**
 * Default MCP configuration values.
 *
 * Timeouts are configured for local development with potentially long-running operations:
 * - connectionTimeoutMs: 10s for establishing connections (WebSocket handshake, HTTP initial connect)
 * - requestTimeoutMs: 300s (5 minutes) for MCP tool execution
 *   Long timeout accounts for operations like: git cloning, sandbox initialization,
 *   large file processing, external API calls, etc.
 */
const defaultMcpConfig = {
	protocolVersion: '2025-06-18',
	capabilities: { tools: { listChanged: true } } as MCPClientCapabilities,
	clientInfo: { name: 'llama-webui-mcp', version: 'dev' } as MCPClientInfo,
	requestTimeoutMs: 300_000, // 5 minutes for long-running tools
	connectionTimeoutMs: 10_000 // 10 seconds for connection establishment
};

/**
 * MCP servers configuration.
 * Add your MCP servers here for development.
 *
 * Configuration parameters:
 * - transport: 'websocket' or 'streamable_http' (default: 'streamable_http')
 * - url: Server endpoint URL (required)
 * - headers: Custom HTTP headers for streamable_http transport
 * - protocols: WebSocket subprotocols for websocket transport
 * - credentials: Fetch credentials policy ('include', 'same-origin', 'omit')
 * - sessionId: Pre-negotiated session ID for streamable_http
 * - connectionTimeoutMs: Connection timeout (both transports, default: 10000ms)
 *
 * Global settings:
 * - requestTimeoutMs: Timeout for MCP tool execution (default: 300000ms = 5 minutes)
 *   Generous timeout for long operations: git clone, sandbox init, file processing, etc.
 *
 * @example
 * ```typescript
 * servers: {
 *   'local-mcp': {
 *     transport: 'websocket',
 *     url: 'ws://localhost:3100',
 *     protocols: ['mcp'],
 *     connectionTimeoutMs: 15000  // Override default 10s
 *   }
 * }
 * ```
 */
export const mcpConfig: MCPConfig = {
	protocolVersion: defaultMcpConfig.protocolVersion,
	capabilities: defaultMcpConfig.capabilities,
	clientInfo: defaultMcpConfig.clientInfo,
	requestTimeoutMs: defaultMcpConfig.requestTimeoutMs,
	servers: {
		serveurperso: {
			transport: 'streamable_http',
			url: 'https://www.serveurperso.com/ia/mcp-streamable-http'
			// Uses global requestTimeoutMs (300s = 5 minutes)
			// Uses default connectionTimeoutMs (10s)
		}
		// Example WebSocket server with custom timeouts:
		// 'local-mcp': {
		//	transport: 'websocket',
		//	url: 'ws://localhost:3100',
		//	protocols: ['mcp'],
		//	connectionTimeoutMs: 15000  // 15s connection timeout
		// },
		// Example Streamable HTTP server with authentication:
		// 'authenticated-server': {
		//	transport: 'streamable_http',
		//	url: 'https://api.example.com/mcp',
		//	credentials: 'include',  // Send cookies
		//	headers: {
		//		'X-API-Key': 'your-api-key'
		//	}
		// }
	}
};

/**
 * Normalizes transport type string to MCPTransportType.
 */
function normalizeTransport(transport?: string): MCPTransportType {
	return transport?.toLowerCase() === 'websocket' ? 'websocket' : 'streamable_http';
}

/**
 * Converts an MCP server entry to an MCP server config.
 */
function buildServerConfig(entry: MCPServerEntry): MCPServerConfig | undefined {
	if (!entry?.url) {
		return undefined;
	}

	return {
		url: entry.url,
		transport: normalizeTransport(entry.transport),
		headers: entry.headers,
		protocols: entry.protocols,
		credentials: entry.credentials,
		sessionId: entry.sessionId,
		handshakeTimeoutMs: entry.connectionTimeoutMs ?? defaultMcpConfig.connectionTimeoutMs
	};
}

/**
 * Builds MCP client configuration from mcpConfig.
 * Returns undefined if no valid servers are configured.
 */
export function buildMcpClientConfig(): MCPClientConfig | undefined {
	const rawServers = mcpConfig.servers;

	if (!rawServers || Object.keys(rawServers).length === 0) {
		return undefined;
	}

	const servers: Record<string, MCPServerConfig> = {};
	for (const [name, entry] of Object.entries(rawServers)) {
		const normalized = buildServerConfig(entry);
		if (normalized) {
			servers[name] = normalized;
		}
	}

	if (Object.keys(servers).length === 0) {
		return undefined;
	}

	return {
		protocolVersion: mcpConfig.protocolVersion ?? defaultMcpConfig.protocolVersion,
		capabilities: mcpConfig.capabilities ?? defaultMcpConfig.capabilities,
		clientInfo: mcpConfig.clientInfo ?? defaultMcpConfig.clientInfo,
		requestTimeoutMs: mcpConfig.requestTimeoutMs ?? defaultMcpConfig.requestTimeoutMs,
		servers
	};
}
