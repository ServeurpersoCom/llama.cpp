/**
 * MCP Client: Manages multiple MCP servers, discovery, routing, and execution
 */

const TransportStdio = require('./transport-stdio.js');
const TransportWebSocket = require('./transport-websocket.js');
const TransportStreamableHTTP = require('./transport-streamable-http.js');
const Protocol = require('./protocol.js');

class MCPClient {
	constructor(config) {
		if (!config || !config.servers) {
			throw new Error('MCPClient requires config.servers');
		}

		this.config = config;
		this.servers = new Map(); // name -> { transport, tools, pending, requestId, capabilities, protocolVersion }
		this.toolsMap = new Map(); // toolName -> serverName
	}

	/**
	 * Initialize all MCP servers: connect, handshake, discover tools
	 */
	async initialize() {
		console.log('[MCP] Initializing MCP servers...');

		for (const [name, serverConfig] of Object.entries(this.config.servers)) {
			console.log(`[MCP] Connecting to server "${name}"...`);

			// Start transport
			const transport = this._createTransport(name, serverConfig);
			await transport.start();

			const serverState = {
				transport: transport,
				pending: new Map(),
				requestId: 0,
				tools: []
			};

			// Setup message handler
			transport.onMessage((msg) => this._handleMessage(name, msg));

			this.servers.set(name, serverState);

			// Initialize MCP protocol
			console.log(`[MCP] Initializing protocol for "${name}"...`);
			const initResult = await this._call(name, 'initialize', {
				protocolVersion: '2025-06-18',
				capabilities: { tools: { listChanged: true } },
				clientInfo: { name: 'llama-mcp-bridge', version: '1.0.0' }
			});

			const negotiatedVersion = initResult?.protocolVersion || '2025-06-18';
			if (negotiatedVersion !== '2025-06-18') {
				throw new Error(
					`Unsupported MCP protocol version from "${name}": ${negotiatedVersion}`
				);
			}

			serverState.capabilities = initResult?.capabilities || {};
			serverState.protocolVersion = negotiatedVersion;

			// Send initialized notification as required by MCP lifecycle
			serverState.transport.send(Protocol.createNotification('notifications/initialized'));

			// Discover tools
			await this._refreshTools(name);

			console.log(`[MCP] Server "${name}" connected (${serverState.tools.length} tools)`);
		}

		console.log(`[MCP] Total tools discovered: ${this.toolsMap.size}`);
	}

	/**
	 * Execute a tool call (OpenAI format)
	 * @param {object} toolCall - { id, function: { name, arguments } }
	 * @returns {string} Tool result
	 */
	async execute(toolCall) {
		const toolName = toolCall.function.name;
		const serverName = this.toolsMap.get(toolName);

		if (!serverName) {
			throw new Error(`Tool "${toolName}" not found in any MCP server`);
		}

		// Parse arguments if string
		let args = toolCall.function.arguments;
		if (typeof args === 'string') {
			try {
				args = JSON.parse(args);
			} catch (e) {
				throw new Error(`Failed to parse tool arguments: ${e.message}`);
			}
		}

		// Call MCP server
		const response = await this._call(serverName, 'tools/call', {
			name: toolName,
			arguments: args
		});

		// Extract content from MCP response
		if (response && typeof response === 'object') {
			if (response.content !== undefined) {
				const contentItems = Array.isArray(response.content)
					? response.content
					: [response.content];
				const flattened = contentItems
					.map((item) => {
						if (item == null) return '';
						if (typeof item === 'string') return item;
						if (typeof item === 'object') {
							if (item.type === 'text' && typeof item.text === 'string') {
								return item.text;
							}
							if (item.type === 'resource' && item.resource) {
								if (typeof item.resource.text === 'string') {
									return item.resource.text;
								}
								return JSON.stringify(item.resource);
							}
							if (item.type === 'image' && item.data) {
								const mimeType = item.mimeType || 'application/octet-stream';
								return `data:${mimeType};base64,${item.data}`;
							}
							if (typeof item.text === 'string') {
								return item.text;
							}
							return JSON.stringify(item);
						}
						return String(item);
					})
					.filter(Boolean)
					.join('\n');
				if (flattened) {
					return flattened;
				}
			}

			if (response.result !== undefined) {
				if (typeof response.result === 'string') return response.result;
				if (typeof response.result === 'object') return JSON.stringify(response.result);
				return String(response.result);
			}
		}

		return '';
	}

	/**
	 * Get OpenAI-compatible tools definition for all servers
	 * @returns {Array} Tools array for llama.cpp
	 */
	async getToolsDefinition() {
		const tools = [];

		for (const [serverName, serverState] of this.servers) {
			for (const tool of serverState.tools) {
				tools.push({
					type: 'function',
					function: {
						name: tool.name,
						description: tool.description,
						parameters: tool.inputSchema || {
							type: 'object',
							properties: {},
							required: []
						}
					}
				});
			}
		}

		return tools;
	}

	/**
	 * List all available tool names
	 * @returns {Array<string>} Tool names
	 */
	listTools() {
		return Array.from(this.toolsMap.keys());
	}

	/**
	 * Shutdown all MCP servers
	 */
	async shutdown() {
		console.log('[MCP] Shutting down all servers...');
		for (const [name, serverState] of this.servers) {
			await serverState.transport.stop();
			console.log(`[MCP] Server "${name}" stopped`);
		}
	}

	// Private methods

	_createTransport(serverName, serverConfig) {
		const rawTransport = serverConfig.transport ?? 'stdio';
		let transportType;
		let transportConfig = serverConfig;

		if (typeof rawTransport === 'string') {
			transportType = rawTransport.toLowerCase();
		} else if (rawTransport && typeof rawTransport === 'object') {
			transportType = (rawTransport.type || 'stdio').toLowerCase();
			transportConfig = { ...serverConfig, ...rawTransport };
			delete transportConfig.transport;
			delete transportConfig.type;
		} else {
			transportType = 'stdio';
		}

		switch (transportType) {
			case 'stdio':
				return new TransportStdio(transportConfig);
			case 'websocket':
			case 'ws':
				return new TransportWebSocket(transportConfig);
			case 'streamable-http':
			case 'http':
				return new TransportStreamableHTTP(transportConfig);
			default:
				throw new Error(
					`Unsupported transport "${transportType}" for server "${serverName}"`
				);
		}
	}

	/**
	 * Call a method on an MCP server
	 * @private
	 */
	_call(serverName, method, params = {}) {
		return new Promise((resolve, reject) => {
			const serverState = this.servers.get(serverName);
			if (!serverState) {
				return reject(new Error(`Server "${serverName}" not found`));
			}

			const id = ++serverState.requestId;
			const request = Protocol.createRequest(id, method, params);

			serverState.pending.set(id, { resolve, reject });

			try {
				serverState.transport.send(request);
			} catch (err) {
				serverState.pending.delete(id);
				return reject(err);
			}

			// Timeout 300s
			setTimeout(() => {
				if (serverState.pending.has(id)) {
					serverState.pending.delete(id);
					reject(new Error(`Timeout: ${method} on ${serverName}`));
				}
			}, 300000);
		});
	}

	/**
	 * Handle incoming message from MCP server
	 * @private
	 */
	_handleMessage(serverName, message) {
		const serverState = this.servers.get(serverName);
		if (!serverState) return;

		if (
			message &&
			typeof message === 'object' &&
			message.method &&
			!Object.prototype.hasOwnProperty.call(message, 'id')
		) {
			this._handleNotification(serverName, message);
			return;
		}

		const response = Protocol.parseResponse(message);
		if (!response || !response.id) return;

		const pending = serverState.pending.get(response.id);
		if (!pending) return;

		serverState.pending.delete(response.id);

		if (response.error) {
			pending.reject(new Error(response.error.message || 'MCP Error'));
		} else {
			pending.resolve(response.result);
		}
	}

	async _refreshTools(serverName) {
		const serverState = this.servers.get(serverName);
		if (!serverState) {
			return;
		}

		console.log(`[MCP] Discovering tools from "${serverName}"...`);

		// Remove previous tools for this server from global map
		for (const tool of serverState.tools) {
			this.toolsMap.delete(tool.name);
		}

		const response = await this._call(serverName, 'tools/list');
		const tools = response.tools || [];
		serverState.tools = tools;

		for (const tool of tools) {
			this.toolsMap.set(tool.name, serverName);
		}

		console.log(`[MCP] Updated tool catalogue for "${serverName}" (${tools.length} tools)`);
	}

	_handleNotification(serverName, message) {
		if (message.method === 'notifications/tools/list_changed') {
			this._refreshTools(serverName).catch((err) => {
				console.error(`[MCP] Failed to refresh tools for "${serverName}": ${err.message}`);
			});
		}
	}
}

module.exports = MCPClient;
