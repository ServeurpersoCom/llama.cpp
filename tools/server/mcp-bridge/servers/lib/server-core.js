/**
 * Base class for MCP servers.
 * Handles JSON-RPC lifecycle, signal handling, and tool dispatch.
 */
class MCPServer {
	constructor(config, toolsModule) {
		this.config = config;
		this.tools = toolsModule.TOOLS_DEFINITIONS;
		this.toolsMapping = toolsModule.TOOLS_MAPPING;
	}

	async handleRequest(request) {
		const { id, method, params } = request;

		switch (method) {
			case 'initialize':
				return this.handleInitialize(id);
			case 'notifications/initialized':
				return null;
			case 'tools/list':
				return this.handleToolsList(id);
			case 'tools/call':
				return this.handleToolCall(id, params);
			case 'ping':
				return { jsonrpc: '2.0', id, result: {} };
			default:
				if (id === undefined || id === null) {
					return null;
				}
				return this.createError(id, -32601, `Method not found: ${method}`);
		}
	}

	handleInitialize(id) {
		return {
			jsonrpc: '2.0',
			id,
			result: {
				protocolVersion: this.config.mcp.protocolVersion,
				capabilities: { tools: {} },
				serverInfo: {
					name: this.config.mcp.serverName,
					version: this.config.mcp.serverVersion
				}
			}
		};
	}

	handleToolsList(id) {
		return {
			jsonrpc: '2.0',
			id,
			result: { tools: this.tools }
		};
	}

	async handleToolCall(id, params = {}) {
		const { name, arguments: args } = params;

		if (!this.toolsMapping[name]) {
			return this.createError(id, -32601, `Unknown tool: ${name}`);
		}

		try {
			const result = await this.toolsMapping[name](args ?? {});

			if (!result || typeof result !== 'object' || !result.type) {
				throw new Error('Tool returned invalid result. Expected typed object.');
			}

			const contentBlock = this.createContentBlock(result);
			return {
				jsonrpc: '2.0',
				id,
				result: {
					content: [contentBlock],
					isError: Boolean(result.isError)
				}
			};
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			return this.createError(id, -32603, `Tool execution failed: ${message}`);
		}
	}

	createContentBlock(result) {
		if (result.type === 'text') {
			return {
				type: 'text',
				text:
					typeof result.text === 'string'
						? result.text
						: JSON.stringify(result.text ?? '')
			};
		}

		if (result.type === 'image') {
			return {
				type: 'image',
				data: typeof result.data === 'string' ? result.data : '',
				mimeType: result.mimeType || 'application/octet-stream'
			};
		}

		throw new Error(`Unsupported tool result type: ${result.type}`);
	}

	createError(id, code, message) {
		return {
			jsonrpc: '2.0',
			id,
			error: { code, message }
		};
	}

	setupSignalHandlers() {
		const shutdown = () => {
			console.error(`[${this.config.mcp.serverName}] Shutting down...`);
			process.exit(0);
		};
		process.on('SIGTERM', shutdown);
		process.on('SIGINT', shutdown);
	}
}

module.exports = MCPServer;
