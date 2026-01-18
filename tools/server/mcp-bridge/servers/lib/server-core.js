/**
 * MCP Server core using @modelcontextprotocol/sdk
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const {
	ListToolsRequestSchema,
	CallToolRequestSchema
} = require('@modelcontextprotocol/sdk/types.js');

class MCPServer {
	constructor(config, toolsModule) {
		this.config = config;
		this.toolsDefinitions = toolsModule.TOOLS_DEFINITIONS;
		this.toolsMapping = toolsModule.TOOLS_MAPPING;

		// Create SDK Server instance
		this.server = new Server(
			{
				name: config.mcp.serverName,
				version: config.mcp.serverVersion
			},
			{
				capabilities: {
					tools: {}
				}
			}
		);

		// Register tool list handler
		this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
			tools: this.toolsDefinitions
		}));

		// Register tool call handler
		this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
			const { name, arguments: args } = request.params;

			if (!this.toolsMapping[name]) {
				throw new Error(`Unknown tool: ${name}`);
			}

			try {
				const result = await this.toolsMapping[name](args ?? {});

				if (!result || typeof result !== 'object' || !result.type) {
					throw new Error('Tool returned invalid result. Expected typed object.');
				}

				const contentBlock = this._createContentBlock(result);
				return {
					content: [contentBlock],
					isError: Boolean(result.isError)
				};
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				throw new Error(`Tool execution failed: ${message}`);
			}
		});
	}

	_createContentBlock(result) {
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

	/**
	 * Get SDK Server instance for transport connection
	 */
	getServer() {
		return this.server;
	}

	setupSignalHandlers() {
		const shutdown = () => {
			console.error(`[${this.config.mcp.serverName}] Shutting down...`);
			this.server.close();
			process.exit(0);
		};
		process.on('SIGTERM', shutdown);
		process.on('SIGINT', shutdown);
	}
}

module.exports = MCPServer;
