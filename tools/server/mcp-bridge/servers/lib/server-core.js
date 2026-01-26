/**
 * MCP Server core using @modelcontextprotocol/sdk
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const {
	ListToolsRequestSchema,
	CallToolRequestSchema,
	ListPromptsRequestSchema,
	GetPromptRequestSchema
} = require('@modelcontextprotocol/sdk/types.js');

class MCPServer {
	constructor(config, toolsModule, promptsModule = null) {
		this.config = config;
		this.toolsDefinitions = toolsModule.TOOLS_DEFINITIONS;
		this.toolsMapping = toolsModule.TOOLS_MAPPING;
		this.promptsDefinitions = promptsModule?.PROMPTS_DEFINITIONS || [];
		this.promptsMapping = promptsModule?.PROMPTS_MAPPING || {};

		// Build capabilities based on available features
		const capabilities = {
			tools: {}
		};
		if (this.promptsDefinitions.length > 0) {
			capabilities.prompts = { listChanged: true };
		}

		// Create SDK Server instance
		this.server = new Server(
			{
				name: config.mcp.serverName,
				version: config.mcp.serverVersion
			},
			{ capabilities }
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

		// Register prompts list handler
		this.server.setRequestHandler(ListPromptsRequestSchema, async (request) => {
			// Simple implementation without pagination (suitable for small prompt lists)
			return {
				prompts: this.promptsDefinitions
			};
		});

		// Register prompt get handler
		this.server.setRequestHandler(GetPromptRequestSchema, async (request) => {
			const { name, arguments: promptArgs } = request.params;

			if (!this.promptsMapping[name]) {
				throw new Error(`Unknown prompt: ${name}`);
			}

			const prompt = this.promptsMapping[name];

			// Process messages with argument substitution if needed
			const messages = prompt.messages.map((msg) => {
				if (msg.content?.type === 'text' && promptArgs) {
					let text = msg.content.text;
					for (const [key, value] of Object.entries(promptArgs)) {
						text = text.replace(new RegExp(`\\{\\{${key}\\}\\}`, 'g'), value);
					}
					return {
						role: msg.role,
						content: { type: 'text', text }
					};
				}
				return msg;
			});

			return {
				description: prompt.description,
				messages
			};
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
