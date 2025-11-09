#!/usr/bin/env node
/**
 * MCP Server - stdio transport
 * Generic MCP protocol implementation for stdio-based communication
 *
 * This server provides a transport layer for the Model Context Protocol (MCP)
 * over standard input/output streams. It dynamically loads tools from a module
 * and handles JSON-RPC 2.0 message exchange according to MCP specification.
 *
 * The server is transport-agnostic and can work with any toolset that exports:
 * - TOOLS_DEFINITIONS: Array of MCP tool definitions
 * - Tool functions matching the definitions
 *
 * Configuration is loaded from sandbox-config.json
 *
 * Protocol: MCP 2025-06-18
 * Transport: stdio (standard input/output)
 */

const readline = require('readline');
const config = require('./sandbox-config.json');
const { TOOLS_DEFINITIONS, TOOLS_MAPPING } = require('./lib/sandbox-tools');

/**
 * Handle MCP tool call
 * @param {string} name - Tool name
 * @param {object} args - Tool arguments
 * @returns {Promise<string>} Tool result
 */
async function handleToolCall(name, args) {
	if (TOOLS_MAPPING[name]) {
		return await TOOLS_MAPPING[name](args);
	}
	return `Unknown tool: ${name}`;
}

/**
 * Send JSON-RPC response
 * @param {number|string} id - Request ID
 * @param {object} result - Result object
 * @param {object|null} error - Error object
 */
function sendResponse(id, result, error = null) {
	const response = {
		jsonrpc: '2.0',
		id: id
	};

	if (error) {
		response.error = error;
	} else {
		response.result = result;
	}

	console.log(JSON.stringify(response));
}

/**
 * Send JSON-RPC notification
 * @param {string} method - Method name
 * @param {object} params - Parameters
 */
function sendNotification(method, params) {
	const notification = {
		jsonrpc: '2.0',
		method: method,
		params: params
	};

	console.log(JSON.stringify(notification));
}

/**
 * Handle MCP request
 * @param {object} request - JSON-RPC request
 */
async function handleRequest(request) {
	const { id, method, params } = request;

	try {
		switch (method) {
			case 'initialize':
				sendResponse(id, {
					protocolVersion: config.mcp.protocolVersion,
					capabilities: {
						tools: {}
					},
					serverInfo: {
						name: config.mcp.serverName,
						version: config.mcp.serverVersion
					}
				});
				break;

			case 'initialized':
				// No response needed for notification
				break;

			case 'tools/list':
				sendResponse(id, {
					tools: TOOLS_DEFINITIONS
				});
				break;

			case 'tools/call':
				const toolName = params.name;
				const toolArgs = params.arguments || {};

				const result = await handleToolCall(toolName, toolArgs);

				sendResponse(id, {
					content: [
						{
							type: 'text',
							text: result
						}
					]
				});
				break;

			case 'ping':
				sendResponse(id, {});
				break;

			default:
				sendResponse(id, null, {
					code: -32601,
					message: `Method not found: ${method}`
				});
		}
	} catch (error) {
		sendResponse(id, null, {
			code: -32603,
			message: `Internal error: ${error.message}`
		});
	}
}

/**
 * Main server loop
 */
async function main() {
	console.error(`[Sandbox MCP] Starting server`);
	console.error(`[Sandbox MCP] Protocol version: ${config.mcp.protocolVersion}`);
	console.error(`[Sandbox MCP] Transport: stdio`);
	console.error(
		`[Sandbox MCP] Tools loaded: ${TOOLS_DEFINITIONS.length} from lib/sandbox-tools.js`
	);
	console.error(`[Sandbox MCP] Host user: ${config.podman.hostUser}`);
	console.error(`[Sandbox MCP] Container: ${config.podman.container}`);
	console.error(`[Sandbox MCP] Container user: ${config.podman.containerUser}`);
	console.error(`[Sandbox MCP] Bash output limit: ${config.podman.bashOutputLimitBytes} bytes`);
	console.error(`[Sandbox MCP] Timeout: ${config.podman.timeout}s`);

	const rl = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
		terminal: false
	});

	rl.on('line', async (line) => {
		try {
			const request = JSON.parse(line);
			await handleRequest(request);
		} catch (error) {
			console.error(`[Sandbox MCP] Parse error: ${error.message}`);
		}
	});

	rl.on('close', () => {
		console.error('[Sandbox MCP] Stdin closed, exiting');
		process.exit(0);
	});
}

// Handle process signals
process.on('SIGTERM', () => {
	console.error('[Sandbox MCP] SIGTERM received, exiting');
	process.exit(0);
});

process.on('SIGINT', () => {
	console.error('[Sandbox MCP] SIGINT received, exiting');
	process.exit(0);
});

// Start server
main().catch((error) => {
	console.error(`[Sandbox MCP] Fatal error: ${error.message}`);
	process.exit(1);
});
