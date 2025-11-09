#!/usr/bin/env node
/**
 * MCP Server - WebSocket transport
 * Generic MCP protocol implementation for WebSocket-based communication
 *
 * This server provides a transport layer for the Model Context Protocol (MCP)
 * over WebSocket connections. It dynamically loads tools from a module and
 * handles JSON-RPC 2.0 message exchange according to MCP specification.
 *
 * The server is transport-agnostic and can work with any toolset that exports:
 * - TOOLS_DEFINITIONS: Array of MCP tool definitions
 * - Tool functions matching the definitions
 *
 * Configuration is loaded from sandbox-config.json
 *
 * Protocol: MCP 2025-06-18
 * Transport: WebSocket (ws://)
 */

const WebSocket = require('ws');
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
 * @param {WebSocket} ws - WebSocket connection
 * @param {number|string} id - Request ID
 * @param {object} result - Result object
 * @param {object|null} error - Error object
 */
function sendResponse(ws, id, result, error = null) {
	const response = {
		jsonrpc: '2.0',
		id: id
	};

	if (error) {
		response.error = error;
	} else {
		response.result = result;
	}

	ws.send(JSON.stringify(response));
}

/**
 * Send JSON-RPC notification
 * @param {WebSocket} ws - WebSocket connection
 * @param {string} method - Method name
 * @param {object} params - Parameters
 */
function sendNotification(ws, method, params) {
	const notification = {
		jsonrpc: '2.0',
		method: method,
		params: params
	};

	ws.send(JSON.stringify(notification));
}

/**
 * Handle MCP request
 * @param {WebSocket} ws - WebSocket connection
 * @param {object} request - JSON-RPC request
 */
async function handleRequest(ws, request) {
	const { id, method, params } = request;

	try {
		switch (method) {
			case 'initialize':
				sendResponse(ws, id, {
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
				sendResponse(ws, id, {
					tools: TOOLS_DEFINITIONS
				});
				break;

			case 'tools/call':
				const toolName = params.name;
				const toolArgs = params.arguments || {};

				const result = await handleToolCall(toolName, toolArgs);

				sendResponse(ws, id, {
					content: [
						{
							type: 'text',
							text: result
						}
					]
				});
				break;

			case 'ping':
				sendResponse(ws, id, {});
				break;

			default:
				sendResponse(ws, id, null, {
					code: -32601,
					message: `Method not found: ${method}`
				});
		}
	} catch (error) {
		sendResponse(ws, id, null, {
			code: -32603,
			message: `Internal error: ${error.message}`
		});
	}
}

/**
 * Main server
 */
async function main() {
	console.error(`[Sandbox MCP] Starting server`);
	console.error(`[Sandbox MCP] Protocol version: ${config.mcp.protocolVersion}`);
	console.error(`[Sandbox MCP] Transport: WebSocket`);
	console.error(
		`[Sandbox MCP] Listening on: ws://${config.websocket.host}:${config.websocket.port}`
	);
	console.error(
		`[Sandbox MCP] Tools loaded: ${TOOLS_DEFINITIONS.length} from lib/sandbox-tools.js`
	);
	console.error(`[Sandbox MCP] Host user: ${config.podman.hostUser}`);
	console.error(`[Sandbox MCP] Container: ${config.podman.container}`);
	console.error(`[Sandbox MCP] Container user: ${config.podman.containerUser}`);
	console.error(`[Sandbox MCP] Bash output limit: ${config.podman.bashOutputLimitBytes} bytes`);
	console.error(`[Sandbox MCP] Timeout: ${config.podman.timeout}s`);

	const wss = new WebSocket.Server({
		host: config.websocket.host,
		port: config.websocket.port
	});

	wss.on('listening', () => {
		console.error(`[Sandbox MCP] Server ready`);
	});

	wss.on('connection', (ws, req) => {
		const clientIP = req.socket.remoteAddress;
		console.error(`[Sandbox MCP] Client connected from ${clientIP}`);

		ws.on('message', async (data) => {
			try {
				const request = JSON.parse(data.toString());
				await handleRequest(ws, request);
			} catch (error) {
				console.error(`[Sandbox MCP] Parse error: ${error.message}`);
				sendResponse(ws, null, null, {
					code: -32700,
					message: 'Parse error'
				});
			}
		});

		ws.on('close', () => {
			console.error(`[Sandbox MCP] Client disconnected from ${clientIP}`);
		});

		ws.on('error', (error) => {
			console.error(`[Sandbox MCP] WebSocket error: ${error.message}`);
		});
	});

	wss.on('error', (error) => {
		console.error(`[Sandbox MCP] Server error: ${error.message}`);
		process.exit(1);
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
