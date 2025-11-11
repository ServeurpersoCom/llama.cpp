#!/usr/bin/env node
const WebSocket = require('ws');
const MCPServer = require('../lib/server-core');
const config = require('./config.json');
const toolsModule = require('./lib/tools');

const server = new MCPServer(config, toolsModule);
server.setupSignalHandlers();

const { host, port } = config.websocket;

console.error('[Sandbox MCP] Starting WebSocket transport');
console.error(`[Sandbox MCP] Listening on ws://${host}:${port}`);
console.error(`[Sandbox MCP] Tools available: ${server.tools.length}`);

const wss = new WebSocket.Server({ host, port });

wss.on('connection', (ws, req) => {
	const clientIP = req.socket.remoteAddress;
	console.error(`[Sandbox MCP] Client connected from ${clientIP}`);

	ws.on('message', async (data) => {
		try {
			const request = JSON.parse(data.toString());
			const response = await server.handleRequest(request);
			if (response) {
				ws.send(JSON.stringify(response));
			}
		} catch (error) {
			console.error(`[Sandbox MCP] Failed to process message: ${error.message}`);
			ws.send(
				JSON.stringify({
					jsonrpc: '2.0',
					id: null,
					error: { code: -32700, message: 'Parse error' }
				})
			);
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
