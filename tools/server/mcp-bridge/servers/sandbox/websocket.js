#!/usr/bin/env node
const WebSocket = require('ws');
const MCPServer = require('../lib/server-core');
const config = require('./config.json');
const toolsModule = require('./lib/tools');
const { createPromptsModule } = require('./lib/prompts');

const promptsModule = createPromptsModule(config);

const { host, port } = config.websocket;

console.error('[Sandbox MCP] Starting WebSocket transport');
console.error(`[Sandbox MCP] Listening on ws://${host}:${port}`);
console.error(`[Sandbox MCP] Tools available: ${toolsModule.TOOLS_DEFINITIONS.length}`);
console.error(`[Sandbox MCP] Prompts available: ${promptsModule.PROMPTS_DEFINITIONS.length}`);

const wss = new WebSocket.Server({ host, port });

let connectionCount = 0;

wss.on('connection', async (ws, req) => {
	const clientIP = req.socket.remoteAddress;
	const connectionId = ++connectionCount;

	console.error(
		`[Sandbox MCP] Connection ${connectionId} from ${clientIP} (${wss.clients.size} active)`
	);

	const mcpServer = new MCPServer(config, toolsModule, promptsModule);
	const server = mcpServer.getServer();

	const transport = {
		async start() {
			return Promise.resolve();
		},

		async send(message) {
			if (ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify(message));
			}
		},

		async close() {
			ws.close();
		},

		onclose: null,
		onerror: null,
		onmessage: null
	};

	await transport.start();
	server.connect(transport);

	ws.on('message', (data) => {
		try {
			const message = JSON.parse(data.toString());

			if (transport.onmessage) {
				transport.onmessage(message);
			}
		} catch (error) {
			console.error(`[Sandbox MCP] Connection ${connectionId} parse error: ${error.message}`);

			if (ws.readyState === WebSocket.OPEN) {
				ws.send(
					JSON.stringify({
						jsonrpc: '2.0',
						error: { code: -32700, message: 'Parse error' },
						id: null
					})
				);
			}
		}
	});

	ws.on('close', (code) => {
		console.error(
			`[Sandbox MCP] Connection ${connectionId} closed (code=${code}, ${wss.clients.size} active)`
		);

		if (transport.onclose) {
			transport.onclose();
		}
	});

	ws.on('error', (error) => {
		console.error(`[Sandbox MCP] Connection ${connectionId} error: ${error.message}`);

		if (transport.onerror) {
			transport.onerror(error);
		}
	});

	ws.on('ping', () => {
		ws.pong();
	});
});

wss.on('error', (error) => {
	console.error(`[Sandbox MCP] Server error: ${error.message}`);
	process.exit(1);
});

const shutdown = () => {
	console.error('[Sandbox MCP] Shutting down...');

	wss.clients.forEach((ws) => {
		ws.close(1001, 'Server shutting down');
	});

	wss.close(() => {
		console.error('[Sandbox MCP] Server closed');
		process.exit(0);
	});
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
