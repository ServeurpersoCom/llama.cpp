#!/usr/bin/env node
const http = require('http');
const { randomUUID } = require('crypto');
const {
	StreamableHTTPServerTransport
} = require('@modelcontextprotocol/sdk/server/streamableHttp.js');
const MCPServer = require('../lib/server-core');
const config = require('./config.json');
const toolsModule = require('./lib/tools');
const { createPromptsModule } = require('./lib/prompts');

const promptsModule = createPromptsModule(config);

const { host, port } = config.streamable_http;

console.error('[Sandbox MCP] Starting Streamable HTTP transport (SDK)');
console.error(`[Sandbox MCP] Listening on http://${host}:${port}`);
console.error(`[Sandbox MCP] Tools available: ${toolsModule.TOOLS_DEFINITIONS.length}`);
console.error(`[Sandbox MCP] Prompts available: ${promptsModule.PROMPTS_DEFINITIONS.length}`);

const transports = {};

const createServer = () => {
	const mcpServer = new MCPServer(config, toolsModule, promptsModule);
	return mcpServer.getServer();
};

const isInitializeRequest = (body) => {
	return body && body.method === 'initialize';
};

const httpServer = http.createServer(async (req, res) => {
	res.setHeader('Access-Control-Allow-Origin', '*');
	res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
	res.setHeader(
		'Access-Control-Allow-Headers',
		'Content-Type, Mcp-Session-Id, MCP-Protocol-Version, Accept'
	);
	res.setHeader('Access-Control-Expose-Headers', 'Mcp-Session-Id');

	if (req.method === 'OPTIONS') {
		res.writeHead(204);
		return res.end();
	}

	if (req.method === 'POST') {
		let body = '';
		req.on('data', (chunk) => (body += chunk.toString()));
		req.on('end', async () => {
			let parsedBody;
			try {
				parsedBody = JSON.parse(body);
			} catch (e) {
				res.writeHead(400, { 'Content-Type': 'application/json' });
				return res.end(JSON.stringify({ error: 'Invalid JSON' }));
			}

			const sessionId = req.headers['mcp-session-id'];
			let transport;

			if (sessionId && transports[sessionId]) {
				transport = transports[sessionId];
			} else if (!sessionId && isInitializeRequest(parsedBody)) {
				transport = new StreamableHTTPServerTransport({
					sessionIdGenerator: () => randomUUID(),
					onsessioninitialized: (sid) => {
						console.error(
							`[Sandbox MCP] Session ${sid} initialized (${Object.keys(transports).length + 1} active)`
						);
						transports[sid] = transport;
					}
				});

				transport.onclose = () => {
					const sid = transport.sessionId;
					if (sid && transports[sid]) {
						console.error(
							`[Sandbox MCP] Session ${sid} closed (${Object.keys(transports).length - 1} active)`
						);
						delete transports[sid];
					}
				};

				const server = createServer();
				await server.connect(transport);
				await transport.handleRequest(req, res, parsedBody);
				return;
			} else {
				res.writeHead(400, { 'Content-Type': 'application/json' });
				return res.end(
					JSON.stringify({
						jsonrpc: '2.0',
						error: { code: -32000, message: 'Bad Request: Mcp-Session-Id required' },
						id: null
					})
				);
			}

			await transport.handleRequest(req, res, parsedBody);
		});
		return;
	}

	if (req.method === 'GET') {
		const sessionId = req.headers['mcp-session-id'];

		if (!sessionId || !transports[sessionId]) {
			res.writeHead(400, { 'Content-Type': 'application/json' });
			return res.end(
				JSON.stringify({
					jsonrpc: '2.0',
					error: { code: -32000, message: 'Bad Request: Invalid or missing session ID' },
					id: null
				})
			);
		}

		await transports[sessionId].handleRequest(req, res);
		return;
	}

	if (req.method === 'DELETE') {
		const sessionId = req.headers['mcp-session-id'];

		if (sessionId && transports[sessionId]) {
			await transports[sessionId].close();
		}

		res.writeHead(200);
		return res.end();
	}

	res.writeHead(405);
	res.end('Method Not Allowed');
});

httpServer.listen(port, host);

httpServer.on('error', (error) => {
	console.error(`[Sandbox MCP] Server error: ${error.message}`);
	process.exit(1);
});

const shutdown = async () => {
	console.error('[Sandbox MCP] Shutting down...');
	for (const sessionId in transports) {
		await transports[sessionId].close();
	}
	process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
