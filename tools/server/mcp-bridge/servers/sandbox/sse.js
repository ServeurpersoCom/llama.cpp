#!/usr/bin/env node
/**
 * SSE Transport for MCP Sandbox Server
 * Implements HTTP POST (client->server) + SSE GET (server->client) transport
 */

const http = require('http');
const MCPServer = require('../lib/server-core');
const config = require('./config.json');
const toolsModule = require('./lib/tools');

const server = new MCPServer(config, toolsModule);
server.setupSignalHandlers();

const { host, port } = config.sse;

// Session management
const sessions = new Map(); // sessionId -> { res, pendingNotifications }

function generateSessionId() {
	return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

console.error('[Sandbox MCP] Starting SSE transport');
console.error(`[Sandbox MCP] Listening on http://${host}:${port}`);
console.error(`[Sandbox MCP] Tools available: ${server.tools.length}`);

const httpServer = http.createServer(async (req, res) => {
	// CORS headers
	const corsHeaders = {
		'Access-Control-Allow-Origin': '*',
		'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
		'Access-Control-Allow-Headers': 'Content-Type, Mcp-Session-Id, MCP-Protocol-Version'
	};

	// OPTIONS preflight
	if (req.method === 'OPTIONS') {
		res.writeHead(204, corsHeaders);
		return res.end();
	}

	const sessionId = req.headers['mcp-session-id'];

	// GET: Open SSE stream for server->client notifications
	if (req.method === 'GET') {
		if (!sessionId || !sessions.has(sessionId)) {
			res.writeHead(404, corsHeaders);
			return res.end('Session not found');
		}

		const session = sessions.get(sessionId);

		res.writeHead(200, {
			...corsHeaders,
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			Connection: 'keep-alive'
		});

		session.res = res;

		// Send pending notifications
		while (session.pendingNotifications.length > 0) {
			const notification = session.pendingNotifications.shift();
			res.write(`data: ${JSON.stringify(notification)}\n\n`);
		}

		req.on('close', () => {
			session.res = null;
			console.error(`[Sandbox MCP] SSE stream closed for session ${sessionId}`);
		});

		return;
	}

	// DELETE: Close session
	if (req.method === 'DELETE') {
		if (sessionId && sessions.has(sessionId)) {
			const session = sessions.get(sessionId);
			if (session.res) {
				session.res.end();
			}
			sessions.delete(sessionId);
			console.error(`[Sandbox MCP] Session ${sessionId} deleted`);
		}
		res.writeHead(200, corsHeaders);
		return res.end();
	}

	// POST: Handle MCP requests/notifications/responses
	if (req.method === 'POST') {
		let body = '';
		req.on('data', (chunk) => (body += chunk.toString()));
		req.on('end', async () => {
			let message;
			try {
				message = JSON.parse(body);
			} catch (e) {
				res.writeHead(400, { ...corsHeaders, 'Content-Type': 'application/json' });
				return res.end(JSON.stringify({ error: 'Invalid JSON' }));
			}

			// Handle initialize specially (create session)
			if (message.method === 'initialize') {
				const newSessionId = generateSessionId();
				sessions.set(newSessionId, {
					res: null,
					pendingNotifications: []
				});

				const response = await server.handleRequest(message);

				res.writeHead(200, {
					...corsHeaders,
					'Content-Type': 'application/json',
					'Mcp-Session-Id': newSessionId,
					'MCP-Protocol-Version': config.mcp.protocolVersion
				});

				console.error(`[Sandbox MCP] Session ${newSessionId} created`);
				return res.end(JSON.stringify(response));
			}

			// For other messages, require session
			if (!sessionId) {
				res.writeHead(400, { ...corsHeaders, 'Content-Type': 'application/json' });
				return res.end(JSON.stringify({ error: 'Mcp-Session-Id header required' }));
			}

			if (!sessions.has(sessionId)) {
				res.writeHead(404, { ...corsHeaders, 'Content-Type': 'application/json' });
				return res.end(JSON.stringify({ error: 'Session not found' }));
			}

			// Handle notification (no response expected)
			if (!message.id) {
				await server.handleRequest(message);
				res.writeHead(202, {
					...corsHeaders,
					'MCP-Protocol-Version': config.mcp.protocolVersion
				});
				return res.end();
			}

			// Handle request (response expected)
			const response = await server.handleRequest(message);

			if (response) {
				res.writeHead(200, {
					...corsHeaders,
					'Content-Type': 'application/json',
					'MCP-Protocol-Version': config.mcp.protocolVersion
				});
				return res.end(JSON.stringify(response));
			} else {
				res.writeHead(202, {
					...corsHeaders,
					'MCP-Protocol-Version': config.mcp.protocolVersion
				});
				return res.end();
			}
		});

		return;
	}

	// Unknown method
	res.writeHead(405, corsHeaders);
	res.end('Method Not Allowed');
});

httpServer.listen(port, host, () => {
	console.error(`[Sandbox MCP] SSE server listening on http://${host}:${port}`);
});

httpServer.on('error', (error) => {
	console.error(`[Sandbox MCP] Server error: ${error.message}`);
	process.exit(1);
});
