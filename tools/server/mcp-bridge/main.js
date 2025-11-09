#!/usr/bin/env node
/**
 * MCP Bridge entry point
 * Initializes MCP client and starts OpenAI-compatible SSE proxy
 */

const fs = require('fs');
const path = require('path');
const MCPClient = require('./lib/mcp-client.js');
const ProxySSE = require('./lib/sse-proxy.js');

async function main() {
	try {
		// Load configuration
		const configPath = path.join(__dirname, 'config.json');
		const configData = fs.readFileSync(configPath, 'utf8');
		const config = JSON.parse(configData);

		console.log('[Main] Starting MCP Bridge...');
		console.log('[Main] Configuration loaded from:', configPath);

		// Initialize MCP client
		console.log('[Main] Initializing MCP client...');
		const mcpClient = new MCPClient(config.mcp);
		await mcpClient.initialize();

		console.log(`[Main] MCP client initialized with ${mcpClient.listTools().length} tools`);

		// Start SSE proxy
		console.log('[Main] Starting SSE proxy...');
		const proxy = new ProxySSE(mcpClient, config.proxy);
		proxy.start();
	} catch (err) {
		console.error('[Main] Fatal error:', err);
		process.exit(1);
	}
}

main();
