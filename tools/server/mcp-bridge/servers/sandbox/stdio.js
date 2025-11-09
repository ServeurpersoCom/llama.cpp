#!/usr/bin/env node
const readline = require('readline');
const MCPServer = require('../lib/server-core');
const config = require('./config.json');
const toolsModule = require('./lib/tools');

const server = new MCPServer(config, toolsModule);
server.setupSignalHandlers();

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
	terminal: false
});

console.error('[Sandbox MCP] Starting stdio transport');
console.error(`[Sandbox MCP] Tools available: ${server.tools.length}`);

rl.on('line', async (line) => {
	if (!line) {
		return;
	}

	try {
		const request = JSON.parse(line);
		const response = await server.handleRequest(request);
		if (response) {
			console.log(JSON.stringify(response));
		}
	} catch (error) {
		console.error(`[Sandbox MCP] Error parsing or handling request: ${error.message}`);
	}
});

rl.on('close', () => process.exit(0));
