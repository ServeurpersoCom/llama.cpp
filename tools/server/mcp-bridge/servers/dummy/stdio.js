#!/usr/bin/env node
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const MCPServer = require('../lib/server-core');
const config = require('./config.json');
const toolsModule = require('./lib/tools');

const mcpServer = new MCPServer(config, toolsModule);
mcpServer.setupSignalHandlers();

const transport = new StdioServerTransport();

console.error('[Dummy MCP] Starting stdio transport');
console.error(`[Dummy MCP] Tools available: ${mcpServer.toolsDefinitions.length}`);

mcpServer.getServer().connect(transport);
