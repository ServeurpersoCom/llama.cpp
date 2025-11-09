#!/usr/bin/env node
/**
 * Dummy MCP Server for testing
 * Implements simple tools with verbose logging
 */

const readline = require('readline');

const TOOLS = [
	{
		name: 'echo',
		description: 'Echo back the input text',
		inputSchema: {
			type: 'object',
			properties: {
				text: {
					type: 'string',
					description: 'Text to echo'
				}
			},
			required: ['text']
		}
	},
	{
		name: 'add',
		description: 'Add two numbers',
		inputSchema: {
			type: 'object',
			properties: {
				a: {
					type: 'number',
					description: 'First number'
				},
				b: {
					type: 'number',
					description: 'Second number'
				}
			},
			required: ['a', 'b']
		}
	}
];

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
	terminal: false
});

function log(message) {
	console.error(`[Dummy] ${message}`);
}

function send(response) {
	console.log(JSON.stringify(response));
}

rl.on('line', (line) => {
	try {
		const request = JSON.parse(line);
		log(`Request: ${request.method} id:${request.id}`);

		if (request.method === 'initialize') {
			send({
				jsonrpc: '2.0',
				id: request.id,
				result: {
					protocolVersion: '2025-06-18',
					capabilities: { tools: { listChanged: true } },
					serverInfo: { name: 'dummy-mcp', version: '1.0.0' }
				}
			});
		} else if (request.method === 'tools/list') {
			send({
				jsonrpc: '2.0',
				id: request.id,
				result: { tools: TOOLS }
			});
		} else if (request.method === 'tools/call') {
			const { name, arguments: args } = request.params;
			log(`Executing: ${name}(${JSON.stringify(args)})`);

			let result;

			if (name === 'echo') {
				result = {
					content: [{ type: 'text', text: `ECHO: ${args.text}` }]
				};
			} else if (name === 'add') {
				const sum = args.a + args.b;
				result = {
					content: [{ type: 'text', text: `${args.a} + ${args.b} = ${sum}` }]
				};
			} else {
				send({
					jsonrpc: '2.0',
					id: request.id,
					error: {
						code: -32601,
						message: `Unknown tool: ${name}`
					}
				});
				return;
			}

			send({
				jsonrpc: '2.0',
				id: request.id,
				result: result
			});
		} else {
			send({
				jsonrpc: '2.0',
				id: request.id,
				error: {
					code: -32601,
					message: `Unknown method: ${request.method}`
				}
			});
		}
	} catch (e) {
		log(`ERROR: ${e.message}`);
	}
});

log('Ready');
