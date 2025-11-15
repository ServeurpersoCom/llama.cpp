/**
 * OpenAI-compatible SSE proxy server
 * Handles HTTP requests, streaming, tool_calls aggregation, and agentic loops
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

class ProxySSE {
	constructor(mcpClient, config) {
		if (!config) {
			throw new Error('ProxySSE requires config parameter');
		}
		if (!config.apiUrl) {
			throw new Error('config.apiUrl is required');
		}
		if (!config.port) {
			throw new Error('config.port is required');
		}
		if (!config.maxTurns) {
			throw new Error('config.maxTurns is required');
		}

		this.mcpClient = mcpClient;
		this.apiUrl = config.apiUrl;
		this.port = config.port;
		this.maxTurns = config.maxTurns;
		this.maxLinesForToolResponsePreview = config.maxLinesForToolResponsePreview;
		this.server = null;

		this.loggingConfig = config.logging || {};

		// System Prompt Profiles configuration
		this.systemPromptProfiles = {
			enabled: config.systemPromptProfiles || false,
			passwords: config.systemPromptPasswords || [],
			files: config.systemPromptFiles || []
		};

		// Validate system prompt profiles configuration
		if (this.systemPromptProfiles.enabled) {
			if (
				this.systemPromptProfiles.passwords.length !==
				this.systemPromptProfiles.files.length
			) {
				throw new Error(
					'systemPromptPasswords and systemPromptFiles must have same length'
				);
			}
			this._log('[Profiles] System Prompt Profiles enabled');
			this._log(
				`[Profiles] ${this.systemPromptProfiles.passwords.length} profile(s) configured`
			);
		}

		this._setupErrorHandler();
	}

	_setupErrorHandler() {
		// System error handler: filters only known undici bugs during brutal disconnections
		// This handler is necessary because undici (Node's fetch engine) rejects internal promises
		// asynchronously after our code has already handled the error properly
		process.on('unhandledRejection', (reason) => {
			// Strict filtering: only undici socket errors during disconnections
			// The UND_ERR_SOCKET code is in the cause.code property, not directly in reason.code
			if (reason?.message === 'terminated' && reason?.cause?.code === 'UND_ERR_SOCKET') {
				return; // Expected behavior, already handled locally
			}
			// All other errors are real bugs to investigate
			this._log(`UNHANDLED ERROR: ${reason}`);
		});
	}

	/**
	 * Detect system prompt profile activation by password
	 * If system prompt content EXACTLY matches a password, replace it with profile file
	 *
	 * @param {Array} messages - Original messages array
	 * @returns {Object} { activated: boolean, profileIndex: number|null, cleanedMessages: Array }
	 */
	_detectProfileActivation(messages) {
		if (!this.systemPromptProfiles.enabled || messages.length === 0) {
			return { activated: false, profileIndex: null, cleanedMessages: messages };
		}

		// Find system message
		const systemMessage = messages.find((msg) => msg.role === 'system');
		if (!systemMessage) {
			return { activated: false, profileIndex: null, cleanedMessages: messages };
		}

		const systemContent = (systemMessage.content || '').trim();

		// Check for EXACT password match in system prompt (case-sensitive)
		for (let i = 0; i < this.systemPromptProfiles.passwords.length; i++) {
			const password = this.systemPromptProfiles.passwords[i];

			if (systemContent === password) {
				this._log(
					`[Profiles] System prompt password matched: "${password}" -> activating MCP`
				);
				return { activated: true, profileIndex: i, cleanedMessages: messages };
			}
		}

		return { activated: false, profileIndex: null, cleanedMessages: messages };
	}

	/**
	 * Load system prompt from file
	 * @param {string} filePath - Path to system prompt file
	 * @returns {string} System prompt content (empty string if file not found)
	 */
	_loadSystemPrompt(filePath) {
		try {
			// Resolve path relative to project root
			const fullPath = path.join(process.cwd(), filePath);
			const content = fs.readFileSync(fullPath, 'utf8');
			this._log(`[Profiles] Loaded system prompt from: ${filePath}`);
			return content;
		} catch (error) {
			// Log error clearly to stderr, return empty string (non-blocking)
			console.error(
				`[Proxy] [Profiles] ERROR: Failed to load system prompt from ${filePath}: ${error.message}`
			);
			console.error(`[Proxy] [Profiles] System prompt will be empty (file not found)`);
			return '';
		}
	}

	/**
	 * Replace system prompt in messages array
	 * @param {Array} messages - Messages array
	 * @param {string} newSystemPrompt - New system prompt content
	 * @returns {Array} Modified messages array
	 */
	_replaceSystemPrompt(messages, newSystemPrompt) {
		// Find system message index
		const systemIndex = messages.findIndex((msg) => msg.role === 'system');

		if (systemIndex !== -1) {
			// Replace existing system prompt
			return messages.map((msg, idx) => {
				if (idx === systemIndex) {
					return { role: 'system', content: newSystemPrompt };
				}
				return msg;
			});
		} else {
			// Prepend system prompt if not present
			return [{ role: 'system', content: newSystemPrompt }, ...messages];
		}
	}

	_log(message) {
		if (!this.loggingConfig.console) return;
		const timestamp = new Date().toISOString().replace('T', ' ').substr(0, 19);
		console.log(`[${timestamp}] ${message}`);
	}

	_logServerToProxy(data) {
		if (!this.loggingConfig.serverToProxy) return;
		const filePath = path.join(this.loggingConfig.debugDir, 'debug-server-to-proxy.jsonl');
		fs.appendFileSync(filePath, `${JSON.stringify(data)}\n`);
	}

	_logProxyToClient(data) {
		if (!this.loggingConfig.proxyToClient) return;
		const filePath = path.join(this.loggingConfig.debugDir, 'debug-proxy-to-client.jsonl');
		fs.appendFileSync(filePath, `${JSON.stringify(data)}\n`);
	}

	_resetDebugLogs() {
		if (!this.loggingConfig.resetOnFirstTurn) return;

		if (this.loggingConfig.serverToProxy) {
			const filePath = path.join(this.loggingConfig.debugDir, 'debug-server-to-proxy.jsonl');
			fs.writeFileSync(filePath, '');
		}

		if (this.loggingConfig.proxyToClient) {
			const filePath = path.join(this.loggingConfig.debugDir, 'debug-proxy-to-client.jsonl');
			fs.writeFileSync(filePath, '');
		}
	}

	start() {
		this.server = http.createServer(this._handleRequest.bind(this));
		this.server.listen(this.port, () => {
			this._log(`Proxy listening on port ${this.port}`);
			this._log(`Endpoint: http://localhost:${this.port}/v1/chat/completions`);
			this._log(`Upstream LLM: ${this.apiUrl}`);
			this._log(`Max turns: ${this.maxTurns}`);
			this._log(`Available tools: ${this.mcpClient.listTools().length}`);
			this._log(`Console logging: ${this.loggingConfig.console}`);
			this._log(
				`Debug server <-> proxy: ${this.loggingConfig.serverToProxy} (debug-server-to-proxy.jsonl)`
			);
			this._log(
				`Debug proxy <-> client: ${this.loggingConfig.proxyToClient} (debug-proxy-to-client.jsonl)`
			);
			this._log(`Reset on first turn: ${this.loggingConfig.resetOnFirstTurn}`);
			this._log(
				`Filter client reasoning after first turn: ${this.loggingConfig.filterClientReasoningAfterFirstTurn}`
			);

			// Log profiles status
			if (this.systemPromptProfiles.enabled) {
				this._log('[Profiles] System Prompt Profiles: ENABLED');
				this._log(
					`[Profiles] Passwords: [${this.systemPromptProfiles.passwords.map((p) => `"${p}"`).join(', ')}]`
				);
				this._log('[Profiles] MCP will activate only when password is detected');
			} else {
				this._log('[Profiles] System Prompt Profiles: DISABLED (MCP always active)');
			}
		});
	}

	async _handleRequest(req, res) {
		// CORS preflight: accept OPTIONS for cross-origin POST
		if (req.method === 'OPTIONS') {
			res.writeHead(204, {
				'Access-Control-Allow-Origin': '*',
				'Access-Control-Allow-Headers': 'Content-Type, Authorization',
				'Access-Control-Allow-Methods': 'POST, OPTIONS'
			});
			return res.end();
		}

		// Passthrough /v1/models for model list
		if (req.method === 'GET' && req.url && req.url.split('?')[0] === '/v1/models') {
			this._log('Passthrough /v1/models');
			try {
				const response = await fetch(`${this.apiUrl}${req.url}`, {
					headers: {
						Authorization: req.headers['authorization'] || '',
						'User-Agent': 'llama-mcp-proxy/1.0'
					}
				});
				const status = response.status;
				const contentType = response.headers.get('content-type') || 'application/json';
				const payload = await response.text();
				res.writeHead(status, {
					'Content-Type': contentType,
					'Access-Control-Allow-Origin': '*'
				});
				res.end(payload);
			} catch (e) {
				res.writeHead(502, {
					'Content-Type': 'application/json',
					'Access-Control-Allow-Origin': '*'
				});
				res.end(JSON.stringify({ error: 'Bad gateway' }));
			}
			return;
		}

		// Minimal routing: only POST to /v1*
		if (req.method !== 'POST' || !req.url.startsWith('/v1')) {
			res.writeHead(404);
			res.end('Not Found');
			return;
		}

		// Reset debug logs for new session if configured
		this._resetDebugLogs();

		// Accumulate incoming HTTP body
		let body = '';
		req.on('data', (chunk) => {
			body += chunk.toString();
		});

		// End of reading: parse JSON, open SSE, and launch loop
		req.on('end', async () => {
			await this._handleChatCompletion(req, res, body);
		});
	}

	async _handleChatCompletion(req, res, body) {
		let requestData;
		try {
			requestData = JSON.parse(body);
		} catch (e) {
			res.writeHead(400, { 'Content-Type': 'application/json' });
			const errorResponse = { error: 'Invalid request' };
			res.end(JSON.stringify(errorResponse));
			this._logProxyToClient(errorResponse);
			return;
		}

		// Log initial client request
		this._logProxyToClient(requestData);

		// Open SSE stream
		res.writeHead(200, {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			Connection: 'keep-alive',
			'Access-Control-Allow-Origin': '*'
		});

		// Session state / flow controls
		const sessionState = {
			clientDisconnected: false,
			currentReader: null
		};

		// Stop button: clean stop if client cuts SSE connection
		res.on('close', () => {
			sessionState.clientDisconnected = true;
			if (sessionState.currentReader) {
				try {
					sessionState.currentReader.cancel();
				} catch (e) {}
			}
			this._log('Client disconnected, stopping streaming');
		});

		// Detect profile activation
		const profileActivation = this._detectProfileActivation(requestData.messages || []);

		let toolsToUse = [];
		let sessionMessages = [...(requestData.messages || [])];

		if (this.systemPromptProfiles.enabled && !profileActivation.activated) {
			// Pass-through mode: no MCP, forward as-is
			this._log(
				'[Profiles] Pass-through mode: system prompt is not a password, MCP disabled'
			);
		} else {
			// MCP active mode
			if (profileActivation.activated) {
				// Profile activated: replace system prompt with file content
				const profileFile = this.systemPromptProfiles.files[profileActivation.profileIndex];
				const systemPrompt = this._loadSystemPrompt(profileFile);

				sessionMessages = this._replaceSystemPrompt(sessionMessages, systemPrompt);

				this._log(
					`[Profiles] System prompt replaced with: ${profileFile.split('/').pop()}`
				);
			}

			// Get available tools from MCP discovery
			toolsToUse = await this.mcpClient.getToolsDefinition();
			this._log(`${toolsToUse.length} tools available via MCP`);
		}

		// Multi-turn agent loop
		let currentTurn = 0;
		while (currentTurn < this.maxTurns) {
			if (sessionState.clientDisconnected) {
				this._log('Stopping processing due to client disconnection');
				break;
			}

			// Build request to upstream LLM
			const llmRequest = {
				...requestData,
				messages: sessionMessages,
				stream: true
			};

			// Inject tools definition from MCP discovery (if MCP active)
			if (toolsToUse && toolsToUse.length > 0) {
				llmRequest.tools = toolsToUse;
				this._log('Tools definition injected into LLM request');
			}

			// Log outgoing request to LLM server
			this._logServerToProxy(llmRequest);
			this._log(`Turn ${currentTurn + 1}/${this.maxTurns}`);

			try {
				const turnResult = await this._processTurn(
					req,
					res,
					llmRequest,
					sessionMessages,
					sessionState,
					currentTurn
				);

				sessionState.currentReader = null;

				if (turnResult.done || sessionState.clientDisconnected) {
					break;
				}

				currentTurn++;
			} catch (error) {
				// Handle LLM backend communication errors
				this._log(`Error during LLM call: ${error.message}`);

				let errorDetails = error.message;
				if (error.cause) {
					errorDetails += `\nCause: ${JSON.stringify(error.cause)}`;
				}

				const errorChunk = {
					choices: [
						{
							delta: {
								content: `\n\nUpstream LLM error:\n\`\`\`\n${errorDetails}\n\`\`\`\n`
							},
							finish_reason: 'stop'
						}
					],
					model: llmRequest.model
				};
				res.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
				this._logProxyToClient(errorChunk);
				res.write('data: [DONE]\n\n');
				this._logProxyToClient('[DONE]');
				break;
			}

			if (sessionState.clientDisconnected) {
				this._log('Stopping processing due to client disconnection');
				break;
			}
		}

		// Session closure if turn limit reached
		if (currentTurn >= this.maxTurns) {
			this._log('Turn limit reached');
			const warningChunk = {
				choices: [
					{
						delta: { content: '\n\nTurn limit reached' }
					}
				]
			};
			res.write(`data: ${JSON.stringify(warningChunk)}\n\n`);
			this._logProxyToClient(warningChunk);
			res.write('data: [DONE]\n\n');
			this._logProxyToClient('[DONE]');
		}

		// End of session: close SSE stream
		this._log('End of session');
		res.end();
	}

	async _processTurn(req, res, llmRequest, sessionMessages, sessionState, turnIndex) {
		// Send request to LLM server
		const response = await fetch(`${this.apiUrl}${req.url}`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Authorization: req.headers['authorization'] || '',
				'User-Agent': 'llama-mcp-proxy/1.0'
			},
			body: JSON.stringify(llmRequest)
		});

		if (!response.ok) {
			const errorBody = await response.text();
			this._log(`LLM returned ${response.status}: ${errorBody}`);
			throw new Error(`HTTP ${response.status}: ${errorBody}`);
		}

		// Read incoming SSE stream
		const reader = response.body.getReader();
		sessionState.currentReader = reader;
		const decoder = new TextDecoder('utf-8');
		let buffer = '';

		// Accumulation state for current turn
		let accumulatedContent = '';
		const toolBuffers = {};
		const toolIndexMap = {};
		let sawToolDelta = false;
		let toolCallsReady = false;

		// SSE stream reading loop
		while (true) {
			if (sessionState.clientDisconnected) {
				try {
					reader.cancel();
				} catch (e) {}
				break;
			}

			const { done, value } = await this._readStreamSecure(reader);

			if (done) break;

			buffer += decoder.decode(value, { stream: true });

			// Extract complete lines "data: ...\n\n"
			const lines = buffer.split('\n\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				if (!line.startsWith('data: ')) continue;
				const dataStr = line.substring(6);
				if (!dataStr || dataStr === '[DONE]') {
					if (dataStr === '[DONE]') {
						this._logServerToProxy('[DONE]');
					}
					continue;
				}

				try {
					const data = JSON.parse(dataStr);

					// Log incoming chunk from LLM server
					this._logServerToProxy(data);

					// Accumulate text content
					if (
						data.choices &&
						data.choices[0] &&
						data.choices[0].delta &&
						data.choices[0].delta.content
					) {
						accumulatedContent += data.choices[0].delta.content;
					}

					// Remove reasoning content after first turn if configured
					if (
						this.loggingConfig.filterClientReasoningAfterFirstTurn &&
						turnIndex > 0 &&
						data.choices &&
						data.choices[0] &&
						data.choices[0].delta &&
						data.choices[0].delta.reasoning_content !== undefined
					) {
						delete data.choices[0].delta.reasoning_content;
					}

					// Accumulate streamed tool_calls
					if (
						data.choices &&
						data.choices[0] &&
						data.choices[0].delta &&
						data.choices[0].delta.tool_calls
					) {
						sawToolDelta = true;
						this._accumulateToolCallsDelta(
							toolBuffers,
							toolIndexMap,
							data.choices[0].delta.tool_calls
						);
					}

					// Detect finish_reason: tool_calls
					if (
						data.choices &&
						data.choices[0] &&
						typeof data.choices[0].finish_reason === 'string'
					) {
						if (data.choices[0].finish_reason === 'tool_calls') {
							toolCallsReady = true;
						}
					}

					res.write(`data: ${JSON.stringify(data)}\n\n`);
					this._logProxyToClient(data);
				} catch (e) {}
			}
		}

		// Handle last fragment remaining in buffer
		if (buffer && buffer.startsWith('data: ')) {
			const dataStr = buffer.substring(6);
			if (dataStr && dataStr !== '[DONE]') {
				try {
					const data = JSON.parse(dataStr);

					// Log incoming chunk from LLM server
					this._logServerToProxy(data);

					if (
						data.choices &&
						data.choices[0] &&
						data.choices[0].delta &&
						data.choices[0].delta.content
					) {
						accumulatedContent += data.choices[0].delta.content;
					}
					if (
						data.choices &&
						data.choices[0] &&
						data.choices[0].delta &&
						data.choices[0].delta.tool_calls
					) {
						sawToolDelta = true;
						this._accumulateToolCallsDelta(
							toolBuffers,
							toolIndexMap,
							data.choices[0].delta.tool_calls
						);
					}
					if (
						data.choices &&
						data.choices[0] &&
						typeof data.choices[0].finish_reason === 'string'
					) {
						if (data.choices[0].finish_reason === 'tool_calls') {
							toolCallsReady = true;
						}
					}
					if (
						this.loggingConfig.filterClientReasoningAfterFirstTurn &&
						turnIndex > 0 &&
						data.choices &&
						data.choices[0] &&
						data.choices[0].delta &&
						data.choices[0].delta.reasoning_content !== undefined
					) {
						delete data.choices[0].delta.reasoning_content;
					}

					res.write(`data: ${JSON.stringify(data)}\n\n`);
					this._logProxyToClient(data);
				} catch (e) {}
			} else if (dataStr === '[DONE]') {
				// Log [DONE] from LLM server
				this._logServerToProxy('[DONE]');
				res.write('data: [DONE]\n\n');
				this._logProxyToClient('[DONE]');
			}
		}

		this._log('Accumulated content updated');
		const cleanContent = accumulatedContent.trim();

		if (sessionState.clientDisconnected) {
			sessionState.currentReader = null;
			return { done: true, reader: null };
		}

		// Build tool_calls from streamed buffers
		const toolCalls = [];
		if (toolCallsReady && sawToolDelta && Object.keys(toolBuffers).length > 0) {
			this._log('Complete tool calls detected (finish_reason: tool_calls)');
			for (const [id, buf] of Object.entries(toolBuffers)) {
				const name = buf.name.trim();
				const argsStr = buf.args || '';
				if (name === '') continue;
				let decoded;
				try {
					const safe = argsStr.trim() === '' ? '{}' : argsStr;
					decoded = JSON.parse(safe);
				} catch (e) {
					decoded = argsStr;
				}
				const argumentsVal = decoded !== null ? decoded : argsStr;
				toolCalls.push({
					id: id,
					type: 'function',
					function: {
						name: name,
						arguments: argumentsVal
					}
				});
			}
		} else {
			if (sawToolDelta) {
				this._log('Tool deltas seen but without finish_reason: tool_calls');
			} else {
				this._log('No tool delta detected');
			}
		}

		// No tool calls: end conversation
		if (toolCalls.length === 0) {
			this._log('No tool calls, end of conversation');
			res.write('data: [DONE]\n\n');
			this._logProxyToClient('[DONE]');
			return { done: true, reader: null };
		}

		// Inject assistant message with tool_calls into context
		sessionMessages.push({
			role: 'assistant',
			content: cleanContent || null,
			tool_calls: toolCalls.map((tc) => ({
				id: tc.id,
				type: 'function',
				function: {
					name: tc.function.name,
					arguments:
						typeof tc.function.arguments === 'string'
							? tc.function.arguments
							: JSON.stringify(tc.function.arguments)
				}
			}))
		});
		this._log('Assistant response (with tool_calls) added to context');

		// Execute tools sequentially and send previews to client
		const allToolResults = [];
		for (const toolCall of toolCalls) {
			const functionName = toolCall.function.name;

			const toolResult = await this._executeTool(toolCall);

			allToolResults.push({
				id: toolCall.id,
				name: functionName,
				result: toolResult
			});

			const isImagePreview = this._isDataImage(toolResult);
			let preview;

			if (isImagePreview) {
				preview = toolResult.trim();
			} else {
				const lines = toolResult.split('\n');
				preview =
					lines.length > this.maxLinesForToolResponsePreview
						? lines.slice(-this.maxLinesForToolResponsePreview).join('\n')
						: toolResult;
			}

			const resultChunk = {
				choices: [
					{
						delta: {
							content: isImagePreview
								? `\n![image](${preview})\n`
								: `\n\`\`\`\n${preview}\n\`\`\`\n`
						},
						finish_reason: null
					}
				],
				model: llmRequest.model
			};
			res.write(`data: ${JSON.stringify(resultChunk)}\n\n`);
			this._logProxyToClient(resultChunk);
		}

		// Inject tool results into context for next turn
		for (const result of allToolResults) {
			let content =
				typeof result.result === 'string' ? result.result : JSON.stringify(result.result);

			// Don't inject base64 images into LLM context (too large)
			if (this._isDataImage(content)) {
				content = '[Image displayed to user]';
			}

			sessionMessages.push({
				role: 'tool',
				tool_call_id: result.id,
				content: content
			});
		}
		this._log('Tool results added to context');

		sessionState.currentReader = null;

		if (sessionState.clientDisconnected) {
			return { done: true, reader: null };
		}

		return { done: false, reader: null };
	}

	async _executeTool(toolCall) {
		const functionName = toolCall.function.name;
		this._log(`Executing tool "${functionName}" requested by model`);

		try {
			const result = await this.mcpClient.execute(toolCall);
			this._log('Tool result retrieved');
			return result;
		} catch (e) {
			this._log(`MCP tool call failed: ${e.message}`);
			return `Error: ${e.message}`;
		}
	}

	_isDataImage(content) {
		if (typeof content !== 'string') {
			return false;
		}

		const trimmed = content.trim();
		return /^data:image\/(png|jpe?g|gif|webp);base64,[a-zA-Z0-9+/=]+$/.test(trimmed);
	}

	async _readStreamSecure(reader) {
		try {
			return await reader.read().catch((err) => {
				// Undici rejects with "terminated" when socket closes brutally
				if (err.message === 'terminated') {
					return { done: true, value: undefined };
				}
				throw err;
			});
		} catch (err) {
			// Additional safety layer for edge cases
			if (err.message === 'terminated') {
				return { done: true, value: undefined };
			}
			throw err;
		}
	}

	_accumulateToolCallsDelta(toolBuffers, toolIndexMap, toolCallsDelta) {
		toolCallsDelta.forEach((tc) => {
			if (tc.index !== undefined && tc.id && tc.id !== '') {
				const idx = parseInt(tc.index);
				toolIndexMap[idx] = tc.id;
				const idxKey = `i${idx}`;
				if (toolBuffers[idxKey] && !toolBuffers[tc.id]) {
					toolBuffers[tc.id] = toolBuffers[idxKey];
					delete toolBuffers[idxKey];
				}
			}
			let key = null;
			if (tc.id && tc.id !== '') {
				key = tc.id;
			} else if (tc.index !== undefined && toolIndexMap[parseInt(tc.index)]) {
				key = toolIndexMap[parseInt(tc.index)];
			} else if (tc.index !== undefined) {
				key = `i${parseInt(tc.index)}`;
			} else {
				return;
			}
			if (!toolBuffers[key]) {
				toolBuffers[key] = {
					name: '',
					args: ''
				};
			}
			if (tc.function && tc.function.name) {
				toolBuffers[key].name = tc.function.name;
			}
			if (tc.function && tc.function.arguments !== undefined) {
				const argDelta = tc.function.arguments;
				toolBuffers[key].args +=
					typeof argDelta === 'string' ? argDelta : JSON.stringify(argDelta);
			} else if (tc.arguments !== undefined) {
				const argDelta = tc.arguments;
				toolBuffers[key].args +=
					typeof argDelta === 'string' ? argDelta : JSON.stringify(argDelta);
			}
		});
	}
}

module.exports = ProxySSE;
