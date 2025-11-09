/**
 * TransportStreamableHTTP: JSON-RPC over HTTP POST + Streamable HTTP (SSE) notifications
 */

const DEFAULT_PROTOCOL_VERSION = '2025-06-18';

class TransportStreamableHTTP {
	constructor(config) {
		if (!config || !config.url) {
			throw new Error('TransportStreamableHTTP requires config.url');
		}

		this.config = config;
		this.baseUrl = config.url;
		this.protocolVersion = config.protocolVersion || DEFAULT_PROTOCOL_VERSION;
		this.fetchImpl =
			config.fetchImpl || (typeof fetch === 'function' ? fetch.bind(globalThis) : null);

		if (!this.fetchImpl) {
			throw new Error(
				'TransportStreamableHTTP requires a global fetch implementation (Node.js 18+)'
			);
		}

		this.messageHandler = null;
		this.sessionId = null;
		this.started = false;
		this.isStopping = false;
		this.streamAbortController = null;
		this.streamReconnectTimer = null;
	}

	async start() {
		if (this.started) {
			return;
		}

		this.started = true;
		this.isStopping = false;
	}

	onMessage(handler) {
		this.messageHandler = handler;
	}

	send(message) {
		if (!this.started) {
			throw new Error('Transport not started');
		}

		if (!message || typeof message !== 'object') {
			throw new Error('Invalid JSON-RPC message');
		}

		this._postMessage(message);
	}

	async stop() {
		this.isStopping = true;
		this.started = false;
		this._stopStream();

		if (this.sessionId) {
			try {
				await this.fetchImpl(this.baseUrl, {
					method: 'DELETE',
					headers: this._buildHeaders()
				});
			} catch (err) {
				console.warn('[TransportStreamableHTTP] Failed to close session:', err.message);
			}
		}

		this.sessionId = null;
	}

	_buildHeaders(options = {}) {
		const headers = {
			'Content-Type': 'application/json',
			Accept: options.accept || 'application/json, text/event-stream',
			'MCP-Protocol-Version': this.protocolVersion
		};

		if (this.sessionId && !options.ignoreSession) {
			headers['Mcp-Session-Id'] = this.sessionId;
		}

		if (this.config.headers && typeof this.config.headers === 'object') {
			for (const [key, value] of Object.entries(this.config.headers)) {
				if (value != null) {
					headers[key] = value;
				}
			}
		}

		return headers;
	}

	async _postMessage(message) {
		const hasId = Object.prototype.hasOwnProperty.call(message, 'id');

		try {
			const response = await this.fetchImpl(this.baseUrl, {
				method: 'POST',
				headers: this._buildHeaders(),
				body: JSON.stringify(message)
			});

			if (!this.sessionId) {
				const newSessionId = response.headers.get('mcp-session-id');
				if (newSessionId) {
					this.sessionId = newSessionId;
					this._ensureStream();
				}
			}

			const negotiatedVersion = response.headers.get('mcp-protocol-version');
			if (negotiatedVersion) {
				this.protocolVersion = negotiatedVersion;
			}

			if (response.status === 404) {
				this.sessionId = null;
				this._stopStream();
				if (hasId) {
					throw new Error('MCP session not found (HTTP 404)');
				}
				return;
			}

			if (!hasId) {
				return;
			}

			if (response.status === 202 || response.status === 204) {
				return;
			}

			const text = await response.text();
			if (!text) {
				throw new Error('Empty response from MCP server');
			}

			let parsed;
			try {
				parsed = JSON.parse(text);
			} catch (err) {
				throw new Error(`Invalid JSON response: ${err.message}`);
			}

			if (this.messageHandler) {
				this.messageHandler(parsed);
			}
		} catch (err) {
			console.error('[TransportStreamableHTTP] POST error:', err.message);
			if (hasId && this.messageHandler) {
				this.messageHandler({
					jsonrpc: '2.0',
					id: message.id,
					error: {
						code: -32000,
						message: err.message || 'Transport error'
					}
				});
			}
		}
	}

	_ensureStream() {
		if (!this.sessionId || this.streamAbortController || this.isStopping) {
			return;
		}

		this._startStream();
	}

	async _startStream() {
		this._stopStream();
		this.streamAbortController = new AbortController();
		let shouldReconnect = true;

		try {
			const response = await this.fetchImpl(this.baseUrl, {
				method: 'GET',
				headers: this._buildHeaders({ accept: 'text/event-stream' }),
				signal: this.streamAbortController.signal
			});

			if (!response.ok || !response.body) {
				throw new Error(`Stream connection failed (${response.status})`);
			}

			await this._consumeEventStream(response.body);
		} catch (err) {
			if (!this.isStopping && err?.name !== 'AbortError') {
				console.error('[TransportStreamableHTTP] Stream error:', err.message);
			}
		} finally {
			this._clearStream();

			if (this.isStopping) {
				shouldReconnect = false;
			}

			if (shouldReconnect && this.sessionId) {
				this._scheduleStreamReconnect();
			}
		}
	}

	async _consumeEventStream(body) {
		const reader = body.getReader();
		const decoder = new TextDecoder('utf-8');
		let buffer = '';

		while (!this.isStopping) {
			const { value, done } = await reader.read();
			if (done) {
				break;
			}

			buffer += decoder.decode(value, { stream: true }).replace(/\r/g, '\n');

			let separatorIndex;
			while ((separatorIndex = buffer.indexOf('\n\n')) >= 0) {
				const rawEvent = buffer.slice(0, separatorIndex);
				buffer = buffer.slice(separatorIndex + 2);
				this._handleSSEEvent(rawEvent);
			}
		}
	}

	_handleSSEEvent(rawEvent) {
		if (!rawEvent) {
			return;
		}

		const dataLines = rawEvent
			.split('\n')
			.map((line) => line.trim())
			.filter((line) => line.startsWith('data:'));

		if (dataLines.length === 0) {
			return;
		}

		const payload = dataLines.map((line) => line.slice(5).trim()).join('\n');
		if (!payload) {
			return;
		}

		try {
			const message = JSON.parse(payload);
			if (this.messageHandler) {
				this.messageHandler(message);
			}
		} catch (err) {
			console.error('[TransportStreamableHTTP] Invalid SSE payload:', err.message);
			console.error('[TransportStreamableHTTP] Raw payload:', payload);
		}
	}

	_scheduleStreamReconnect() {
		if (this.streamReconnectTimer || this.isStopping || !this.sessionId) {
			return;
		}

		this.streamReconnectTimer = setTimeout(() => {
			this.streamReconnectTimer = null;
			if (!this.isStopping && this.sessionId) {
				this._startStream();
			}
		}, 1000);
	}

	_stopStream() {
		if (this.streamReconnectTimer) {
			clearTimeout(this.streamReconnectTimer);
			this.streamReconnectTimer = null;
		}

		if (this.streamAbortController) {
			this.streamAbortController.abort();
		}

		this._clearStream();
	}

	_clearStream() {
		this.streamAbortController = null;
	}
}

module.exports = TransportStreamableHTTP;
