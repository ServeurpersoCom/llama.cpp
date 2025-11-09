/**
 * TransportWebSocket: Connects to remote MCP server via WebSocket
 */

const hasBuffer = typeof Buffer !== 'undefined';
const hasArrayBuffer = typeof ArrayBuffer === 'function';
const isBuffer = (value) => hasBuffer && Buffer.isBuffer(value);
const toUtf8String = (value) => {
	if (isBuffer(value)) {
		return value.toString('utf8');
	}
	if (hasArrayBuffer && value instanceof ArrayBuffer) {
		if (hasBuffer) {
			return Buffer.from(value).toString('utf8');
		}
		if (typeof TextDecoder === 'function') {
			return new TextDecoder('utf-8').decode(value);
		}
	}
	return value;
};

function resolveWebSocketImpl() {
	if (typeof globalThis !== 'undefined' && typeof globalThis.WebSocket === 'function') {
		return globalThis.WebSocket;
	}

	try {
		// Lazy require to avoid dependency when native WebSocket is available
		return require('ws');
	} catch (err) {
		return null;
	}
}

function normalizeProtocols(protocols) {
	if (protocols == null) {
		return undefined;
	}

	if (Array.isArray(protocols)) {
		return protocols.filter((value) => typeof value === 'string' && value.trim().length > 0);
	}

	if (typeof protocols === 'string' && protocols.trim().length > 0) {
		return protocols;
	}

	return undefined;
}

class TransportWebSocket {
	constructor(config) {
		if (!config || !config.url) {
			throw new Error('TransportWebSocket requires config.url');
		}

		this.config = config;
		this.ws = null;
		this.messageHandler = null;
	}

	/**
	 * Start the WebSocket connection
	 */
	async start() {
		if (this.ws) {
			return;
		}

		return new Promise((resolve, reject) => {
			const WebSocketImpl = resolveWebSocketImpl();
			if (!WebSocketImpl) {
				reject(
					new Error(
						'No WebSocket implementation available. Install the "ws" package or use Node.js 18+ with built-in WebSocket support.'
					)
				);
				return;
			}

			const protocols = normalizeProtocols(this.config.protocols);
			const options = {};
			if (this.config.headers && typeof this.config.headers === 'object') {
				options.headers = this.config.headers;
			}
			if (typeof this.config.handshakeTimeout === 'number') {
				options.handshakeTimeout = this.config.handshakeTimeout;
			}
			if (typeof this.config.rejectUnauthorized === 'boolean') {
				options.rejectUnauthorized = this.config.rejectUnauthorized;
			}

			const useGlobal =
				typeof globalThis !== 'undefined' && WebSocketImpl === globalThis.WebSocket;
			const hasOptions = Object.keys(options).length > 0;

			try {
				if (useGlobal) {
					if (hasOptions) {
						console.warn(
							'[TransportWebSocket] Custom headers/options are not supported by the global WebSocket implementation.'
						);
					}
					this.ws =
						protocols !== undefined
							? new WebSocketImpl(this.config.url, protocols)
							: new WebSocketImpl(this.config.url);
				} else {
					if (protocols !== undefined) {
						this.ws = new WebSocketImpl(this.config.url, protocols, options);
					} else if (hasOptions) {
						this.ws = new WebSocketImpl(this.config.url, options);
					} else {
						this.ws = new WebSocketImpl(this.config.url);
					}
				}
			} catch (err) {
				reject(err);
				return;
			}

			const attach = (event, handler) => {
				if (!this.ws) return;
				if (typeof this.ws.on === 'function') {
					this.ws.on(event, handler);
				} else if (typeof this.ws.addEventListener === 'function') {
					this.ws.addEventListener(event, handler);
				} else {
					const prop = `on${event}`;
					this.ws[prop] = handler;
				}
			};

			let settled = false;

			attach('open', () => {
				if (settled) return;
				settled = true;
				resolve();
			});

			attach('error', (eventOrError) => {
				const error =
					eventOrError instanceof Error
						? eventOrError
						: eventOrError?.error || eventOrError;
				console.error('[TransportWebSocket] Connection error:', error?.message || error);
				if (!settled) {
					settled = true;
					reject(
						error instanceof Error ? error : new Error('WebSocket connection error')
					);
				}
			});

			attach('close', (code, reason) => {
				const readableReason =
					typeof reason === 'string' && reason.length > 0
						? reason
						: isBuffer(reason)
							? reason.toString('utf8')
							: undefined;
				console.log(
					`[TransportWebSocket] Connection closed (code=${code}${readableReason ? `, reason=${readableReason}` : ''})`
				);
			});

			attach('message', (event, isBinary) => {
				if (!this.messageHandler) {
					return;
				}

				let data;
				if (event && typeof event === 'object' && 'data' in event) {
					data = event.data;
				} else if (isBuffer(event) || (hasArrayBuffer && event instanceof ArrayBuffer)) {
					data = event;
				} else if (typeof event === 'string') {
					data = event;
				} else if (event && typeof event === 'object' && 'toString' in event) {
					data = event.toString();
				}

				if (isBinary === true && isBuffer(event)) {
					data = event.toString('utf8');
				}

				data = toUtf8String(data);

				if (typeof data !== 'string') {
					console.warn('[TransportWebSocket] Ignored non-text message');
					return;
				}

				try {
					const message = JSON.parse(data);
					this.messageHandler(message);
				} catch (err) {
					console.error('[TransportWebSocket] JSON parse error:', err.message);
					console.error('[TransportWebSocket] Raw message:', data);
				}
			});
		});
	}

	/**
	 * Set message handler for incoming messages
	 * @param {function} handler - Handler function(message)
	 */
	onMessage(handler) {
		this.messageHandler = handler;
	}

	/**
	 * Send message to MCP server
	 * @param {object} message - JSON-RPC message
	 */
	send(message) {
		if (!this.ws) {
			throw new Error('Transport not started');
		}

		const readyState = this.ws.readyState;
		const OPEN = this.ws.OPEN ?? (this.ws.constructor && this.ws.constructor.OPEN) ?? 1;
		if (readyState !== OPEN) {
			throw new Error('WebSocket connection is not open');
		}

		const payload = JSON.stringify(message);
		this.ws.send(payload);
	}

	/**
	 * Stop the WebSocket connection
	 */
	async stop() {
		if (this.ws) {
			try {
				this.ws.close();
			} catch (err) {
				console.error('[TransportWebSocket] Error closing WebSocket:', err.message);
			}
			this.ws = null;
		}
	}
}

module.exports = TransportWebSocket;
