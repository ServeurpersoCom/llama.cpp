/**
 * TransportStdio: Spawns MCP server process and handles stdio communication
 */

const { spawn } = require('child_process');
const readline = require('readline');

class TransportStdio {
	constructor(config) {
		if (!config.command) {
			throw new Error('TransportStdio requires config.command');
		}
		if (!config.args) {
			throw new Error('TransportStdio requires config.args');
		}

		this.config = {
			...config,
			args: Array.isArray(config.args) ? config.args : []
		};
		this.process = null;
		this.rl = null;
		this.messageHandler = null;
	}

	/**
	 * Start the MCP server process
	 */
	async start() {
		return new Promise((resolve, reject) => {
			try {
				this.process = spawn(this.config.command, this.config.args, {
					env: { ...process.env, ...(this.config.env || {}) },
					stdio: ['pipe', 'pipe', 'pipe']
				});

				this.rl = readline.createInterface({
					input: this.process.stdout,
					crlfDelay: Infinity
				});

				this.rl.on('line', (line) => {
					if (this.messageHandler) {
						try {
							const message = JSON.parse(line);
							this.messageHandler(message);
						} catch (e) {
							console.error('[Transport] JSON parse error:', e.message);
							console.error('[Transport] Raw line:', line);
						}
					}
				});

				this.process.stderr.on('data', (data) => {
					process.stderr.write(data);
				});

				this.process.on('error', (err) => {
					console.error('[Transport] Process error:', err);
					reject(err);
				});

				this.process.on('exit', (code, signal) => {
					console.log(`[Transport] Process exited with code ${code}, signal ${signal}`);
				});

				// Give process a moment to start
				setTimeout(() => resolve(), 100);
			} catch (err) {
				reject(err);
			}
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
		if (!this.process || !this.process.stdin) {
			throw new Error('Transport not started or stdin unavailable');
		}
		this.process.stdin.write(JSON.stringify(message) + '\n');
	}

	/**
	 * Stop the MCP server process
	 */
	async stop() {
		if (this.process) {
			this.process.kill();
			this.process = null;
		}
		if (this.rl) {
			this.rl.close();
			this.rl = null;
		}
	}
}

module.exports = TransportStdio;
