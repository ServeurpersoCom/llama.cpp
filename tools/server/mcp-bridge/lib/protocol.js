/**
 * JSON-RPC 2.0 protocol helpers
 */

class Protocol {
	/**
	 * Create a JSON-RPC 2.0 request
	 * @param {number} id - Request ID
	 * @param {string} method - Method name
	 * @param {object} params - Method parameters
	 * @returns {object} JSON-RPC request
	 */
	static createRequest(id, method, params) {
		return {
			jsonrpc: '2.0',
			id: id,
			method: method,
			params: params
		};
	}

	/**
	 * Create a JSON-RPC 2.0 notification
	 * @param {string} method - Method name
	 * @param {object} params - Notification parameters
	 * @returns {object} JSON-RPC notification
	 */
	static createNotification(method, params = {}) {
		return {
			jsonrpc: '2.0',
			method: method,
			params: params
		};
	}

	/**
	 * Parse and validate JSON-RPC 2.0 response
	 * @param {object} message - Raw message
	 * @returns {object|null} Parsed response or null if invalid
	 */
	static parseResponse(message) {
		if (!message || typeof message !== 'object') {
			return null;
		}
		if (message.jsonrpc !== '2.0') {
			return null;
		}
		return message;
	}

	/**
	 * Create a JSON-RPC 2.0 error response
	 * @param {number} id - Request ID
	 * @param {number} code - Error code
	 * @param {string} message - Error message
	 * @returns {object} JSON-RPC error response
	 */
	static createError(id, code, message) {
		return {
			jsonrpc: '2.0',
			id: id,
			error: {
				code: code,
				message: message
			}
		};
	}
}

module.exports = Protocol;
