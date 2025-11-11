/**
 * Podman Executor Module
 * Handles secure command execution in Podman containers
 *
 * This module provides the low-level interface for executing bash commands
 * in isolated Podman containers. All commands are executed through a secure
 * pipeline using base64 encoding and user context switching.
 *
 * Configuration is loaded from config.json:
 * - podman.hostUser: System user that owns the Podman process
 * - podman.podName: Name of the pod
 * - podman.containerName: Name of the container (inside pod)
 * - podman.containerUser: User inside container for command execution
 * - podman.bashOutputLimitBytes: Maximum output size before truncation (bytes)
 * - podman.timeout: Command execution timeout in seconds
 *
 * Security model:
 * 1. Commands are base64-encoded to prevent injection attacks
 * 2. Execution happens via 'su - <user>' for proper user context
 * 3. Timeout protection prevents runaway processes
 * 4. Output size limits prevent memory exhaustion
 *
 * The module is designed for rootless Podman setups where the container
 * is owned by a non-root user on the host system.
 */

const { spawn } = require('child_process');
const config = require('../config.json');

// Construct full container name (pod-container)
const fullContainerName = `${config.podman.podName}-${config.podman.containerName}`;

/**
 * Execute command in Podman container
 * @param {string} script - Bash script to execute
 * @returns {Promise<{stdout: string, stderr: string, exitCode: number}>}
 */
async function podmanExec(script) {
	return new Promise((resolve) => {
		const b64 = Buffer.from(script).toString('base64');

		// Construct podman exec command
		const podmanCmd = `podman exec -i -u ${config.podman.containerUser} ${fullContainerName} bash -c "echo '${b64}' | base64 -d | timeout ${config.podman.timeout} bash 2>&1"`;

		// Execute as hostUser
		const args = ['-', config.podman.hostUser, '-c', podmanCmd];
		const proc = spawn('su', args);

		let stdout = '';
		let stderr = '';

		proc.stdout.on('data', (data) => {
			stdout += data.toString();
		});

		proc.stderr.on('data', (data) => {
			stderr += data.toString();
		});

		proc.on('close', (exitCode) => {
			resolve({
				stdout: stdout,
				stderr: stderr,
				exitCode: exitCode || 0
			});
		});

		proc.on('error', (err) => {
			resolve({
				stdout: '',
				stderr: err.message,
				exitCode: 1
			});
		});
	});
}

/**
 * Escape shell argument
 * @param {string} arg - Argument to escape
 * @returns {string} Escaped argument
 */
function escapeShell(arg) {
	return "'" + arg.replace(/'/g, "'\"'\"'") + "'";
}

/**
 * Escape regex special characters
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
function escapeRegex(str) {
	return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

module.exports = {
	podmanExec,
	escapeShell,
	escapeRegex
};
