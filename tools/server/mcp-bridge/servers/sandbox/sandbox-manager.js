#!/usr/bin/env node
/**
 * Sandbox Manager - MCP Sandbox Lifecycle Orchestrator
 *
 * Manages the lifecycle of the MCP sandbox container:
 * - Builds clean image if not present
 * - Deploys pod from YAML
 * - Restarts container to revert to clean state
 * - Monitors container health
 *
 * Usage:
 *   node sandbox-manager.js build   # Build clean image
 *   node sandbox-manager.js deploy  # Deploy pod
 *   node sandbox-manager.js start   # Start stopped pod
 *   node sandbox-manager.js stop    # Stop pod
 *   node sandbox-manager.js restart # Restart container (clean state)
 *   node sandbox-manager.js status  # Show status
 *   node sandbox-manager.js logs    # Show container logs
 *   node sandbox-manager.js shell   # Open bash in container
 *   node sandbox-manager.js cleanup # Stop and remove pod
 *   node sandbox-manager.js setup   # Full setup (build + deploy)
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Load configuration
const config = require('./config.json');

const CONFIG = {
	hostUser: config.podman.hostUser,
	podName: 'sandbox',
	containerName: 'sandbox',
	imageName: 'localhost/sandbox-image:clean',
	containerfilePath: path.join(__dirname, 'Containerfile'),
	podYamlPath: path.join(__dirname, 'pod.yaml'),
	volumePaths: config.paths
};

class SandboxManager {
	constructor(cfg = CONFIG) {
		this.config = cfg;
	}

	/**
	 * Execute command as host user
	 */
	exec(cmd, options = {}) {
		const fullCmd = `su - ${this.config.hostUser} -c '${cmd.replace(/'/g, "'\"'\"'")}'`;

		if (!options.silent) {
			console.log(`[Manager] ${cmd}`);
		}

		try {
			return execSync(fullCmd, {
				encoding: 'utf8',
				stdio: options.silent ? 'pipe' : 'inherit',
				...options
			});
		} catch (error) {
			if (options.ignoreError) return null;
			throw error;
		}
	}

	/**
	 * Build clean image from Containerfile
	 */
	buildImage() {
		console.log('[Manager] Building clean sandbox image...');

		const buildCmd = `podman build -t ${this.config.imageName} -f ${this.config.containerfilePath} ${path.dirname(this.config.containerfilePath)}`;

		this.exec(buildCmd);

		console.log('[Manager] Image built successfully');
		console.log(`[Manager] Image: ${this.config.imageName}`);
	}

	/**
	 * Check if image exists
	 */
	imageExists() {
		const result = this.exec(`podman image exists ${this.config.imageName}`, {
			silent: true,
			ignoreError: true
		});
		return result !== null;
	}

	/**
	 * Ensure volumes exist
	 */
	ensureVolumes() {
		console.log('[Manager] Ensuring volume directories exist...');

		for (const [name, volPath] of Object.entries(this.config.volumePaths)) {
			if (!fs.existsSync(volPath)) {
				fs.mkdirSync(volPath, { recursive: true, mode: 0o755 });
				console.log(`[Manager]   Created: ${volPath}`);
			}
		}
	}

	/**
	 * Deploy pod from YAML
	 */
	deployPod() {
		console.log('[Manager] Deploying pod from YAML...');

		// Ensure image exists
		if (!this.imageExists()) {
			console.log('[Manager] Image not found, building...');
			this.buildImage();
		}

		// Ensure volumes exist
		this.ensureVolumes();

		// Deploy pod
		this.exec(`podman kube play --replace ${this.config.podYamlPath}`);

		console.log('[Manager] Pod deployed');

		// Wait for container to be ready
		console.log('[Manager] Waiting for container to be ready...');
		let attempts = 0;
		while (attempts < 30) {
			const status = this.getContainerStatus();
			if (status === 'running') {
				console.log('[Manager] Container is running');
				return;
			}
			attempts++;
			execSync('sleep 1');
		}

		console.warn('[Manager] Container may not be ready yet');
	}

	/**
	 * Start pod
	 */
	startPod() {
		console.log('[Manager] Starting pod...');
		this.exec(`podman pod start ${this.config.podName}`);
		console.log('[Manager] Pod started');
	}

	/**
	 * Stop pod
	 */
	stopPod() {
		console.log('[Manager] Stopping pod...');
		this.exec(`podman pod stop ${this.config.podName}`, { ignoreError: true });
		console.log('[Manager] Pod stopped');
	}

	/**
	 * Restart container to revert to clean state
	 */
	restartContainer() {
		console.log('[Manager] Restarting container (reverting to clean state)...');

		this.exec(`podman restart ${this.config.containerName}`);

		console.log('[Manager] Container restarted');
		console.log('[Manager] All volumes cleaned, back to pristine state');
	}

	/**
	 * Check if pod exists
	 */
	podExists() {
		const result = this.exec(`podman pod exists ${this.config.podName}`, {
			silent: true,
			ignoreError: true
		});
		return result !== null;
	}

	/**
	 * Get container status
	 */
	getContainerStatus() {
		try {
			const status = this.exec(
				`podman inspect ${this.config.containerName} --format '{{.State.Status}}'`,
				{ silent: true }
			);
			return status.trim();
		} catch {
			return 'not-found';
		}
	}

	/**
	 * Get pod status
	 */
	getPodStatus() {
		try {
			const status = this.exec(
				`podman pod inspect ${this.config.podName} --format '{{.State}}'`,
				{ silent: true }
			);
			return status.trim();
		} catch {
			return 'not-found';
		}
	}

	/**
	 * Display status
	 */
	showStatus() {
		console.log('[Manager] Sandbox Status');

		// Image
		const imageExists = this.imageExists();
		console.log(`Image:     ${imageExists ? 'OK' : 'NOT FOUND'} ${this.config.imageName}`);

		// Pod
		const podStatus = this.getPodStatus();
		console.log(`Pod:       ${podStatus}`);

		// Container
		const containerStatus = this.getContainerStatus();
		console.log(`Container: ${containerStatus}`);

		// Volumes
		console.log('\nVolumes:');
		for (const [name, volPath] of Object.entries(this.config.volumePaths)) {
			const exists = fs.existsSync(volPath);
			console.log(`  ${name.padEnd(12)} ${exists ? 'OK' : 'NOT FOUND'} ${volPath}`);
		}
	}

	/**
	 * Show container logs
	 */
	showLogs() {
		console.log('[Manager] Container logs:');
		this.exec(`podman logs ${this.config.containerName}`);
	}

	/**
	 * Open shell in container
	 */
	openShell() {
		console.log('[Manager] Opening bash in container...');
		console.log('[Manager] (Type "exit" to close)');

		const cmd = `su - ${this.config.hostUser} -c 'podman exec -it ${this.config.containerName} /bin/bash'`;

		try {
			execSync(cmd, { stdio: 'inherit' });
		} catch (error) {
			// User exited shell, normal behavior
		}
	}

	/**
	 * Full cleanup: stop and remove pod
	 */
	cleanup() {
		console.log('[Manager] Full cleanup...');

		this.stopPod();

		console.log('[Manager] Removing pod...');
		this.exec(`podman pod rm ${this.config.podName}`, { ignoreError: true });

		console.log('[Manager] Cleanup complete');
	}

	/**
	 * Full setup: build + deploy
	 */
	async setup() {
		console.log('[Manager] Full setup starting...');

		// Build image if needed
		if (!this.imageExists()) {
			this.buildImage();
		} else {
			console.log('[Manager] Image already exists');
		}

		// Deploy pod
		if (!this.podExists()) {
			this.deployPod();
		} else {
			const status = this.getPodStatus();
			if (status !== 'Running') {
				this.startPod();
			} else {
				console.log('[Manager] Pod already running');
			}
		}

		console.log('[Manager] Setup complete');
		this.showStatus();
	}
}

// CLI Interface
if (require.main === module) {
	const manager = new SandboxManager();
	const command = process.argv[2];

	const commands = {
		build: () => manager.buildImage(),
		deploy: () => manager.deployPod(),
		start: () => manager.startPod(),
		stop: () => manager.stopPod(),
		restart: () => manager.restartContainer(),
		status: () => manager.showStatus(),
		logs: () => manager.showLogs(),
		shell: () => manager.openShell(),
		cleanup: () => manager.cleanup(),
		setup: () => manager.setup()
	};

	if (!command || !commands[command]) {
		console.log('Usage: node sandbox-manager.js <command>\n');
		console.log('Commands:');
		console.log('build   - Build clean sandbox image');
		console.log('deploy  - Deploy pod from YAML');
		console.log('start   - Start stopped pod');
		console.log('stop    - Stop running pod');
		console.log('restart - Restart container (revert to clean state)');
		console.log('status  - Show current status');
		console.log('logs    - Show container logs');
		console.log('shell   - Open bash in container');
		console.log('cleanup - Stop and remove pod');
		console.log('setup   - Full setup (build + deploy)');
		process.exit(1);
	}

	(async () => {
		try {
			await commands[command]();
		} catch (err) {
			console.error('[Manager] Error:', err.message);
			process.exit(1);
		}
	})();
}

module.exports = SandboxManager;
