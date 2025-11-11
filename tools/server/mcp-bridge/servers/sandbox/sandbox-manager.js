#!/usr/bin/env node
/**
 * Sandbox Manager - MCP Sandbox Lifecycle Orchestrator
 *
 * Manages the lifecycle of the MCP sandbox pod and containers:
 * - Builds clean image if not present
 * - Deploys pod from YAML (Kubernetes-style manifest)
 * - Manages pod and container lifecycle (start/stop/restart)
 * - Monitors pod and container health
 *
 * Architecture:
 * - 1 Pod = 1 User isolation (network, resources)
 * - N Containers per pod = N Conversations/sessions
 * - Volumes: inputs (RO), workspace (RW), outputs (RW)
 *
 * Usage:
 *   node sandbox-manager.js build      # Build clean image
 *   node sandbox-manager.js deploy     # Deploy pod from YAML
 *   node sandbox-manager.js start      # Start pod (all containers)
 *   node sandbox-manager.js stop       # Stop pod (all containers)
 *   node sandbox-manager.js restart    # Restart pod (clean state)
 *   node sandbox-manager.js status     # Show pod and containers status
 *   node sandbox-manager.js logs       # Show container logs
 *   node sandbox-manager.js shell      # Open bash in container
 *   node sandbox-manager.js cleanup    # Stop and remove pod
 *   node sandbox-manager.js setup      # Full setup (build + deploy)
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Load configuration
const config = require('./config.json');

const resolvePath = (baseDir, fileName, fallback) => {
	const target = fileName || fallback;
	return path.resolve(baseDir, target);
};

class SandboxManager {
	constructor(configFile) {
		this.podman = configFile.podman;
		this.paths = configFile.paths;

		this.rootDir = __dirname;
		this.containerfilePath = resolvePath(
			this.rootDir,
			this.podman.containerfile,
			'Containerfile'
		);
		this.podYamlPath = resolvePath(this.rootDir, this.podman.podYaml, 'pod.yaml');

		const envHostUser = process.env.SANDBOX_HOST_USER;
		this.hostUser = envHostUser || this.podman.hostUser;

		if (!this.hostUser) {
			throw new Error(
				'Invalid podman configuration: hostUser required (or SANDBOX_HOST_USER env var)'
			);
		}
		this.podman.hostUser = this.hostUser;

		if (!fs.existsSync(this.containerfilePath)) {
			throw new Error(`Containerfile not found: ${this.containerfilePath}`);
		}

		if (!fs.existsSync(this.podYamlPath)) {
			throw new Error(`pod.yaml not found: ${this.podYamlPath}`);
		}

		if (!this.podman.podName || !this.podman.containerName) {
			throw new Error('Invalid podman configuration: podName and containerName are required');
		}

		// Full container name: <pod>-<container> (Podman Kube naming convention)
		this.fullContainerName = `${this.podman.podName}-${this.podman.containerName}`;

		// Temp files in /tmp
		this.tmpContainerfile = `/tmp/${this.podman.containerfile}`;
		this.tmpPodYaml = `/tmp/${this.podman.podYaml}`;
	}

	/**
	 * Execute command as hostUser
	 */
	exec(cmd, options = {}) {
		const { ignoreError, stdio, ...rest } = options;
		const fullCmd = `su - ${this.hostUser} -c '${cmd.replace(/'/g, "'\"'\"'")}'`;

		console.log(`[Manager] ${fullCmd}`);

		try {
			return execSync(fullCmd, {
				encoding: 'utf8',
				stdio: stdio ?? 'inherit',
				...rest
			});
		} catch (error) {
			if (ignoreError) return null;
			throw error;
		}
	}

	/**
	 * Build clean image from Containerfile
	 */
	buildImage() {
		console.log('[Manager] Building clean sandbox image...');
		console.log(`[Manager] Containerfile source: ${this.containerfilePath}`);

		// Copy Containerfile to /tmp (accessible by hostUser)
		fs.copyFileSync(this.containerfilePath, this.tmpContainerfile);

		const image = this.podman.imageName;
		const buildCmd = `podman build -t ${image} -f ${this.tmpContainerfile} /tmp`;

		this.exec(buildCmd);

		// Cleanup
		fs.unlinkSync(this.tmpContainerfile);

		console.log('[Manager] Image built successfully');
		console.log(`[Manager] Image: ${image}`);
	}

	/**
	 * Check if image exists
	 */
	imageExists() {
		const image = this.podman.imageName;
		const result = this.exec(`podman image exists ${image}`, {
			stdio: 'pipe',
			ignoreError: true
		});
		return result !== null;
	}

	/**
	 * Ensure volume directories exist
	 */
	ensureVolumes() {
		console.log('[Manager] Ensuring volume directories exist...');

		for (const [name, volPath] of Object.entries(this.paths)) {
			if (!fs.existsSync(volPath)) {
				fs.mkdirSync(volPath, { recursive: true, mode: 0o755 });
				console.log(`[Manager] Created: ${volPath}`);
			}
		}
	}

	/**
	 * Deploy pod from YAML (Kubernetes-style manifest)
	 */
	deployPod() {
		console.log('[Manager] Deploying pod from YAML...');
		console.log(`[Manager] Pod manifest source: ${this.podYamlPath}`);

		// Ensure image exists
		if (!this.imageExists()) {
			console.log('[Manager] Image not found, building...');
			this.buildImage();
		}

		// Ensure volumes exist
		this.ensureVolumes();

		// Copy pod.yaml to /tmp (accessible by hostUser)
		fs.copyFileSync(this.podYamlPath, this.tmpPodYaml);

		// Deploy pod with Podman Kube
		this.exec(`podman kube play --replace ${this.tmpPodYaml}`);

		// Cleanup
		fs.unlinkSync(this.tmpPodYaml);

		console.log('[Manager] Pod deployed');

		// Wait for pod to be ready
		console.log('[Manager] Waiting for pod to be ready...');
		let attempts = 0;
		while (attempts < 30) {
			const status = this.getPodStatus();
			if (status === 'Running') {
				console.log('[Manager] Pod is running');
				return;
			}
			attempts++;
			execSync('sleep 1');
		}

		console.warn('[Manager] Pod may not be ready yet');
	}

	/**
	 * Start pod (starts all containers in the pod)
	 */
	startPod() {
		console.log('[Manager] Starting pod...');
		const pod = this.podman.podName;
		this.exec(`podman pod start ${pod}`);
		console.log('[Manager] Pod started');
	}

	/**
	 * Stop pod (stops all containers in the pod)
	 */
	stopPod() {
		console.log('[Manager] Stopping pod...');
		const pod = this.podman.podName;
		this.exec(`podman pod stop ${pod}`, { ignoreError: true });
		console.log('[Manager] Pod stopped');
	}

	/**
	 * Restart pod (revert to clean state)
	 *
	 * Podman Kube: restart does NOT recreate containers from image
	 * Use kube play --replace to destroy and recreate from pristine image
	 */
	restartPod() {
		console.log('[Manager] Restarting pod (reverting to clean state)...');

		// Copy pod.yaml to /tmp (accessible by hostUser)
		fs.copyFileSync(this.podYamlPath, this.tmpPodYaml);

		// --replace stops, removes, and recreates the pod from image
		this.exec(`podman kube play --replace ${this.tmpPodYaml}`);

		// Cleanup
		fs.unlinkSync(this.tmpPodYaml);

		console.log('[Manager] Pod restarted');
		console.log('[Manager] All volumes and containers cleaned, back to pristine state');
	}

	/**
	 * Check if pod exists
	 */
	podExists() {
		const pod = this.podman.podName;
		const result = this.exec(`podman pod exists ${pod}`, {
			stdio: 'pipe',
			ignoreError: true
		});
		return result !== null;
	}

	/**
	 * Get container status
	 */
	getContainerStatus() {
		try {
			const container = this.fullContainerName;
			const status = this.exec(`podman inspect ${container} --format "{{.State.Status}}"`, {
				stdio: 'pipe'
			});
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
			const pod = this.podman.podName;
			const status = this.exec(`podman pod inspect ${pod} --format "{{.State}}"`, {
				stdio: 'pipe'
			});
			return status.trim();
		} catch {
			return 'not-found';
		}
	}

	/**
	 * Display pod and containers status
	 */
	showStatus() {
		console.log('[Manager] Sandbox Status');

		// Image
		const image = this.podman.imageName;
		const imageExists = this.imageExists();
		console.log(`[Manager] Image: ${imageExists ? 'OK' : 'NOT FOUND'} ${image}`);

		// Pod
		const podStatus = this.getPodStatus();
		console.log(`[Manager] Pod: ${podStatus}`);

		// Container
		const containerStatus = this.getContainerStatus();
		console.log(`[Manager] Container: ${containerStatus}`);

		// Volumes
		console.log('[Manager] Volumes:');
		for (const [name, volPath] of Object.entries(this.paths)) {
			const exists = fs.existsSync(volPath);
			console.log(`[Manager] ${name}: ${exists ? 'OK' : 'NOT FOUND'} ${volPath}`);
		}
	}

	/**
	 * Show container logs
	 */
	showLogs() {
		console.log('[Manager] Container logs:');
		const container = this.fullContainerName;
		this.exec(`podman logs ${container}`);
	}

	/**
	 * Open shell in container
	 */
	openShell() {
		console.log('[Manager] Opening bash in container...');
		console.log('[Manager] (Type "exit" to close)');

		const container = this.fullContainerName;
		const cmd = `su - ${this.podman.hostUser} -c 'podman exec -it ${container} /bin/bash'`;

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
		const pod = this.podman.podName;
		this.exec(`podman pod rm ${pod}`, { ignoreError: true });

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
	const manager = new SandboxManager(config);
	const command = process.argv[2];

	const commands = {
		build: () => manager.buildImage(),
		deploy: () => manager.deployPod(),
		start: () => manager.startPod(),
		stop: () => manager.stopPod(),
		restart: () => manager.restartPod(),
		status: () => manager.showStatus(),
		logs: () => manager.showLogs(),
		shell: () => manager.openShell(),
		cleanup: () => manager.cleanup(),
		setup: () => manager.setup()
	};

	if (!command || !commands[command]) {
		console.log('Usage: node sandbox-manager.js <command>');
		console.log('');
		console.log('Commands:');
		console.log('build   - Build clean sandbox image');
		console.log('deploy  - Deploy pod from YAML');
		console.log('start   - Start pod (all containers)');
		console.log('stop    - Stop pod (all containers)');
		console.log('restart - Restart pod (revert to clean state)');
		console.log('status  - Show pod and containers status');
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
