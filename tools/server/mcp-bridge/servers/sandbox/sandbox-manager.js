#!/usr/bin/env node
/**
 * Sandbox Manager
 * Manages the MCP sandbox pod lifecycle via Podman Kube
 *
 * Lifecycle:
 *   build    Build image + deploy pod (idempotent from any state)
 *   destroy  Stop + remove pod + remove image (idempotent, inverse of build)
 *   start    Start pod
 *   stop     Stop pod
 *   reset    Recreate container from existing image (no rebuild)
 *
 * Utilities:
 *   status   Show image, pod, container and volumes state
 *   logs     Show container logs
 *   shell    Open bash in container
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const config = require('./config.json');

class SandboxManager {
	constructor(cfg) {
		this.podman = cfg.podman;
		this.paths = cfg.paths;
		this.rootDir = __dirname;

		// Host user from env or config
		const envUser = process.env.SANDBOX_HOST_USER;
		this.hostUser = envUser || this.podman.hostUser;
		if (!this.hostUser) {
			throw new Error('hostUser required (config or SANDBOX_HOST_USER env)');
		}

		// Resolve source files
		this.containerfileSrc = path.resolve(this.rootDir, this.podman.containerfile || 'Containerfile');
		this.podYamlSrc = path.resolve(this.rootDir, this.podman.podYaml || 'pod.yaml');

		if (!fs.existsSync(this.containerfileSrc)) {
			throw new Error('Containerfile not found: ' + this.containerfileSrc);
		}
		if (!fs.existsSync(this.podYamlSrc)) {
			throw new Error('pod.yaml not found: ' + this.podYamlSrc);
		}
		if (!this.podman.podName || !this.podman.containerName) {
			throw new Error('podName and containerName required in config');
		}

		// Podman Kube naming: <pod> + <container>
		this.fullContainerName = this.podman.podName + '-' + this.podman.containerName;

		// Temp copies (host user cannot read root owned source files)
		this.tmpContainerfile = '/tmp/' + (this.podman.containerfile || 'Containerfile');
		this.tmpPodYaml = '/tmp/' + (this.podman.podYaml || 'pod.yaml');
	}

	// Execute command as hostUser via su
	exec(cmd, opts = {}) {
		const escaped = cmd.replace(/'/g, "'\"'\"'");
		const full = "su - " + this.hostUser + " -c '" + escaped + "'";
		console.log('[Manager] ' + full);

		try {
			return execSync(full, {
				encoding: 'utf8',
				stdio: opts.stdio || 'inherit'
			});
		} catch (e) {
			if (opts.ignoreError) return null;
			throw e;
		}
	}

	// Copy source file to /tmp so hostUser can read it
	stageTo(src, dst) {
		fs.copyFileSync(src, dst);
	}

	// Remove staged temp file
	unstage(tmp) {
		if (fs.existsSync(tmp)) fs.unlinkSync(tmp);
	}

	// Check if image exists in local store
	imageExists() {
		return this.exec('podman image exists ' + this.podman.imageName, {
			stdio: 'pipe',
			ignoreError: true
		}) !== null;
	}

	// Check if pod exists
	podExists() {
		return this.exec('podman pod exists ' + this.podman.podName, {
			stdio: 'pipe',
			ignoreError: true
		}) !== null;
	}

	// Query pod state (Running, Stopped, etc)
	podState() {
		try {
			const out = this.exec(
				'podman pod inspect ' + this.podman.podName + ' --format "{{.State}}"',
				{ stdio: 'pipe' }
			);
			return out.trim();
		} catch (e) {
			return 'not found';
		}
	}

	// Query container state
	containerState() {
		try {
			const out = this.exec(
				'podman inspect ' + this.fullContainerName + ' --format "{{.State.Status}}"',
				{ stdio: 'pipe' }
			);
			return out.trim();
		} catch (e) {
			return 'not found';
		}
	}

	// Ensure host volume directories exist
	ensureVolumes() {
		for (const [name, volPath] of Object.entries(this.paths)) {
			if (!fs.existsSync(volPath)) {
				fs.mkdirSync(volPath, { recursive: true, mode: 0o755 });
				console.log('[Manager] created volume: ' + volPath);
			}
		}
	}

	// Wait for pod to reach Running state
	waitForPod(maxSeconds) {
		console.log('[Manager] waiting for pod...');
		for (let i = 0; i < maxSeconds; i++) {
			if (this.podState() === 'Running') {
				console.log('[Manager] pod is running');
				return;
			}
			execSync('sleep 1');
		}
		console.warn('[Manager] pod not ready after ' + maxSeconds + 's');
	}

	// build: full idempotent (re)build from any state
	// Removes existing pod and image, rebuilds everything
	build() {
		console.log('[Manager] build: full rebuild from any state');

		// Tear down pod if present (ignore errors)
		this.exec('podman pod stop ' + this.podman.podName, { ignoreError: true, stdio: 'pipe' });
		this.exec('podman pod rm ' + this.podman.podName, { ignoreError: true, stdio: 'pipe' });

		// Remove old image if present
		this.exec('podman rmi ' + this.podman.imageName, { ignoreError: true, stdio: 'pipe' });

		// Build fresh image
		this.stageTo(this.containerfileSrc, this.tmpContainerfile);
		this.exec('podman build -t ' + this.podman.imageName + ' -f ' + this.tmpContainerfile + ' /tmp');
		this.unstage(this.tmpContainerfile);

		// Deploy pod
		this.ensureVolumes();
		this.stageTo(this.podYamlSrc, this.tmpPodYaml);
		this.exec('podman kube play ' + this.tmpPodYaml);
		this.unstage(this.tmpPodYaml);

		this.waitForPod(30);
		console.log('[Manager] build complete');
	}

	// destroy: inverse of build, idempotent
	// Removes pod and image, always reaches empty state
	destroy() {
		console.log('[Manager] destroy: removing everything');

		this.exec('podman pod stop ' + this.podman.podName, { ignoreError: true, stdio: 'pipe' });
		this.exec('podman pod rm ' + this.podman.podName, { ignoreError: true, stdio: 'pipe' });
		this.exec('podman rmi ' + this.podman.imageName, { ignoreError: true, stdio: 'pipe' });

		console.log('[Manager] destroy complete');
	}

	// start: resume a stopped pod
	start() {
		if (!this.podExists()) {
			console.error('[Manager] no pod to start, run build first');
			process.exit(1);
		}

		this.exec('podman pod start ' + this.podman.podName);
		console.log('[Manager] started');
	}

	// stop: pause pod without destroying anything
	stop() {
		this.exec('podman pod stop ' + this.podman.podName, { ignoreError: true });
		console.log('[Manager] stopped');
	}

	// reset: recreate container from existing image (no rebuild)
	// Fast way to get a clean container without waiting for image build
	reset() {
		if (!this.imageExists()) {
			console.error('[Manager] no image found, run build first');
			process.exit(1);
		}

		console.log('[Manager] reset: recreating container from existing image');

		this.stageTo(this.podYamlSrc, this.tmpPodYaml);
		this.exec('podman kube play --replace ' + this.tmpPodYaml);
		this.unstage(this.tmpPodYaml);

		this.waitForPod(30);
		console.log('[Manager] reset complete');
	}

	status() {
		const img = this.imageExists() ? 'OK' : 'NOT FOUND';
		const pod = this.podState();
		const ctr = this.containerState();

		console.log('[Manager] image: ' + img + ' (' + this.podman.imageName + ')');
		console.log('[Manager] pod: ' + pod);
		console.log('[Manager] container: ' + ctr);

		for (const [name, volPath] of Object.entries(this.paths)) {
			const exists = fs.existsSync(volPath) ? 'OK' : 'NOT FOUND';
			console.log('[Manager] ' + name + ': ' + exists + ' (' + volPath + ')');
		}
	}

	logs() {
		this.exec('podman logs ' + this.fullContainerName);
	}

	shell() {
		console.log('[Manager] opening shell (type exit to close)');
		const cmd = "su - " + this.hostUser + " -c 'podman exec -it " + this.fullContainerName + " /bin/bash'";
		try {
			execSync(cmd, { stdio: 'inherit' });
		} catch (e) {
			// Normal exit
		}
	}
}

// CLI
const commands = {
	build:   'Build image + deploy pod (idempotent from any state)',
	destroy: 'Stop + remove pod + remove image',
	start:   'Start pod',
	stop:    'Stop pod',
	reset:   'Recreate container from existing image (no rebuild)',
	status:  'Show current state',
	logs:    'Show container logs',
	shell:   'Open bash in container'
};

const command = process.argv[2];

if (!command || !commands[command]) {
	console.log('Usage: node sandbox-manager.js <command>\n');
	for (const [cmd, desc] of Object.entries(commands)) {
		console.log('  ' + cmd.padEnd(10) + desc);
	}
	process.exit(1);
}

const manager = new SandboxManager(config);

try {
	manager[command]();
} catch (err) {
	console.error('[Manager] error: ' + err.message);
	process.exit(1);
}
