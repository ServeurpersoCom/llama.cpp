import { browser } from '$app/environment';
import { MCPClient } from '$lib/mcp';
import { buildMcpClientConfig } from '$lib/config/mcp';
import { config } from '$lib/stores/settings.svelte';

const globalState = globalThis as typeof globalThis & {
	__llamaMcpClient?: MCPClient;
	__llamaMcpInitPromise?: Promise<MCPClient | undefined>;
	__llamaMcpConfigSignature?: string;
	__llamaMcpInitConfigSignature?: string;
};

function serializeConfigSignature(): string | undefined {
	const mcpConfig = buildMcpClientConfig(config());
	return mcpConfig ? JSON.stringify(mcpConfig) : undefined;
}

async function shutdownClient(): Promise<void> {
	if (!globalState.__llamaMcpClient) return;

	try {
		await globalState.__llamaMcpClient.shutdown();
	} catch (error) {
		console.error('[MCP] Failed to shutdown client:', error);
	} finally {
		globalState.__llamaMcpClient = undefined;
		globalState.__llamaMcpConfigSignature = undefined;
	}
}

async function bootstrapClient(): Promise<MCPClient | undefined> {
	if (!browser) {
		return undefined;
	}

	const mcpConfig = buildMcpClientConfig(config());
	const signature = mcpConfig ? JSON.stringify(mcpConfig) : undefined;
	if (!mcpConfig || !signature) {
		return undefined;
	}

	if (globalState.__llamaMcpClient && globalState.__llamaMcpConfigSignature === signature) {
		return globalState.__llamaMcpClient;
	}

	if (
		globalState.__llamaMcpInitPromise &&
		globalState.__llamaMcpInitConfigSignature === signature
	) {
		return globalState.__llamaMcpInitPromise;
	}

	const client = new MCPClient(mcpConfig);
	globalState.__llamaMcpInitConfigSignature = signature;
	const initPromise = client
		.initialize()
		.then(() => {
			// Ignore initialization if config changed during bootstrap
			if (globalState.__llamaMcpInitConfigSignature !== signature) {
				void client.shutdown().catch((shutdownError) => {
					console.error(
						'[MCP] Failed to shutdown stale client after config change:',
						shutdownError
					);
				});
				return undefined;
			}

			globalState.__llamaMcpClient = client;
			globalState.__llamaMcpConfigSignature = signature;
			return client;
		})
		.catch((error) => {
			console.error('[MCP] Failed to initialize client:', error);
			void client.shutdown().catch((shutdownError) => {
				console.error('[MCP] Failed to shutdown client after init error:', shutdownError);
			});
			return undefined;
		})
		.finally(() => {
			if (globalState.__llamaMcpInitPromise === initPromise) {
				globalState.__llamaMcpInitPromise = undefined;
			}

			if (globalState.__llamaMcpInitConfigSignature === signature) {
				globalState.__llamaMcpInitConfigSignature = undefined;
			}
		});

	globalState.__llamaMcpInitPromise = initPromise;

	return initPromise;
}

export function getMcpClient(): MCPClient | undefined {
	return globalState.__llamaMcpClient;
}

export async function ensureMcpClient(): Promise<MCPClient | undefined> {
	const signature = serializeConfigSignature();

	// Configuration removed: shut down active client if present
	if (!signature) {
		await shutdownClient();
		globalState.__llamaMcpInitPromise = undefined;
		globalState.__llamaMcpInitConfigSignature = undefined;
		return undefined;
	}

	if (globalState.__llamaMcpConfigSignature !== signature) {
		await shutdownClient();

		if (globalState.__llamaMcpInitConfigSignature !== signature) {
			globalState.__llamaMcpInitPromise = undefined;
			globalState.__llamaMcpInitConfigSignature = undefined;
		}
	}

	return globalState.__llamaMcpClient ?? bootstrapClient();
}
