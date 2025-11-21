import { browser } from '$app/environment';
import { MCPClient } from '$lib/mcp';
import { buildMcpClientConfig } from '$lib/config/mcp';
import { config } from '$lib/stores/settings.svelte';

const globalState = globalThis as typeof globalThis & {
	__llamaMcpClient?: MCPClient;
	__llamaMcpInitPromise?: Promise<MCPClient | undefined>;
};

async function bootstrapClient(): Promise<MCPClient | undefined> {
	if (!browser) {
		return undefined;
	}

	if (globalState.__llamaMcpClient) {
		return globalState.__llamaMcpClient;
	}

	if (globalState.__llamaMcpInitPromise) {
		return globalState.__llamaMcpInitPromise;
	}

	const mcpConfig = buildMcpClientConfig(config());
	if (!mcpConfig) {
		return undefined;
	}

	const client = new MCPClient(mcpConfig);
	globalState.__llamaMcpInitPromise = client
		.initialize()
		.then(() => {
			globalState.__llamaMcpClient = client;
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
			globalState.__llamaMcpInitPromise = undefined;
		});

	return globalState.__llamaMcpInitPromise;
}

export function getMcpClient(): MCPClient | undefined {
	return globalState.__llamaMcpClient;
}

export async function ensureMcpClient(): Promise<MCPClient | undefined> {
	return globalState.__llamaMcpClient ?? bootstrapClient();
}
