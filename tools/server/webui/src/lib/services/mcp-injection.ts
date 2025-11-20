import { browser } from '$app/environment';
import { MCPClient } from '$lib/mcp';
import type { MCPClientConfig } from '$lib/mcp';
import { buildMcpClientConfig } from '$lib/config/mcp';

/**
 * Lightweight bootstrapper that instantiates a shared MCP client.
 *
 * This file intentionally contains all Svelte-specific glue (via
 * `$app/environment`). The MCP implementation itself lives in `$lib/mcp` and
 * can be reused without any dependency on the Svelte runtime.
 */

const globalState = globalThis as typeof globalThis & {
	__llamaMcpClient?: MCPClient;
	__llamaMcpInitPromise?: Promise<MCPClient | undefined>;
	__llamaMcpConfig?: MCPClientConfig;
	__LLAMA_MCP_CONFIG__?: MCPClientConfig;
};

function resolveConfig(): MCPClientConfig | null {
	const explicit = globalState.__llamaMcpConfig ?? globalState.__LLAMA_MCP_CONFIG__;
	if (explicit && typeof explicit === 'object' && 'servers' in explicit) {
		return explicit as MCPClientConfig;
	}

	return buildMcpClientConfig() ?? null;
}

async function bootstrapClient(): Promise<MCPClient | undefined> {
	if (!browser) {
		return undefined;
	}

	if (globalState.__llamaMcpClient) {
		return globalState.__llamaMcpClient;
	}

	if (!globalState.__llamaMcpInitPromise) {
		const config = resolveConfig();
		if (!config) {
			return undefined;
		}

		const client = new MCPClient(config);
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
	}

	return globalState.__llamaMcpInitPromise;
}

export function getMcpClient(): MCPClient | undefined {
	return globalState.__llamaMcpClient;
}

export async function ensureMcpClient(): Promise<MCPClient | undefined> {
	if (globalState.__llamaMcpClient) {
		return globalState.__llamaMcpClient;
	}
	return bootstrapClient();
}

if (browser) {
	void bootstrapClient();
}
