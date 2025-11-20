import { mcpConfig } from './mcp';

/**
 * Agentic orchestration configuration.
 */
export interface AgenticConfig {
	enabled: boolean;
	maxTurns: number;
	maxToolPreviewLines: number;
	filterReasoningAfterFirstTurn: boolean;
	loggingConsole: boolean;
}

/**
 * Default agentic orchestration settings.
 * Modify these values to change agentic behavior.
 */
export const agenticConfig: AgenticConfig = {
	enabled: true,
	maxTurns: 100,
	maxToolPreviewLines: 25,
	filterReasoningAfterFirstTurn: true,
	loggingConsole: true
};

/**
 * Gets the current agentic configuration.
 * Automatically disables agentic mode if no MCP servers are configured.
 */
export function getAgenticConfig(): AgenticConfig {
	const servers = mcpConfig.servers ?? {};
	const hasServers = Object.keys(servers).length > 0;

	return {
		...agenticConfig,
		enabled: agenticConfig.enabled && hasServers
	};
}
