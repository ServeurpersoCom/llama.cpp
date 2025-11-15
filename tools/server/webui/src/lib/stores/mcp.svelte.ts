import { browser } from '$app/environment';
import { toast } from 'svelte-sonner';
import { SvelteMap } from 'svelte/reactivity';
import { config } from '$lib/stores/settings.svelte';
import type { SettingsConfigType } from '$lib/types/settings';
import type { DatabaseMessage } from '$lib/types/database';
import type {
	ApiChatCompletionToolCall,
	ApiChatCompletionToolDefinition,
	ApiChatMessageData
} from '$lib/types/api';
import type {
	MCPContextFragment,
	MCPToolDescription,
	MCPToolExecutionResult
} from '$lib/types/mcp';
import { MCPClient } from '$lib/services/mcp-client';
import { MCP_REQUEST_TIMEOUT_MS } from '$lib/constants/mcp';

class MCPStore {
	status = $state<'idle' | 'connecting' | 'ready' | 'error'>('idle');
	lastError = $state<string | null>(null);
	private client: MCPClient | null = null;
	private toolExecutions = new SvelteMap<string, MCPToolExecutionResult[]>();
	private pendingContext = new SvelteMap<string, MCPContextFragment[]>();
	private pendingMessages = new SvelteMap<string, boolean>();
	private activeConnectionKey: string | null = null;
	private toolDefinitions: ApiChatCompletionToolDefinition[] = [];

	getExecutionsForMessage(messageId: string): MCPToolExecutionResult[] {
		return this.toolExecutions.get(messageId) ?? [];
	}

	isMessageRunning(messageId: string): boolean {
		return Boolean(this.pendingMessages.get(messageId));
	}

	async getToolDefinitions(
		settings?: SettingsConfigType
	): Promise<ApiChatCompletionToolDefinition[]> {
		const targetSettings = settings ?? config();
		const client = await this.ensureClient(targetSettings);

		if (!client) {
			return [];
		}

		this.toolDefinitions = this.mapToolsToDefinitions(client.getTools());
		return this.toolDefinitions;
	}

	consumeContextInjections(convId: string): ApiChatMessageData[] {
		const fragments = this.pendingContext.get(convId) ?? [];
		if (!fragments.length) {
			return [];
		}

		this.pendingContext.delete(convId);
		return fragments.map((fragment) => ({
			role: 'system',
			content: fragment.content
		}));
	}

	async processToolCalls(
		message: DatabaseMessage,
		serializedToolCalls?: string | null
	): Promise<number> {
		if (!browser || !serializedToolCalls?.trim()) {
			return 0;
		}

		const toolCalls = this.parseToolCalls(serializedToolCalls);
		if (!toolCalls.length) {
			return 0;
		}

		const currentConfig = config();
		const client = await this.ensureClient(currentConfig);
		if (!client) {
			return 0;
		}

		let executedCount = 0;
		const executions: MCPToolExecutionResult[] = [];
		this.pendingMessages.set(message.id, true);

		try {
			for (const call of toolCalls) {
				if (!call?.function?.name) {
					continue;
				}

				if (!(await this.requestExecutionApproval(call.function.name, currentConfig))) {
					break;
				}

				const args = this.parseArguments(call.function.arguments);

				try {
					const rawResult = await client.callTool(call.function.name, args);
					const execution = this.buildExecution(call, rawResult, currentConfig);
					executions.push(execution);
					executedCount += 1;
				} catch (error) {
					const messageText = error instanceof Error ? error.message : 'Unknown MCP error';
					this.lastError = messageText;
					toast.error(`Tool ${call.function.name} failed: ${messageText}`);
				}
			}
		} finally {
			this.pendingMessages.delete(message.id);
		}

		if (executions.length) {
			this.toolExecutions.set(message.id, executions);
			const fragments = this.buildContextFragments(executions, currentConfig);
			if (fragments.length) {
				const existing = this.pendingContext.get(message.convId) ?? [];
				existing.push(...fragments);
				this.pendingContext.set(message.convId, existing);
			}
		}

		return executedCount;
	}

	private parseToolCalls(payload: string): ApiChatCompletionToolCall[] {
		try {
			const parsed = JSON.parse(payload) as ApiChatCompletionToolCall[];
			return Array.isArray(parsed) ? parsed.filter(Boolean) : [];
		} catch (error) {
			console.warn('Failed to parse tool calls payload:', error);
			return [];
		}
	}

	private mapToolsToDefinitions(tools: MCPToolDescription[]): ApiChatCompletionToolDefinition[] {
		if (!Array.isArray(tools) || tools.length === 0) {
			return [];
		}

		return tools.map((tool) => {
			const parameters: Record<string, unknown> =
				tool.inputSchema && Object.keys(tool.inputSchema).length > 0
					? tool.inputSchema
					: { type: 'object', properties: {} };

			const definition: ApiChatCompletionToolDefinition = {
				type: 'function',
				function: {
					name: tool.name,
					description: tool.description || undefined,
					parameters
				}
			};

			return definition;
		});
	}

	private async ensureClient(settings: SettingsConfigType): Promise<MCPClient | null> {
		if (!browser) {
			return null;
		}

		const endpoint = ((settings.mcpEndpointUrl as string) || './mcp').trim() || './mcp';
		const transport = (settings.mcpTransport as 'sse' | 'websocket') || 'sse';
		const connectionKey = `${transport}:${endpoint}`;

		if (!this.client || this.activeConnectionKey !== connectionKey) {
			if (this.client) {
				void this.client.disconnect();
			}

			this.client = new MCPClient({
				endpointUrl: endpoint,
				transport,
				timeoutMs: MCP_REQUEST_TIMEOUT_MS,
				autoReconnect: true
			});
			this.activeConnectionKey = connectionKey;
			this.status = 'connecting';
			this.toolDefinitions = [];
		}

		if (!this.client) {
			return null;
		}

		try {
			await this.client.ensureInitialized();
			this.toolDefinitions = this.mapToolsToDefinitions(this.client.getTools());
			this.status = 'ready';
			this.lastError = null;
			return this.client;
		} catch (error) {
			this.status = 'error';
			const message = error instanceof Error ? error.message : 'Failed to initialize MCP client';
			this.lastError = message;
			toast.error(`MCP connection failed: ${message}`);
			return null;
		}
	}

	private parseArguments(rawArgs?: string): string | Record<string, unknown> | undefined {
		if (!rawArgs) {
			return undefined;
		}

		try {
			const parsed = JSON.parse(rawArgs);
			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				return parsed as Record<string, unknown>;
			}

			return JSON.stringify(parsed);
		} catch {
			return rawArgs;
		}
	}

	private buildExecution(
		call: ApiChatCompletionToolCall,
		rawResult: unknown,
		settings: SettingsConfigType
	): MCPToolExecutionResult {
		const toolName = call.function?.name || 'unknown_tool';
		const timestamp = Date.now();
		const maxPreviewLines = Number(settings.mcpMaxPreviewLines) || 20;

		const uniqueId =
			typeof globalThis.crypto !== 'undefined' && typeof globalThis.crypto.randomUUID === 'function'
				? globalThis.crypto.randomUUID()
				: `${timestamp}-${Math.random()}`;

		const execution: MCPToolExecutionResult = {
			id: uniqueId,
			callId: call.id,
			toolName,
			arguments: this.parseArguments(call.function?.arguments),
			preview: '',
			fullOutput: '',
			kind: 'text',
			timestamp
		};

		const { textOutput, imageDataUrl, metadata } = this.extractResultContent(rawResult);
		execution.fullOutput = textOutput || '[No textual result returned]';
		execution.preview = this.buildPreview(execution.fullOutput, maxPreviewLines);
		if (imageDataUrl) {
			execution.kind = 'image';
			execution.imageDataUrl = imageDataUrl;
		}
		if (metadata) {
			execution.metadata = metadata;
		}

		if (!execution.preview && execution.kind === 'image') {
			execution.preview = '[Image result ready]';
		}

		return execution;
	}

	private extractResultContent(rawResult: unknown): {
		textOutput: string;
		imageDataUrl?: string;
		metadata?: MCPToolExecutionResult['metadata'];
	} {
		if (!rawResult) {
			return { textOutput: '' };
		}

		if (typeof rawResult === 'string') {
			return { textOutput: rawResult };
		}

		if (typeof rawResult !== 'object') {
			return { textOutput: String(rawResult) };
		}

		const record = rawResult as Record<string, unknown>;
		const content = record.content;

		if (Array.isArray(content)) {
			const textParts: string[] = [];
			let imageDataUrl: string | undefined;
			let metadata: MCPToolExecutionResult['metadata'];

			for (const item of content) {
				if (!item || typeof item !== 'object') continue;
				const typed = item as Record<string, unknown>;
				const type = typed.type;
				if (type === 'text' && typeof typed.text === 'string') {
					textParts.push(typed.text);
				} else if (type === 'image' && typeof typed.data === 'string') {
					const mimeType = typeof typed.mimeType === 'string' ? typed.mimeType : 'image/png';
					imageDataUrl = `data:${mimeType};base64,${typed.data}`;
					metadata = {
						mimeType,
						sizeBytes: typeof typed.size === 'number' ? typed.size : undefined,
						width: typeof typed.width === 'number' ? typed.width : undefined,
						height: typeof typed.height === 'number' ? typed.height : undefined
					};
				} else if (type === 'resource' && typeof typed.resource === 'object') {
					textParts.push(JSON.stringify(typed.resource));
				}
			}

			return {
				textOutput: textParts.join('\n').trim(),
				imageDataUrl,
				metadata
			};
		}

		if (typeof record.result === 'string') {
			return { textOutput: record.result };
		}

		if (record.result !== undefined) {
			return { textOutput: JSON.stringify(record.result, null, 2) };
		}

		return { textOutput: JSON.stringify(rawResult, null, 2) };
	}

	private buildPreview(fullOutput: string, maxLines: number): string {
		if (!fullOutput.trim()) {
			return '';
		}

		const normalizedLines = fullOutput.split(/\r?\n/);
		if (maxLines <= 0 || normalizedLines.length <= maxLines) {
			return fullOutput;
		}

		const sliced = normalizedLines.slice(0, maxLines);
		sliced.push('…');
		return sliced.join('\n');
	}

	private buildContextFragments(
		executions: MCPToolExecutionResult[],
		settings: SettingsConfigType
	): MCPContextFragment[] {
		const fragments: MCPContextFragment[] = [];

		for (const execution of executions) {
			const content = this.formatContextContent(execution, settings);
			if (content) {
				fragments.push({
					id: execution.id,
					content
				});
			}
		}

		return fragments;
	}

	private formatContextContent(
		execution: MCPToolExecutionResult,
		settings: SettingsConfigType
	): string | null {
		if (execution.kind === 'image') {
			if ((settings.mcpImageResultBehavior as string) === 'none') {
				return null;
			}

			const parts: string[] = [];
			if (settings.mcpIncludeToolName) {
				parts.push(`Tool: ${execution.toolName}`);
			}
			if (settings.mcpIncludeTimestamp) {
				parts.push(new Date(execution.timestamp).toISOString());
			}

			const metadata = execution.metadata;
			const descriptors: string[] = [];
			if (metadata?.mimeType) {
				descriptors.push(metadata.mimeType);
			}
			if (metadata?.width && metadata?.height) {
				descriptors.push(`${metadata.width}x${metadata.height}`);
			}
			if (metadata?.sizeBytes) {
				descriptors.push(`${Math.round(metadata.sizeBytes / 1024)}KB`);
			}

			const header = parts.length ? `[${parts.join(' · ')}]` : '';
			const body = descriptors.length
				? `[Image result: ${descriptors.join(', ')}]`
				: '[Image result]';
			return `${header}${header ? '\n' : ''}${body}`;
		}

		const behavior = (settings.mcpTextResultBehavior as string) || 'summary';
		if (behavior === 'none') {
			return null;
		}

		const limitBytes = Number(settings.mcpMaxContextBytes) || 0;
		let payload = execution.fullOutput;

		if (behavior === 'truncated') {
			payload = this.truncateToBytes(payload, limitBytes);
		} else if (behavior === 'summary') {
			payload = this.buildSummary(payload, limitBytes || 512);
		} else if (limitBytes > 0) {
			payload = this.truncateToBytes(payload, limitBytes);
		}

		const headerParts: string[] = [];
		if (settings.mcpIncludeToolName) {
			headerParts.push(`Tool: ${execution.toolName}`);
		}
		if (settings.mcpIncludeTimestamp) {
			headerParts.push(new Date(execution.timestamp).toISOString());
		}

		if (headerParts.length === 0) {
			return payload;
		}

		return `[${headerParts.join(' · ')}]\n${payload}`;
	}

	private truncateToBytes(text: string, maxBytes: number): string {
		if (maxBytes <= 0) {
			return text;
		}

		const encoder = new TextEncoder();
		const data = encoder.encode(text);
		if (data.length <= maxBytes) {
			return text;
		}

		const truncated = data.slice(0, maxBytes);
		const decoder = new TextDecoder();
		return `${decoder.decode(truncated)}…`;
	}

	private buildSummary(text: string, maxBytes: number): string {
		const truncated = this.truncateToBytes(text, maxBytes);
		if (truncated === text) {
			return truncated;
		}

		const newlineIndex = truncated.indexOf('\n');
		const firstLine = newlineIndex === -1 ? truncated : truncated.slice(0, newlineIndex);
		return `${firstLine.trim()}… (summary)`;
	}

	private async requestExecutionApproval(
		toolName: string,
		settings: SettingsConfigType
	): Promise<boolean> {
		if (settings.mcpAutoExecuteTools) {
			return true;
		}

		return window.confirm(`Allow MCP tool "${toolName}" to run?`);
	}
}

export const mcpStore = new MCPStore();
export const getToolExecutionsForMessage = (messageId: string) =>
	mcpStore.getExecutionsForMessage(messageId);
export const isMcpMessageRunning = (messageId: string) => mcpStore.isMessageRunning(messageId);
