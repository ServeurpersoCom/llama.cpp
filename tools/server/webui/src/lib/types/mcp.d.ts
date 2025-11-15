export type MCPJsonRpcId = number | string;

export interface MCPJsonRpcRequest {
	jsonrpc: '2.0';
	id?: MCPJsonRpcId;
	method: string;
	params?: Record<string, unknown>;
}

export interface MCPJsonRpcError {
	code: number;
	message: string;
	data?: unknown;
}

export interface MCPJsonRpcResponse {
	jsonrpc: '2.0';
	id: MCPJsonRpcId;
	result?: unknown;
	error?: MCPJsonRpcError;
}

export interface MCPToolDescription {
	name: string;
	description?: string;
	inputSchema?: Record<string, unknown>;
}

export interface MCPToolExecutionMetadata {
	mimeType?: string;
	sizeBytes?: number;
	width?: number;
	height?: number;
}

export type MCPToolExecutionKind = 'text' | 'image' | 'unknown';

export interface MCPToolExecutionResult {
	id: string;
	callId?: string;
	toolName: string;
	arguments?: Record<string, unknown> | string;
	preview: string;
	fullOutput: string;
	kind: MCPToolExecutionKind;
	timestamp: number;
	imageDataUrl?: string;
	metadata?: MCPToolExecutionMetadata;
}

export interface MCPContextFragment {
	id: string;
	content: string;
}
