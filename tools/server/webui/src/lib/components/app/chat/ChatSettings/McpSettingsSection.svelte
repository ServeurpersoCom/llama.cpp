<script lang="ts">
	import { Plus, Trash2 } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		detectMcpTransportFromUrl,
		parseMcpServerSettings,
		type MCPServerSettingsEntry
	} from '$lib/config/mcp';
	import type { SettingsConfigType } from '$lib/types/settings';

	interface Props {
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { localConfig, onConfigChange }: Props = $props();

	const DEFAULT_TIMEOUT_SECONDS = 300;

	function serializeServers(servers: MCPServerSettingsEntry[]) {
		onConfigChange('mcpServers', JSON.stringify(servers));
	}

	function getServers(): MCPServerSettingsEntry[] {
		return parseMcpServerSettings(localConfig.mcpServers);
	}

	function addServer() {
		const servers = getServers();
		const fallbackTimeoutSeconds =
			servers[servers.length - 1]?.requestTimeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS;
		const newServer: MCPServerSettingsEntry = {
			id: crypto.randomUUID ? crypto.randomUUID() : `server-${Date.now()}`,
			enabled: true,
			url: '',
			requestTimeoutSeconds: fallbackTimeoutSeconds
		};

		serializeServers([...servers, newServer]);
	}

	function updateServer(id: string, updates: Partial<MCPServerSettingsEntry>) {
		const servers = getServers();
		const nextServers = servers.map((server) =>
			server.id === id
				? {
						...server,
						...updates
					}
				: server
		);

		serializeServers(nextServers);
	}

	function removeServer(id: string) {
		const servers = getServers().filter((server) => server.id !== id);
		serializeServers(servers);
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between gap-4">
		<div>
			<h4 class="text-base font-semibold">MCP Servers</h4>
			<p class="text-sm text-muted-foreground">
				Configure one or more MCP Servers. Only enabled servers with a URL are used.
			</p>
		</div>

		<Button variant="outline" class="shrink-0" onclick={addServer}>
			<Plus class="mr-2 h-4 w-4" />
			Add MCP Server
		</Button>
	</div>

	{#if getServers().length === 0}
		<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
			No MCP Servers configured yet. Add one to enable agentic features.
		</div>
	{/if}

	<div class="space-y-3">
		{#each getServers() as server, index (server.id)}
			<div class="space-y-3 rounded-lg border p-4 shadow-sm">
				<div class="flex flex-wrap items-center gap-3">
					<div class="flex items-center gap-2">
						<Checkbox
							id={`mcp-enabled-${server.id}`}
							checked={server.enabled}
							onCheckedChange={(checked) =>
								updateServer(server.id, {
									enabled: Boolean(checked)
								})}
						/>
						<div class="space-y-1">
							<Label for={`mcp-enabled-${server.id}`} class="cursor-pointer text-sm font-medium">
								MCP Server {index + 1}
							</Label>
							<p class="text-xs text-muted-foreground">
								{detectMcpTransportFromUrl(server.url) === 'websocket'
									? 'WebSocket'
									: 'Streamable HTTP'}
							</p>
						</div>
					</div>

					<div class="ml-auto flex items-center gap-2">
						<Button
							variant="ghost"
							size="icon"
							class="text-muted-foreground hover:text-foreground"
							onclick={() => removeServer(server.id)}
							aria-label={`Remove MCP Server ${index + 1}`}
						>
							<Trash2 class="h-4 w-4" />
						</Button>
					</div>
				</div>

				<div class="space-y-3">
					<div class="space-y-2">
						<Label class="text-sm font-medium">Endpoint URL</Label>
						<Input
							value={server.url}
							placeholder="http://127.0.0.1:8080"
							class="w-full"
							oninput={(event) =>
								updateServer(server.id, {
									url: event.currentTarget.value
								})}
						/>
					</div>

					<div class="space-y-2 md:min-w-[14rem]">
						<Label class="text-sm font-medium">Request timeout (seconds)</Label>
						<Input
							type="number"
							min="1"
							inputmode="numeric"
							value={String(server.requestTimeoutSeconds ?? '')}
							class="w-full md:max-w-[14rem]"
							oninput={(event) => {
								const parsed = Number(event.currentTarget.value);
								updateServer(server.id, {
									requestTimeoutSeconds:
										Number.isFinite(parsed) && parsed > 0
											? parsed
											: server.requestTimeoutSeconds || DEFAULT_TIMEOUT_SECONDS
								});
							}}
						/>
					</div>
				</div>
			</div>
		{/each}
	</div>
</div>
