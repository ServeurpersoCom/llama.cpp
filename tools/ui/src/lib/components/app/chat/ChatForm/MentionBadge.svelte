<script lang="ts">
	import { Folder } from '@lucide/svelte';

	interface Props {
		class?: string;
		href?: string;
		name: string;
		path: string;
	}

	let { class: className = '', href, name, path }: Props = $props();
</script>

{#if href}
	<a
		{href}
		target="_blank"
		rel="noopener noreferrer"
		data-href={href}
		title={path}
		class={[
			'mention-badge-link inline-flex w-fit shrink-0 items-center gap-1 whitespace-nowrap rounded-md border border-border/50 bg-foreground/15 px-1.5 py-0.5 text-xs font-mono text-foreground hover:bg-foreground/25 dark:bg-foreground/10 dark:text-secondary-foreground',
			className
		]}
	>
		<Folder class="h-3 w-3 shrink-0" aria-hidden="true" />
		<span class="shrink-0 truncate">{name}</span>
	</a>
{:else}
	<span
		data-mention-badge="true"
		data-mention-name={name}
		data-mention-path={path}
		title={path}
		class={[
			'chat-form-mention-badge inline-flex w-fit shrink-0 items-center gap-1 whitespace-nowrap rounded-md border border-border/50 bg-foreground/15 px-1.5 py-0.5 text-xs font-mono text-foreground dark:bg-foreground/10 dark:text-secondary-foreground',
			className
		]}
	>
		<Folder class="h-3 w-3 shrink-0" aria-hidden="true" />
		<span class="shrink-0 truncate">{name}</span>
	</span>
{/if}
