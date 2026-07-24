<script lang="ts">
	import { File, Folder } from '@lucide/svelte';
	import { FilesystemService } from '$lib/services';
	import { abbreviateWorkingDir, ApiError, debounce, lastPathSegment } from '$lib/utils';
	import {
		browseRoots,
		ensureBrowseRoots,
		isBrowseEndpointDisabled
	} from '$lib/stores/browse-roots.svelte';
	import * as Popover from '$lib/components/ui/popover';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import HighlightedMatch from '$lib/components/app/forms/HighlightedMatch.svelte';
	import { ChatFormPickerList, ChatFormPickerListItem } from '$lib/components/app/chat';
	import type { ApiFilesystemSearchEntry } from '$lib/types';

	/**
	 * Floating file/folder mention picker.
	 *
	 * Opens when the user types `@<query>` at a token boundary inside the
	 * chat textarea. Returns the picked `ApiFilesystemSearchEntry` via
	 * `onSelect` so the parent can splice a `[name](file:///<abs>)<space>`
	 * markdown link into the textarea at the cursor.
	 *
	 * Closes via Escape, outside-click, or selection. The parent owns the
	 * "user dismissed this token, don't re-open until it changes" snapshot
	 * so the picker stays simple.
	 */
	interface Props {
		class?: string;
		isOpen: boolean;
		query: string;
		customAnchor?: HTMLElement | null;
		onClose: () => void;
		onSelect: (entry: ApiFilesystemSearchEntry) => void;
	}

	let {
		class: className = '',
		isOpen,
		query,
		customAnchor = null,
		onClose,
		onSelect
	}: Props = $props();

	let queryResults = $state<ApiFilesystemSearchEntry[]>([]);
	let isLoading = $state(false);
	let searchError = $state<string | null>(null);
	let hoveredIndex = $state(0);

	// Drop stale responses when the user keeps typing; both an AbortController
	// (cancels the network) and a sequence counter (covers the gap between
	// abort and the catch handler) guard against letting older results paint
	// over newer state.
	let searchController: AbortController | null = null;
	let searchSeq = 0;

	// Tooltips only on wider viewports - hover surfaces get in the way on
	// touch / narrow layouts. Same gate used elsewhere (ActionIcon, WD chip).
	let innerWidth = $state(0);
	const showTooltip = $derived(innerWidth > 768);

	const endpointDisabled = $derived(isBrowseEndpointDisabled());

	// Reactive subscription to the shared browse-roots cache so the path
	// abbreviation re-renders once the roots resolve.
	const roots = $derived(browseRoots());

	/**
	 * Display form of a search entry's path. While the roots cache is still
	 * loading (i.e. `roots` is null) we fall back to the raw path - the
	 * hard basename-only fallback in `abbreviateWorkingDir` would hide
	 * everything below the filename, defeating the disambiguation that
	 * the subtitle row exists to provide.
	 */
	function displayPath(entry: ApiFilesystemSearchEntry): string {
		return roots ? abbreviateWorkingDir(entry.path, roots) : entry.path;
	}

	$effect(() => {
		if (typeof window === 'undefined') return;
		void ensureBrowseRoots();
	});

	$effect(() => {
		if (isOpen) {
			hoveredIndex = 0;
		}
	});

	$effect(() => {
		// Re-run whenever the query string changes (parent owns it). Disabled
		// endpoint short-circuits to a single zero-result sticky state so the
		// user gets the "filesystem unavailable" message instead of a growing
		// spinner.
		if (!isOpen) {
			cancelSearch();
			return;
		}
		if (endpointDisabled) {
			queryResults = [];
			isLoading = false;
			searchError = null;
			return;
		}
		runSearch(query);
	});

	const runSearch = debounce((q: string) => {
		void doSearch(q);
	}, 180);

	function cancelSearch() {
		searchController?.abort();
		searchSeq++;
		isLoading = false;
	}

	async function doSearch(q: string) {
		const trimmed = q.trim();
		if (!trimmed) {
			queryResults = [];
			isLoading = false;
			searchError = null;
			return;
		}

		cancelSearch();
		const controller = new AbortController();
		searchController = controller;
		const mySeq = ++searchSeq;

		isLoading = true;
		try {
			const response = await FilesystemService.search(
				{
					query: trimmed,
					type: 'any',
					limit: 20,
					max_depth: 6,
					show_hidden: true
				},
				controller.signal
			);
			if (mySeq !== searchSeq) return;
			queryResults = response.results;
			searchError = null;
		} catch (err) {
			if (mySeq !== searchSeq) return;
			queryResults = [];
			if (controller.signal.aborted) return;
			searchError = err instanceof Error ? err.message : String(err);
		} finally {
			if (mySeq === searchSeq) isLoading = false;
		}
	}

	function handleSelect(entry: ApiFilesystemSearchEntry) {
		onSelect(entry);
		onClose();
	}

	export function handleKeydown(event: KeyboardEvent): boolean {
		if (!isOpen) return false;

		const results = queryResults;

		if (event.key === 'Escape') {
			event.preventDefault();
			onClose();
			return true;
		}

		if (event.key === 'ArrowDown') {
			event.preventDefault();
			if (results.length > 0) {
				hoveredIndex = (hoveredIndex + 1) % results.length;
			}
			return true;
		}

		if (event.key === 'ArrowUp') {
			event.preventDefault();
			if (results.length > 0) {
				hoveredIndex = hoveredIndex === 0 ? results.length - 1 : hoveredIndex - 1;
			}
			return true;
		}

		if (event.key === 'Enter') {
			if (results[hoveredIndex]) {
				event.preventDefault();
				handleSelect(results[hoveredIndex]);
				return true;
			}
			// No result selected - let the textarea's Enter-to-submit run.
			return false;
		}

		return false;
	}
</script>

<svelte:window bind:innerWidth />

<Popover.Root
	open={isOpen}
	onOpenChange={(open) => {
		if (!open) onClose();
	}}
>
	<!-- Invisible form-wide trigger: stops bits-ui's outside-click detector
	     from closing the picker when the user clicks inside the textarea.
	     We DO NOT use this trigger for opening (we open programmatically via
	     `open={isOpen}`) so it's tabindex=-1 + pointer-events-none + opacity-0
	     + aria-hidden. Positioning comes from `customAnchor` below, which
	     sits at the form's top edge so the popover floats above the box. -->
	<Popover.Trigger
		class="pointer-events-none absolute inset-0 opacity-0"
		tabindex={-1}
		aria-hidden="true"
	>
		<span class="sr-only">Open file mention picker</span>
	</Popover.Trigger>

	<Popover.Content
		align="start"
		side="top"
		sideOffset={12}
		{customAnchor}
		onkeydown={handleKeydown}
		onOpenAutoFocus={(event) => event.preventDefault()}
		class={[
			'w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl',
			className
		]}
	>
		{#if endpointDisabled}
			<div class="px-2 py-3 text-sm text-muted-foreground">
				Filesystem browsing is disabled. Start the server with
				<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--tools</code>
				or
				<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--agent</code>
				to enable it.
			</div>
		{:else if query.trim().length === 0}
			<div class="px-2 py-3 text-sm text-muted-foreground">Type a path or filename to search</div>
		{:else}
			{@const trimmed = query.trim()}
			<ChatFormPickerList
				items={queryResults}
				{isLoading}
				selectedIndex={hoveredIndex}
				showSearchInput={false}
				searchQuery={trimmed}
				searchPlaceholder="Search files..."
				emptyMessage={searchError
					? `Search failed - ${searchError}`
					: 'No matching files or folders'}
				itemKey={(entry) => entry.type + ':' + entry.path}
			>
				{#snippet item(entry, index, isSelected)}
					<ChatFormPickerListItem
						dataIndex={index}
						{isSelected}
						onclick={() => handleSelect(entry)}
						onmouseenter={() => (hoveredIndex = index)}
					>
						{@const Icon = entry.type === 'directory' ? Folder : File}
						<Icon
							class={[
								'mt-0.5 h-4 w-4 shrink-0',
								entry.type === 'directory' ? 'text-amber-500' : 'text-muted-foreground'
							]}
						/>
						<div class="flex min-w-0 flex-1 flex-col">
							<div class="flex min-w-0 items-center gap-2">
								{#if showTooltip}
									<Tooltip.Root>
										<Tooltip.Trigger>
											{#snippet child({ props })}
												<span {...props} class="truncate text-sm font-medium"
													>{lastPathSegment(entry.path)}</span
												>
											{/snippet}
										</Tooltip.Trigger>
										<Tooltip.Content>
											<p>{entry.path}</p>
										</Tooltip.Content>
									</Tooltip.Root>
								{:else}
									<span class="truncate text-sm font-medium">{lastPathSegment(entry.path)}</span>
								{/if}
								<span
									class="shrink-0 rounded-full bg-muted px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-muted-foreground"
								>
									{entry.type}
								</span>
							</div>
							<span class="min-w-0 flex-1 truncate font-mono text-left text-xs">
								<HighlightedMatch text={displayPath(entry)} query={trimmed} />
							</span>
						</div>
					</ChatFormPickerListItem>
				{/snippet}
			</ChatFormPickerList>
		{/if}
	</Popover.Content>
</Popover.Root>
