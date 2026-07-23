<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { Folder, FolderOpen, GitBranch, X } from '@lucide/svelte';
	import { fly } from 'svelte/transition';
	import { FilesystemService } from '$lib/services';
	import { ApiError } from '$lib/utils';
	import { debounce } from '$lib/utils/debounce';
	import * as Popover from '$lib/components/ui/popover';
	import { Button } from '$lib/components/ui/button';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { cn } from '$lib/components/ui/utils';
	import type { ApiFilesystemRoot, ApiFilesystemSearchEntry } from '$lib/types';

	interface Props {
		class?: string;
		disabled?: boolean;
		directory?: string | null;
		onChange?: (directory: string | null) => void;
	}

	let {
		class: className = '',
		disabled = false,
		directory = $bindable(null),
		onChange
	}: Props = $props();

	// File System Access API is opt-in: when available (Chrome / Edge / Opera) the popover
	// exposes a "Browse" button that opens the native folder picker. When unavailable the
	// popover still works via the text input - no alerts, no upload semantics.
	const pickerSupported =
		typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';

	// Popover open state. The popover element handles outside-click and Escape;
	// we just react to open and seed the search field with the active path.
	let isOpen = $state(false);
	let inputValue = $state('');
	let searchInputRef: HTMLInputElement | null = $state(null);

	// Search / autocomplete state
	let queryResults = $state<ApiFilesystemSearchEntry[]>([]);
	let isSearching = $state(false);
	let searchError = $state<string | null>(null);
	let endpointDisabled = $state(false);
	let hoveredIndex = $state(-1);
	let showHidden = $state(false);

	// Browse roots loaded once per session; default root anchors the search.
	let roots = $state<ApiFilesystemRoot[] | null>(null);
	let loadingRoots = $state(false);
	let rootsError = $state<string | null>(null);

	let defaultRootPath = $derived.by(() => {
		if (!roots || roots.length === 0) return null;
		const def = roots.find((r) => r.default);
		return def ? def.path : roots[0].path;
	});

	// Label on the trigger button: full path when set, ghost prompt otherwise.
	let displayLabel = $derived(directory ?? 'Select working directory');

	// Git metadata for the picked directory. Probed by the server walking up
	// from `directory` looking for `.git/`. Updated whenever the active
	// `directory` changes; stale responses from earlier paths are dropped.
	let gitInfo = $state<{ is_repo: boolean; branch: string } | null>(null);
	let gitController: AbortController | null = null;
	let gitSeq = 0;

	$effect(() => {
		const path = directory;

		// Cancel any in-flight pose for a previous directory before kicking
		// off the next one.
		gitController?.abort();
		gitSeq++;

		if (!path) {
			gitInfo = null;
			return;
		}

		const controller = new AbortController();
		gitController = controller;
		const mySeq = gitSeq;

		FilesystemService.getGitInfo({ path }, controller.signal)
			.then((response) => {
				if (mySeq !== gitSeq) return;
				gitInfo = response.is_repo ? { is_repo: true, branch: response.branch } : null;
			})
			.catch((err: unknown) => {
				if (mySeq !== gitSeq) return;
				// 501 from servers without --tools / --agent is a normal
				// operational state; just hide the branch badge silently.
				if (err instanceof ApiError && err.status === 501) {
					gitInfo = null;
					return;
				}
				if (controller.signal.aborted) return;
				gitInfo = null;
			});
	});

	// AbortController + sequence counter to discard stale responses when the user
	// keeps typing; a newer call aborts the previous one. The sequence counter
	// also covers the gap between abort and the catch handler.
	let searchController: AbortController | null = null;
	let searchSeq = 0;

	const runSearch = debounce((query: string) => {
		void doSearch(query);
	}, 180);

	async function ensureRoots() {
		if (roots !== null || loadingRoots) return;
		loadingRoots = true;
		rootsError = null;
		try {
			const res = await FilesystemService.getRoots();
			roots = res.roots;
		} catch (err) {
			if (err instanceof ApiError && err.status === 501) {
				roots = [];
				endpointDisabled = true;
			} else {
				roots = [];
				rootsError = err instanceof Error ? err.message : String(err);
			}
		} finally {
			loadingRoots = false;
		}
	}

	function cancelSearch() {
		searchController?.abort();
		searchSeq++;
		isSearching = false;
	}

	async function doSearch(query: string) {
		const trimmed = query.trim();
		if (!trimmed) {
			queryResults = [];
			searchError = null;
			isSearching = false;
			hoveredIndex = -1;
			return;
		}

		cancelSearch();
		const controller = new AbortController();
		searchController = controller;
		const mySeq = ++searchSeq;

		isSearching = true;
		try {
			const response = await FilesystemService.search(
				{
					query: trimmed,
					type: 'directory',
					path: defaultRootPath ?? '',
					limit: 20,
					max_depth: 6,
					show_hidden: showHidden
				},
				controller.signal
			);
			if (mySeq !== searchSeq) return;
			queryResults = response.results;
			hoveredIndex = -1;
			searchError = null;
		} catch (err) {
			if (mySeq !== searchSeq) return;
			queryResults = [];
			hoveredIndex = -1;
			if (controller.signal.aborted) return;
			if (err instanceof ApiError && err.status === 501) {
				endpointDisabled = true;
				searchError = null;
			} else {
				searchError = err instanceof Error ? err.message : String(err);
			}
		} finally {
			if (mySeq === searchSeq) isSearching = false;
		}
	}

	function commit(entry: ApiFilesystemSearchEntry) {
		directory = entry.path;
		onChange?.(entry.path);
		isOpen = false;
	}

	function setDirectory(value: string) {
		const trimmed = value.trim();
		if (!trimmed) return;
		directory = trimmed;
		onChange?.(trimmed);
	}

	// Resolve a folder name picked via the browser-native picker (which exposes
	// only the leaf name) to a server-side absolute path. Falls back to the
	// leaf name when the server cannot locate a matching directory.
	async function resolveNativeName(name: string): Promise<string> {
		if (endpointDisabled || !defaultRootPath) return name;
		try {
			const res = await FilesystemService.search(
				{
					query: name,
					type: 'directory',
					path: defaultRootPath,
					limit: 1,
					max_depth: 4
				},
				new AbortController().signal
			);
			const match = res.results[0];
			return match && match.name === name ? match.path : name;
		} catch {
			return name;
		}
	}

	async function browseNative() {
		if (disabled || !window.showDirectoryPicker) return;
		try {
			const handle = await window.showDirectoryPicker();
			const path = await resolveNativeName(handle.name);
			setDirectory(path);
			isOpen = false;
		} catch (err) {
			// user cancelled - silently ignore; other errors are logged
			if (err instanceof DOMException && err.name === 'AbortError') return;
			console.error('[ChatFormWorkingDirectory] showDirectoryPicker failed:', err);
		}
	}

	function handleSubmit() {
		const value = inputValue.trim();
		if (!value) {
			isOpen = false;
			return;
		}
		setDirectory(value);
		isOpen = false;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			event.preventDefault();
			handleSubmit();
		} else if (event.key === 'ArrowDown') {
			if (queryResults.length > 0) {
				event.preventDefault();
				hoveredIndex = (hoveredIndex + 1) % queryResults.length;
			}
		} else if (event.key === 'ArrowUp') {
			if (queryResults.length > 0) {
				event.preventDefault();
				hoveredIndex = hoveredIndex <= 0 ? queryResults.length - 1 : hoveredIndex - 1;
			}
		}
	}

	function handleInputInput(value: string) {
		hoveredIndex = -1;
		if (value.trim().length > 0) {
			runSearch(value);
		}
	}

	function clearDirectory(event: MouseEvent) {
		// Stop the click from bubbling into the popover trigger and re-opening
		// the picker on top of the now-cleared state.
		event.stopPropagation();
		event.preventDefault();
		directory = null;
		onChange?.(null);
		isOpen = false;
	}

	// Chip is always visible - the X just clears the picked directory and
	// reveals the empty "Select working directory" placeholder again. No-op
	// when there's already nothing to clear.
	function handleDismiss(event: MouseEvent) {
		event.stopPropagation();
		event.preventDefault();
		if (directory) {
			clearDirectory(event);
		}
	}

	function handleOpenChange(open: boolean) {
		isOpen = open;
		if (open) {
			// Seed the search field with the current path so the user can refine it
			// (or hit Enter to confirm / clear via the X icon).
			inputValue = directory ?? '';
			hoveredIndex = -1;
			queryResults = [];
			searchError = null;
			void ensureRoots();
			// Move focus to the search field on next tick so it wins over the
			// popover's own focus-stealing on open.
			queueMicrotask(() => searchInputRef?.focus({ preventScroll: true }));
		} else {
			cancelSearch();
		}
	}

	// Re-run the active query whenever the show-hidden toggle flips while the
	// popover is open.
	$effect(() => {
		void showHidden;

		if (isOpen && inputValue.trim()) {
			runSearch(inputValue);
		}
	});

	// Splits `text` into alternating segments at each case-insensitive
	// occurrence of `query`. Used by the results list to highlight the search
	// terms inside full-path strings.
	function highlightMatch(text: string, query: string): { text: string; match: boolean }[] {
		if (!query) return [{ text, match: false }];
		const segments: { text: string; match: boolean }[] = [];
		const lowerText = text.toLowerCase();
		const lowerQuery = query.toLowerCase();
		let i = 0;
		while (i < text.length) {
			const idx = lowerText.indexOf(lowerQuery, i);
			if (idx < 0) {
				segments.push({ text: text.slice(i), match: false });
				break;
			}
			if (idx > i) segments.push({ text: text.slice(i, idx), match: false });
			segments.push({ text: text.slice(idx, idx + query.length), match: true });
			i = idx + query.length;
		}
		return segments;
	}

	// Imperative API: opens the picker without requiring the chip's own
	// trigger to be clicked. Used by ChatForm so picking the "Working
	// Directory" item from the Add dropdown reveals the chip and instantly
	// drops the user into the picker.
	export function openPicker() {
		isOpen = true;
	}
</script>

{#snippet resultsList()}
	<div
		class="max-h-48 overflow-y-auto rounded-md border border-border/40 bg-popover"
		transition:fly={{ y: -4, duration: 100 }}
	>
		{#if isSearching && queryResults.length === 0}
			<div class="px-3 py-2 text-xs text-muted-foreground/70">Searching...</div>
		{:else if endpointDisabled}
			<div class="px-3 py-2 text-xs text-muted-foreground/70">
				Filesystem browsing is disabled. Start the server with
				<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--tools</code>
				or
				<code class="rounded bg-muted px-1 py-0.5 text-[10px]">--agent</code>
				to enable it.
			</div>
		{:else if searchError}
			<div class="px-3 py-2 text-xs text-destructive">{searchError}</div>
		{:else if queryResults.length === 0}
			<div class="px-3 py-2 text-xs text-muted-foreground/70">No matching folders</div>
		{:else}
			{#each queryResults as entry, index (entry.path)}
				<button
					type="button"
					data-result-index={index}
					class={cn(
						'flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors',
						index === hoveredIndex ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/50'
					)}
					onclick={() => commit(entry)}
					onmouseenter={() => (hoveredIndex = index)}
				>
					<Folder class="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
					<span class="min-w-0 flex-1 truncate font-mono">
						{#each highlightMatch(entry.path, inputValue.trim()) as seg, segIndex (segIndex)}
							{#if seg.match}
								<mark class="rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30"
									>{seg.text}</mark
								>
							{:else}
								{seg.text}
							{/if}
						{/each}
					</span>
				</button>
			{/each}
		{/if}
	</div>
{/snippet}

<div class={cn('flex min-w-0 w-full items-center gap-1 pt-3 px-1.5', className)}>
	<Popover.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
		<Popover.Trigger {disabled} class="w-full flex justify-start">
			<span class="text-muted-foreground inline-flex items-center gap-1.5 text-xs group" class:text-foreground={directory}>
				<div class="flex min-w-0 items-center gap-1">
    				<Folder class="w-3.5 h-3.5" />

					<span class="max-w-64 truncate">{displayLabel}</span>

					{#if gitInfo?.is_repo && gitInfo.branch}
						<span
							class="inline-flex shrink-0 items-center gap-1 rounded-full bg-muted/70 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
							title="Git branch on disk"
						>
							<GitBranch class="h-2.5 w-2.5" />
							<span>{gitInfo.branch}</span>
						</span>
					{/if}
				</div>

					{#if directory}
    					<button
    						type="button"
    						class="inline-flex h-4 w-4 shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100"
    						onclick={handleDismiss}
    						{disabled}
    						tabindex={-1}
    						aria-label={directory ? 'Clear working directory' : 'Hide working directory'}
    					>
    						<X class="h-3 w-3" />
    					</button>
					{/if}
			</span>
		</Popover.Trigger>

		<Popover.Content
			side="top"
			align="start"
			sideOffset={12}
			class="space-y-2 rounded-xl border-border/50 p-3 shadow-xl"
			onkeydown={handleKeydown}
			onOpenAutoFocus={(event) => event.preventDefault()}
		>
			<SearchInput
				bind:ref={searchInputRef}
				bind:value={inputValue}
				placeholder="Choose working directory"
				onInput={handleInputInput}
				onClose={() => (isOpen = false)}
				onKeyDown={handleKeydown}
				class="w-full"
			/>

			{#if inputValue.trim() && (isSearching || queryResults.length > 0 || searchError || endpointDisabled)}
				{@render resultsList()}
			{/if}

			<div class="flex items-center justify-between gap-2">
				<label class="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
					<Checkbox
						bind:checked={showHidden}
						aria-label="Show hidden directories"
						class="size-3.5"
					/>
					<span>Show hidden</span>
				</label>

				{#if pickerSupported}
					<Button type="button" variant="outline" size="sm" class="shrink-0" onclick={browseNative}>
						<FolderOpen class={ICON_CLASS_DEFAULT} />
						<span>Browse</span>
					</Button>
				{/if}
			</div>

			{#if defaultRootPath}
				<div
					class="truncate font-mono text-[10px] text-muted-foreground/70"
					title="Search is bounded to this scope"
				>
					{defaultRootPath}
				</div>
			{:else if rootsError}
				<div class="text-[10px] text-destructive">
					Cannot load browse roots - {rootsError}
				</div>
			{/if}
		</Popover.Content>
	</Popover.Root>
</div>
