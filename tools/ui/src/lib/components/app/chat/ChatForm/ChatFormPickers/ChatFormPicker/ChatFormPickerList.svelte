<script lang="ts" generics="T">
	import type { Snippet } from 'svelte';
	import { SearchInput } from '$lib/components/app';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import { CHAT_FORM_POPOVER_MAX_HEIGHT } from '$lib/constants';

	interface Props {
		items: T[];
		isLoading: boolean;
		selectedIndex: number;
		searchQuery: string;
		showSearchInput: boolean;
		searchPlaceholder?: string;
		emptyMessage?: string;
		autofocus?: boolean;
		inputRef?: HTMLInputElement | null;
		onSearchClose?: () => void;
		itemKey: (item: T, index: number) => string;
		item: Snippet<[T, number, boolean]>;
		skeleton?: Snippet;
		footer?: Snippet;
	}

	let {
		items,
		isLoading,
		selectedIndex,
		searchQuery = $bindable(),
		showSearchInput,
		searchPlaceholder = 'Search...',
		emptyMessage = 'No items available',
		autofocus = false,
		inputRef = $bindable(null),
		onSearchClose,
		itemKey,
		item,
		skeleton,
		footer
	}: Props = $props();

	let listContainer = $state<HTMLDivElement | null>(null);

	/**
	 * Keep the keyboard-selected row visible without jerk on hover.
	 *
	 * The picker feeds `selectedIndex` from two sources:
	 *  - mouse hover (every onmouseenter bumps the index)
	 *  - keyboard ArrowUp / ArrowDown (only on key events)
	 *
	 * With `block: 'center'` the browser scrolled on every hover, which
	 * visibly drifted the row the user pointed at when it sat near the
	 * top or bottom edge. Using `block: 'nearest'` makes the call a
	 * no-op for any row that is already fully visible - hovering only
	 * updates the highlight; keyboard nav that actually exits the visible
	 * area still snaps to the nearest edge.
	 */
	$effect(() => {
		if (listContainer && selectedIndex >= 0 && selectedIndex < items.length) {
			const selectedElement = listContainer.querySelector(
				`[data-picker-index="${selectedIndex}"]`
			) as HTMLElement;

			if (selectedElement) {
				selectedElement.scrollIntoView({
					block: 'nearest',
					inline: 'nearest'
				});
			}
		}
	});
</script>

<ScrollArea>
	{#if showSearchInput}
		<div class="absolute top-0 right-0 left-0 z-10 p-2 pb-0">
			<SearchInput
				{autofocus}
				placeholder={searchPlaceholder}
				bind:value={searchQuery}
				bind:ref={inputRef}
				onClose={onSearchClose}
			/>
		</div>
	{/if}

	<div
		bind:this={listContainer}
		class={[`${CHAT_FORM_POPOVER_MAX_HEIGHT} p-2`, showSearchInput && 'pt-13']}
	>
		{#if isLoading}
			{#if skeleton}
				{@render skeleton()}
			{/if}
		{:else if items.length === 0}
			<div class="py-6 text-center text-sm text-muted-foreground">{emptyMessage}</div>
		{:else}
			{#each items as itemData, index (itemKey(itemData, index))}
				{@render item(itemData, index, index === selectedIndex)}
			{/each}
		{/if}
	</div>

	{#if footer}
		{@render footer()}
	{/if}
</ScrollArea>
