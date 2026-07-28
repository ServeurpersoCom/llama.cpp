import { FilesystemService } from '$lib/services';
import { ApiError, debounce } from '$lib/utils';
import { markBrowseEndpointDisabled } from './browse-roots.svelte';
import type { ApiFilesystemSearchEntry, ApiFilesystemSearchRequest } from '$lib/types';

interface FilesystemSearchOptions {
	// read at fetch time so reactive inputs (scope path, configured depth)
	// are picked up on every call
	params: () => Omit<ApiFilesystemSearchRequest, 'query'>;
	// called with the fresh results on success, and with [] whenever they are cleared
	onResults?: (entries: ApiFilesystemSearchEntry[]) => void;
}

/**
 * Search state machine shared by the filesystem pickers (working-directory
 * chip, @-mention picker): debounced query, AbortController + sequence
 * guard against stale responses, 501s forwarded to the shared browse-roots
 * store (the search and roots endpoints are gated by the same server flag).
 *
 * `schedule(query)` flips `isLoading` synchronously so callers can render a
 * skeleton during the debounce window; `reset()` drops results and cancels
 * everything, including a debounced fire that has not run yet.
 */
export function createFilesystemSearch(options: FilesystemSearchOptions) {
	let results = $state<ApiFilesystemSearchEntry[]>([]);
	let isLoading = $state(false);
	let error = $state<string | null>(null);

	let controller: AbortController | null = null;
	let seq = 0;
	let pendingQuery: string | null = null;

	function cancel() {
		controller?.abort();
		seq++;
		isLoading = false;
	}

	function reset() {
		pendingQuery = null;
		cancel();
		results = [];
		error = null;
		options.onResults?.([]);
	}

	async function search(query: string) {
		cancel();
		const ctrl = new AbortController();
		controller = ctrl;
		const mySeq = ++seq;

		isLoading = true;
		try {
			const response = await FilesystemService.search(
				{ ...options.params(), query: query.trim() },
				ctrl.signal
			);
			if (mySeq !== seq) return;
			results = response.results;
			error = null;
			options.onResults?.(results);
		} catch (err) {
			if (mySeq !== seq) return;
			results = [];
			options.onResults?.([]);
			if (ctrl.signal.aborted) return;
			if (err instanceof ApiError && err.status === 501) {
				markBrowseEndpointDisabled();
				error = null;
			} else {
				error = err instanceof Error ? err.message : String(err);
			}
		} finally {
			if (mySeq === seq) isLoading = false;
		}
	}

	const fire = debounce((q: string) => {
		if (q !== pendingQuery) return;
		void search(q);
	}, 180);

	function schedule(query: string) {
		pendingQuery = query;
		cancel();
		isLoading = true;
		fire(query);
	}

	return {
		get results() {
			return results;
		},
		get isLoading() {
			return isLoading;
		},
		get error() {
			return error;
		},
		schedule,
		reset
	};
}
