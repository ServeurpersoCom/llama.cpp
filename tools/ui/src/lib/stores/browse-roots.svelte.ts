import { FilesystemService } from '$lib/services';
import { ApiError } from '$lib/utils';
import type { ApiFilesystemRoot } from '$lib/types';

/**
 * Single-source-of-truth for the server's browse-roots endpoint.
 *
 * Both the working-directory picker and the file-mention picker
 * need the same data: which directories the server is willing to let the
 * client walk into, and which one the user hasn't overridden. Without
 * sharing, each picker would otherwise re-issue the fetch on its own
 * schedule, race on the disabled flag, and re-derive the default root
 * independently.
 *
 * The fetch is idempotent and promise-cached: if two consumers call
 * `ensureBrowseRoots()` on the same render pass, only one network
 * request goes out. Once resolved (success or terminal error), subsequent
 * calls are synchronous no-ops until `invalidateBrowseRoots()`.
 */

let roots = $state<ApiFilesystemRoot[] | null>(null);
let loading = $state(false);
let error = $state<string | null>(null);
let endpointDisabled = $state(false);
let loadPromise: Promise<void> | null = null;

/** Reactive read of the loaded roots. `null` until the first fetch resolves. */
export function browseRoots() {
	return roots;
}

/** Reactive read of the in-flight fetch state. */
export function browseRootsLoading() {
	return loading;
}

/** Reactive read of the last fetch error. Cleared by a successful fetch. */
export function browseRootsError() {
	return error;
}

/** Reactive read of whether the server has filesystem browsing disabled. */
export function isBrowseEndpointDisabled() {
	return endpointDisabled;
}

/**
 * Resolve the default browse root - the one server marks with `default: true`,
 * or the first root if the server didn't flag one. `null` when no roots are
 * loaded or the server returned an empty list.
 */
export function defaultBrowseRoot(): ApiFilesystemRoot | null {
	if (!roots || roots.length === 0) return null;
	return roots.find((r) => r.default) ?? roots[0];
}

/** Convenience: just the path of the default root. */
export function defaultBrowseRootPath(): string | null {
	return defaultBrowseRoot()?.path ?? null;
}

/**
 * Kick off the fetch if not already in flight or resolved. Returns the
 * in-flight (or completed) promise so callers can `await` for tests or
 * sequencing; in normal use reading the reactive getters after a
 * mount-time `void ensureBrowseRoots()` is enough.
 *
 * Behavior:
 * - 200: `roots = response.roots`. `endpointDisabled = false`.
 * - 501: server does not have `--tools` / `--agent` enabled. `roots = []`,
 *   `endpointDisabled = true`. Consumers should render a "browsing
 *   disabled" state; the search endpoint is gated by the same flag.
 * - Other errors: `roots = []`, `error = <message>`.
 */
export function ensureBrowseRoots(): Promise<void> {
	if (typeof window === 'undefined') return Promise.resolve();
	if (roots !== null) return Promise.resolve();
	if (loadPromise) return loadPromise;

	loading = true;
	error = null;
	endpointDisabled = false;
	loadPromise = (async () => {
		try {
			const res = await FilesystemService.getRoots();
			roots = res.roots;
		} catch (err) {
			if (err instanceof ApiError && err.status === 501) {
				roots = [];
				endpointDisabled = true;
			} else {
				roots = [];
				error = err instanceof Error ? err.message : String(err);
			}
		} finally {
			loading = false;
			loadPromise = null;
		}
	})();
	return loadPromise;
}

/**
 * Reset the cache so the next `ensureBrowseRoots()` re-issues the request.
 * Useful when the user's server-backed context changes (e.g. a new session).
 */
export function invalidateBrowseRoots(): void {
	roots = null;
	loading = false;
	error = null;
	endpointDisabled = false;
	loadPromise = null;
}

/**
 * Mark the endpoint as disabled without a fresh fetch - used when a
 * filesystem call (e.g. `search`) returns 501 from the search endpoint.
 * The search and roots endpoints share the same server flag, so a 501 on
 * either side implies both are off.
 */
export function markBrowseEndpointDisabled(): void {
	endpointDisabled = true;
	roots = [];
}
