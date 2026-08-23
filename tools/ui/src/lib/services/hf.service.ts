/**
 * HuggingFaceService - Stateless Hugging Face Hub layer
 *
 * Searches repos and folds their GGUF files into downloadable variants. The
 * Hub is reached directly from the browser with plain fetch: apiFetch adds
 * the llama-server credentials, which have no business leaving for a third
 * party host.
 */

import {
	GGUF_COMPANION_MARKERS,
	GGUF_EXTENSION,
	GGUF_FIRST_SPLIT_INDEX,
	GGUF_QUANT_TAG_PATTERN,
	GGUF_SPLIT_PATTERN,
	HF_API_MODELS_URL,
	HF_FILE_TYPE,
	HF_GGUF_FILTER,
	HF_REPO_TAG_SEPARATOR,
	HF_SEARCH_LIMIT,
	HF_SORT_DESCENDING,
	HF_SORT_DOWNLOADS,
	HF_TREE_REVISION,
	PATH_SEPARATOR
} from '$lib/constants';
import type { HfModelSummary, HfModelVariant, HfRepoFile } from '$lib/types/hf';

interface SplitInfo {
	prefix: string;
	count: number;
	index: number;
}

export class HuggingFaceService {
	/**
	 * List the models a repo offers, shards folded into one entry each.
	 *
	 * @param repoId - repo identifier, e.g. unsloth/gemma-4-E2B-it-GGUF
	 * @param signal - abort signal
	 * @returns Variants ordered as the repo returns them
	 */
	static async listVariants(repoId: string, signal?: AbortSignal): Promise<HfModelVariant[]> {
		const url = new URL(`${HF_API_MODELS_URL}/${repoId}/tree/${HF_TREE_REVISION}?recursive=true`);
		const response = await fetch(url, { signal });

		if (!response.ok) {
			throw new Error(`Hugging Face repo listing failed: ${response.status}`);
		}

		const entries = (await response.json()) as HfRepoFile[];
		const files = entries.filter((entry) => entry.type === HF_FILE_TYPE);

		return HuggingFaceService.buildVariants(repoId, files);
	}

	/**
	 * Search the Hub for repos holding GGUF weights, most downloaded first.
	 *
	 * @param query - free text query, an empty one lists the top repos
	 * @param signal - abort signal, a new keystroke drops the pending request
	 * @returns Repo summaries
	 */
	static async searchModels(query: string, signal?: AbortSignal): Promise<HfModelSummary[]> {
		const url = new URL(HF_API_MODELS_URL);

		url.searchParams.set('search', query);
		url.searchParams.set('filter', HF_GGUF_FILTER);
		url.searchParams.set('sort', HF_SORT_DOWNLOADS);
		url.searchParams.set('direction', HF_SORT_DESCENDING);
		url.searchParams.set('limit', String(HF_SEARCH_LIMIT));

		const response = await fetch(url, { signal });

		if (!response.ok) {
			throw new Error(`Hugging Face search failed: ${response.status}`);
		}

		return (await response.json()) as HfModelSummary[];
	}

	private static basename(path: string): string {
		const index = path.lastIndexOf(PATH_SEPARATOR);

		return index === -1 ? path : path.slice(index + 1);
	}

	/**
	 * Fold the GGUF files of a repo into one entry per quantization.
	 *
	 * A repo may hold several files sharing one tag, and the server addresses
	 * them all by that single tag, so they are one choice here too. A file
	 * whose name carries no tag is left out: the server cannot name it, so it
	 * would vanish from the list once downloaded.
	 */
	private static buildVariants(repoId: string, files: HfRepoFile[]): HfModelVariant[] {
		const variants: HfModelVariant[] = [];
		const seen = new Set<string>();

		for (const file of files) {
			if (!file.path.endsWith(GGUF_EXTENSION) || HuggingFaceService.isCompanionFile(file.path)) {
				continue;
			}

			const split = HuggingFaceService.splitInfo(file.path);

			if (split.index !== GGUF_FIRST_SPLIT_INDEX) {
				continue; // the first shard stands for the whole set
			}

			const tag = HuggingFaceService.quantTag(split.prefix);

			if (!tag || seen.has(tag)) {
				continue;
			}

			seen.add(tag);

			const size = files
				.filter((sibling) => {
					if (!sibling.path.endsWith(GGUF_EXTENSION)) return false;

					const siblingSplit = HuggingFaceService.splitInfo(sibling.path);

					return siblingSplit.prefix === split.prefix && siblingSplit.count === split.count;
				})
				.reduce((total, sibling) => total + sibling.size, 0);

			variants.push({
				label: tag,
				parts: split.count,
				size,
				tag: `${repoId}${HF_REPO_TAG_SEPARATOR}${tag}`
			});
		}

		return variants;
	}

	/**
	 * A companion file sits next to a model without being one: projector,
	 * importance matrix or speculative decoding head.
	 */
	private static isCompanionFile(path: string): boolean {
		const filename = HuggingFaceService.basename(path);

		return GGUF_COMPANION_MARKERS.some((marker) => filename.includes(marker));
	}

	/**
	 * Quantization tag of a GGUF shard prefix, uppercased, empty when the name
	 * carries none. The server names a cached model after this tag, so it is
	 * the only selector that survives a download.
	 */
	private static quantTag(prefix: string): string {
		const match = prefix.match(GGUF_QUANT_TAG_PATTERN);

		return match ? match[1].toUpperCase() : '';
	}

	/**
	 * Split a GGUF path into its shard prefix, index and count. A single file
	 * model reports one part with itself as prefix.
	 */
	private static splitInfo(path: string): SplitInfo {
		const stem = path.slice(0, -GGUF_EXTENSION.length);
		const match = stem.match(GGUF_SPLIT_PATTERN);

		if (!match) {
			return { count: 1, index: GGUF_FIRST_SPLIT_INDEX, prefix: stem };
		}

		return {
			count: Number(match[2]),
			index: Number(match[1]),
			prefix: stem.slice(0, match.index)
		};
	}
}
