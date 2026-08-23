/**
 * Hugging Face Hub types for the model browser.
 */

/** one repo of a Hub search result */
export interface HfModelSummary {
	id: string;
	downloads: number;
	likes: number;
	gated: boolean | string;
	lastModified?: string;
}

/** one entry of the recursive repo tree */
export interface HfRepoFile {
	type: string;
	path: string;
	size: number;
}

/**
 * One downloadable model of a repo, its shards folded into a single entry.
 * `tag` is the selector sent back to the server, it addresses this exact file.
 */
export interface HfModelVariant {
	tag: string;
	label: string;
	size: number;
	parts: number;
}
