/**
 * Hugging Face Hub browsing constants.
 *
 * The Hub API is queried straight from the browser, it answers with CORS
 * headers. Requests carry no credentials, so gated and private repos stay
 * out of reach of this surface.
 */

export const HF_API_MODELS_URL = 'https://huggingface.co/api/models';

export const HF_TREE_REVISION = 'main';

export const HF_SEARCH_LIMIT = 20;

export const HF_SEARCH_DEBOUNCE_MS = 400;

/** the Hub tags a repo holding GGUF weights with this library filter */
export const HF_GGUF_FILTER = 'gguf';

export const HF_SORT_DOWNLOADS = 'downloads';

/** newest and most downloaded first */
export const HF_SORT_DESCENDING = '-1';

export const HF_FILE_TYPE = 'file';

export const GGUF_EXTENSION = '.gguf';

/**
 * Filenames carrying one of these markers are companions of a model, not a
 * model: projectors, importance matrices and speculative decoding heads.
 * Mirrors gguf_filename_is_model() in common/download.cpp.
 */
export const GGUF_COMPANION_MARKERS = [
	'mmproj',
	'imatrix',
	'mtp-',
	'eagle3-',
	'dflash-',
	'dspark-'
];

export const GGUF_MMPROJ_MARKER = 'mmproj';

/** trailing shard marker of a multi part GGUF, e.g. -00001-of-00005 */
export const GGUF_SPLIT_PATTERN = /-(\d{5})-of-(\d{5})$/;

/**
 * Trailing quantization tag of a GGUF filename, e.g. Q8_0 in Qwen3-0.6B-Q8_0.
 * Mirrors the re_tag pattern of get_gguf_split_info() in common/download.cpp:
 * the server derives the name of a cached model from it, so it is the only
 * selector that survives a download.
 */
export const GGUF_QUANT_TAG_PATTERN = /[-.]([A-Za-z0-9_]+)$/;

export const GGUF_FIRST_SPLIT_INDEX = 1;

/** separates a repo id from the quant selector the server resolves against */
export const HF_REPO_TAG_SEPARATOR = ':';
