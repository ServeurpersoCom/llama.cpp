/**
 * Shared progress helpers. getETASecs serves prompt prefill, the model load
 * helpers serve the /models/sse load surfaces (selector row and chat message).
 */

/**
 * Estimate remaining seconds from a fractional progress ratio.
 * Returns undefined while the sample is too early to be meaningful
 * (zero progress, or less than half a second elapsed).
 *
 * @param done - work completed (same unit as total)
 * @param total - total work
 * @param elapsedMs - elapsed time since progress started, in ms
 */
export function getETASecs(done: number, total: number, elapsedMs: number): number | undefined {
	const elapsedSecs = elapsedMs / 1000;

	if (done === 0 || elapsedSecs < 0.5) {
		return undefined;
	}

	return elapsedSecs * (total / done - 1);
}

const MODEL_LOAD_STAGE_LABELS: Record<ApiModelLoadStage, string> = {
	fit_params: 'Fitting params',
	text_model: 'Loading weights',
	spec_model: 'Loading draft',
	mmproj_model: 'Loading projector'
};

/**
 * Human label for a model load stage, with a generic fallback before the
 * feed reports its first stage.
 */
export function modelLoadStageLabel(stage?: ApiModelLoadStage): string {
	return stage ? MODEL_LOAD_STAGE_LABELS[stage] : 'Loading';
}

/**
 * Single line describing load progress: stage label with a trailing ellipsis
 * while indeterminate, or stage label with a percentage once a value arrives.
 * Returns null when there is no progress to show.
 */
export function modelLoadProgressText(progress: ModelLoadProgress | null): string | null {
	if (!progress) return null;

	const label = modelLoadStageLabel(progress.stage);
	if (progress.value === undefined) return `${label}...`;

	return `${label} ${Math.round(progress.value * 100)}%`;
}
