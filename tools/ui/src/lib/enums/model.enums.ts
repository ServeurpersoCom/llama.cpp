export enum ModelModality {
	AUDIO = 'AUDIO',
	TEXT = 'TEXT',
	VIDEO = 'VIDEO',
	VISION = 'VISION'
}

/** Which phase a model is in while it comes up, each with its own progress shape. */
export enum ModelProgressKind {
	DOWNLOAD = 'download',
	LOAD = 'load'
}
