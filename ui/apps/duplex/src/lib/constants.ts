/** The rate of the stream both ways, PCM16 mono: Hibiki runs Mimi at 24 kHz. */
export const SAMPLE_RATE = 24000;

/** What the worklet ships at a time: 20 ms. */
export const FRAME_SAMPLES = 480;

/** Frames per input_audio_buffer.append: 80 ms, one frame of the model. */
export const FRAMES_PER_APPEND = 4;

/** Audio the speaker holds back after every underrun, so network jitter does not cut the voice. */
export const PREBUFFER_MS = 160;

/** Points of the level history the meters draw, one per worklet frame. */
export const LEVELS_LENGTH = 300;

/** Range of the meters, from full scale down. */
export const LEVELS_RANGE_DB = 60;

export const REALTIME_PATH = 'v1/realtime';

export const PCM16_MAX = 32767;
export const PCM16_SCALE = 32768;
export const WAV_HEADER_BYTES = 44;
export const WAV_CHANNELS = 2;
export const WAV_MIME = 'audio/wav';

/** The events this client sends and reads; see the full duplex section of the server README. */
export const EVENT = {
	APPEND: 'input_audio_buffer.append',
	AUDIO_DELTA: 'response.output_audio.delta',
	CANCEL: 'response.cancel',
	ERROR: 'error',
	RESPONSE_CREATED: 'response.created',
	RESPONSE_DONE: 'response.done',
	SESSION_CREATED: 'session.created',
	SESSION_UPDATE: 'session.update',
	TEXT_DELTA: 'response.output_text.delta'
} as const;

export const AUDIO_FORMAT = 'audio/pcm';

export const WORKLET_NAME = 'llama-duplex';
export const WORKLET_MIME = 'application/javascript';
