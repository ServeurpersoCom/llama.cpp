# Llama Duplex

Live speech translation with a full duplex model, such as [Hibiki](../../../docs/multimodal/tts.md#hibiki) (French
to English), on llama-server's realtime socket: the microphone streams in, the translation plays as it comes back, a
few seconds behind, and its text shows as it is spoken.

## Run

From the root of llama.cpp:

```sh
npm --prefix ui/apps/duplex ci
npm --prefix ui/apps/duplex run build

llama-server -hf ggml-org/hibiki-1b-GGUF:Q8_0 -c 1500 -np 1 --path ui/apps/duplex/dist

# or with local files
llama-server -m hibiki-1b-Q8_0.gguf --mmproj mmproj-hibiki-1b-BF16.gguf -c 1500 -np 1 --path ui/apps/duplex/dist
```

The page uses the one model of the server: it does not work behind a router yet. Open llama-server's address. The microphone needs a secure context: https, or localhost.

To work on it, `npm run dev` serves the app with Vite and forwards `/v1` to `LLAMA_SERVER_ORIGIN`
(`http://localhost:8080` by default).

## How it works

- One AudioWorklet carries the audio both ways, at the rate of the device, so the browser echo canceller hears the
  playback in the same graph as the microphone and the translation can come out of the speakers. It converts to and
  from 24 kHz itself, and holds 160 ms back after an underrun so network jitter does not cut the voice.
- Every 80 ms of microphone goes out as one `input_audio_buffer.append`. The first one opens a response on the
  server; `response.output_audio.delta` and `response.output_text.delta` come back until `response.cancel` closes it
  (**New segment**), and the next audio opens another one.
- **Download WAV** saves the session as a stereo WAV, the microphone left and what played right, in step.

The page draws with the tokens and fonts of Llama UI (`ui/lib/ui`), compiled from its sources.
