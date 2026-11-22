import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';
import { defineConfig, searchForWorkspaceRoot } from 'vite';

// the tokens and fonts of Llama UI, compiled from its sources
const LLAMA_UI_STYLES = fileURLToPath(new URL('../../lib/ui/src/lib', import.meta.url));
const SERVER_ORIGIN = process.env.LLAMA_SERVER_ORIGIN ?? 'http://localhost:8080';

export default defineConfig({
	// relative, so the page works under any prefix a reverse proxy puts in front of llama-server
	base: './',
	plugins: [svelte()],
	server: {
		fs: { allow: [searchForWorkspaceRoot(process.cwd()), LLAMA_UI_STYLES] },
		proxy: { '/v1': { target: SERVER_ORIGIN, ws: true } }
	}
});
