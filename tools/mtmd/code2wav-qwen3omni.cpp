// Qwen3-Omni code2wav CLI, thin harness over the code2wav module.
// Reads raw int32 talker codes, writes a 24kHz mono wav, and self checks that
// the streaming path matches the one shot decode byte for byte.

#include "code2wav.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// reads raw int32 codes, the file holds num_quantizers values per frame, frame major.
static std::vector<int32_t> read_codes(const char * path, int Q, int & T) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "code2wav: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<int32_t> codes(bytes / sizeof(int32_t));
    if (fread(codes.data(), 1, bytes, f) != (size_t) bytes) { exit(1); }
    fclose(f);
    T = (int) (codes.size() / Q);
    return codes;
}

// writes a mono 16 bit PCM wav.
static void write_wav(const char * path, const std::vector<float> & pcm, uint32_t sample_rate) {
    const uint32_t n = (uint32_t) pcm.size();
    const uint32_t data_bytes = n * 2;
    FILE * f = fopen(path, "wb");
    auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16(1);
    u32(sample_rate); u32(sample_rate * 2); u16(2); u16(16);
    fwrite("data", 1, 4, f); u32(data_bytes);
    for (float s : pcm) {
        int v = (int) lrintf(s * 32767.0f);
        if (v >  32767) v =  32767;
        if (v < -32768) v = -32768;
        u16((uint16_t) (int16_t) v);
    }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s <code2wav.gguf> <codes.i32> <out.wav> [chunk_size] [left_context]\n", argv[0]);
        return 1;
    }

    code2wav_params params;
    if (argc > 4) params.chunk_size   = atoi(argv[4]);
    if (argc > 5) params.left_context = atoi(argv[5]);

    code2wav_context * ctx = code2wav_init(argv[1], params);
    if (!ctx) return 2;

    const int Q = code2wav_n_quantizers(ctx);
    int T = 0;
    std::vector<int32_t> codes = read_codes(argv[2], Q, T);
    fprintf(stderr, "code2wav: %d frames, chunk %d, left_context %d\n", T, params.chunk_size, params.left_context);

    // one shot decode, written to the output wav.
    std::vector<float> pcm = code2wav_decode(ctx, codes.data(), T);
    write_wav(argv[3], pcm, code2wav_sample_rate(ctx));
    fprintf(stderr, "code2wav: wrote %zu samples to %s\n", pcm.size(), argv[3]);

    // streaming self check, feed the frames in small batches like the LM would.
    code2wav_stream_reset(ctx);
    std::vector<float> spcm;
    const int batch = 7;
    for (int i = 0; i < T; i += batch) {
        const int n = i + batch < T ? batch : T - i;
        code2wav_stream_push(ctx, codes.data() + (size_t) i * Q, n, spcm);
    }
    code2wav_stream_flush(ctx, spcm);

    // the stream must reproduce the one shot exactly.
    double maxd = 0.0;
    const size_t n = pcm.size() < spcm.size() ? pcm.size() : spcm.size();
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double) pcm[i] - (double) spcm[i]);
        if (d > maxd) maxd = d;
    }
    const bool same = spcm.size() == pcm.size() && maxd == 0.0;
    fprintf(stderr, "code2wav: stream %zu vs oneshot %zu, maxdiff %.3e -> %s\n",
            spcm.size(), pcm.size(), maxd, same ? "IDENTICAL" : "MISMATCH");

    code2wav_free(ctx);
    return same ? 0 : 4;
}
