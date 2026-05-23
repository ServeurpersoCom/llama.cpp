// Qwen3-Omni talker module test, validation only.
// Loads the bisect reference conditioning, runs talker_generate with the kv cache,
// and compares the produced codes against the HF reference ar_codes frame by frame.
// frame 0 must be byte exact, later frames track until a bf16 tie flip diverges.

#include "talker.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & path, size_t n) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::vector<float> v(n);
    if (fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    fclose(f);
    return v;
}

static std::vector<int> read_shape(const std::string & path) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::vector<int> d; int x;
    while (fscanf(f, "%d", &x) == 1) d.push_back(x);
    fclose(f);
    return d;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <talker.gguf> <talker_ref_dir>\n", argv[0]);
        return 1;
    }
    const std::string gguf = argv[1];
    const std::string ref  = argv[2];

    talker_params p;
    p.max_frames = 5;
    talker_context * ctx = talker_init(gguf.c_str(), p);
    if (!ctx) { fprintf(stderr, "talker_init failed\n"); return 2; }
    const int n_embd = talker_n_embd(ctx);
    const int ncb    = talker_n_codebooks(ctx);

    // prefill embed [Tp, n_embd], trailing [Ttr, n_embd], tts pad [n_embd], pos [3, Tp]
    std::vector<int> cs = read_shape(ref + "/ar_cond_00.shape");
    int Tp = cs[0];
    std::vector<float> prefill = read_bin(ref + "/ar_cond_00.bin", (size_t) Tp * n_embd);
    std::vector<int> ts = read_shape(ref + "/ar_trailing.shape");
    int Ttr = ts[0];
    std::vector<float> trailing = read_bin(ref + "/ar_trailing.bin", (size_t) Ttr * n_embd);
    std::vector<float> ttspad = read_bin(ref + "/ar_tts_pad.bin", n_embd);
    std::vector<int> ps = read_shape(ref + "/ar_pos_00.shape");
    std::vector<float> posf = read_bin(ref + "/ar_pos_00.bin", (size_t) ps[0] * ps[1]);
    std::vector<int32_t> posi(posf.size());
    for (size_t i = 0; i < posf.size(); i++) posi[i] = (int32_t) lroundf(posf[i]);

    talker_cond cond;
    cond.input_embed   = prefill.data();
    cond.n_prefill     = Tp;
    cond.trailing_text = trailing.data();
    cond.n_trail       = Ttr;
    cond.tts_pad       = ttspad.data();
    cond.prefill_pos   = posi.data();

    std::vector<int32_t> codes;
    int nframes = talker_generate(ctx, cond, codes);
    fprintf(stderr, "talker_generate produced %d frames\n", nframes);
    if (nframes > 0) {
        fprintf(stderr, "frame 0 produced:");
        for (int k = 0; k < ncb; k++) fprintf(stderr, " %d", codes[k]);
        fprintf(stderr, "\n");
    }

    // compare against the reference codes
    std::vector<int> ks = read_shape(ref + "/ar_codes.shape");
    int K = ks[0], W = ks[1];
    std::vector<float> arc = read_bin(ref + "/ar_codes.bin", (size_t) K * W);

    int total_ok = 0, total = 0;
    for (int n = 0; n < nframes && n < K; n++) {
        int match = 0;
        for (int k = 0; k < ncb; k++) {
            int mine = codes[(size_t) n * ncb + k];
            int ref_c = (int) lroundf(arc[(size_t) n * W + k]);
            if (mine == ref_c) match++;
        }
        total_ok += match; total += ncb;
        fprintf(stderr, "frame %d: %d/%d %s (cb0 mine=%d ref=%d)\n", n, match, ncb,
                match == ncb ? "OK" : "diff",
                codes[(size_t) n * ncb], (int) lroundf(arc[(size_t) n * W]));
    }
    fprintf(stderr, "total %d/%d codes match HF\n", total_ok, total);

    talker_free(ctx);
    return 0;
}
