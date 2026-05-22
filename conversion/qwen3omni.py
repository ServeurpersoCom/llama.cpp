from __future__ import annotations

import re
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import TextModel, gguf, logger

# transposed convs (upsample stage and decoder block.1), recognised by name.
# forward convs need no permute, PyTorch [OC, IC, K] already maps to ggml [K, IC, OC].
_TRANSCONV = re.compile(r"^(upsample\.\d+\.0|decoder\.\d+\.block\.1)\.conv\.weight$")
_SNAKE_EPS = 1e-9


# Converts the code2wav vocoder of a Qwen3-Omni checkpoint to its own GGUF.
# The talker and thinker are dropped here, they convert through their own path.
@TextModel.register("Qwen3OmniMoeForConditionalGeneration")
class Qwen3OmniCode2WavModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3OMNI_CODE2WAV

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item

        # keep only the vocoder weights.
        if not name.startswith("code2wav."):
            return None
        return item

    def set_vocab(self):
        self._set_vocab_none()

    def set_gguf_parameters(self):
        cfg = self.hparams["code2wav_config"]
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        gw = self.gguf_writer

        gw.add_uint32 (arch + ".hidden_size",        cfg["hidden_size"])
        gw.add_uint32 (arch + ".block_count",        cfg["num_hidden_layers"])
        gw.add_uint32 (arch + ".head_count",         cfg["num_attention_heads"])
        gw.add_uint32 (arch + ".head_count_kv",      cfg["num_key_value_heads"])
        gw.add_uint32 (arch + ".feed_forward_length", cfg["intermediate_size"])
        gw.add_float32(arch + ".rms_eps",            cfg["rms_norm_eps"])
        gw.add_float32(arch + ".rope_theta",         cfg["rope_theta"])
        gw.add_uint32 (arch + ".sliding_window",     cfg["sliding_window"])
        gw.add_uint32 (arch + ".decoder_dim",        cfg["decoder_dim"])
        gw.add_uint32 (arch + ".num_quantizers",     cfg["num_quantizers"])
        gw.add_uint32 (arch + ".codebook_size",      cfg["codebook_size"])
        gw.add_array  (arch + ".upsample_rates",     list(cfg["upsample_rates"]))
        gw.add_array  (arch + ".upsampling_ratios",  list(cfg["upsampling_ratios"]))
        gw.add_float32(arch + ".layer_scale_init",   cfg["layer_scale_initial_scale"])

        total_upsample = 1
        for r in list(cfg["upsample_rates"]) + list(cfg["upsampling_ratios"]):
            total_upsample *= r
        gw.add_uint32(arch + ".total_upsample", total_upsample)
        gw.add_uint32(arch + ".sample_rate",    24000)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        import torch

        base = name[len("code2wav."):]

        # SnakeBeta stores alpha and beta in log space, bake the pair the ggml op wants:
        # a = exp(alpha), inv_b = 1 / (exp(beta) + eps).
        if base.endswith(".alpha"):
            return [(base[: -len(".alpha")] + ".snake_a", torch.exp(data_torch.float()))]
        if base.endswith(".beta"):
            return [(base[: -len(".beta")] + ".snake_inv_b", 1.0 / (torch.exp(data_torch.float()) + _SNAKE_EPS))]

        # ConvTranspose1d [IC, OC, K] -> columns [OC*K, IC] in oc*K+k order,
        # matching the col2im_1d kernel index col[oc*K + k].
        if _TRANSCONV.match(base):
            ic, oc, k = data_torch.shape
            return [(base, data_torch.permute(1, 2, 0).reshape(oc * k, ic).contiguous())]

        return [(base, data_torch)]
