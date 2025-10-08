(function () {
    "use strict";

    const SUFFIX_ORDER = new Map([
        ["attn_norm", 10],
        ["attn_norm_2", 11],
        ["attn_q_norm", 12],
        ["attn_k_norm", 13],
        ["attn_output_norm", 14],
        ["post_attention_norm", 18],
        ["attn_post_norm", 19],
        ["ffn_norm", 30],
        ["ffn_norm_exps", 31],
        ["layer_output_norm", 32],
        ["post_ffw_norm", 33],
        ["ffn_pre_norm", 34],
        ["ffn_post_norm", 35],
        ["attn_q", 50],
        ["attn_qkv", 51],
        ["attn_q_a", 52],
        ["attn_q_b", 53],
        ["attn_k", 54],
        ["attn_v", 55],
        ["attn_output", 56],
        ["attn_rot_embd", 57],
        ["attn_sinks", 58],
        ["attn_q_bias", 59],
        ["attn_k_bias", 60],
        ["attn_v_bias", 61],
        ["attn_output_bias", 62],
        ["ffn_gate", 100],
        ["ffn_up", 101],
        ["ffn_down", 102],
        ["ffn_gate_inp", 103],
        ["ffn_gate_exps", 104],
        ["ffn_up_exps", 105],
        ["ffn_down_exps", 106],
        ["ffn_gate_shexp", 107],
        ["ffn_gate_inp_shexp", 108],
        ["ffn_gate_chexps", 109],
        ["ffn_gate_inp_chexps", 110],
        ["ffn_gate_bias", 111],
        ["ffn_up_bias", 112],
        ["ffn_down_bias", 113],
        ["ffn_gate_exps_bias", 114],
        ["ffn_up_exps_bias", 115],
        ["ffn_down_exps_bias", 116],
        ["router", 150],
        ["router_norm", 151],
        ["exp_probs_b", 152],
        ["inp_gate", 153],
        ["inp_gate_norm", 154],
        ["altup_router", 160],
        ["altup_router_norm", 161],
        ["altup_proj", 162],
        ["altup_unembd_proj", 163],
        ["altup_predict_coef", 164],
        ["altup_correct_coef", 165],
        ["altup_correct_scale", 166],
        ["altup_post_norm", 167],
        ["laurel_l", 170],
        ["laurel_r", 171],
        ["laurel_post_norm", 172],
    ]);

    const ATTENTION_NORM_BASES = new Set([
        "attn_norm",
        "attn_norm_2",
        "attn_q_norm",
        "attn_q_a_norm",
        "attn_k_norm",
        "attn_kv_a_norm",
        "attn_output_norm",
        "attn_q_b_norm",
        "attn_gate_norm",
        "attn_pre_norm",
        "attn_post_norm",
        "post_attention_norm",
    ]);

    const FFN_NORM_BASES = new Set([
        "ffn_norm",
        "ffn_norm_exps",
        "post_ffw_norm",
        "layer_output_norm",
        "ffn_pre_norm",
        "ffn_post_norm",
    ]);

    const ROUTER_KEYWORDS = [
        "router",
        "exp_probs",
        "expert",
        "moe",
        "gating",
    ];

    function buildMetadataIndex(entries) {
        const map = new Map();
        const keys = [];
        if (Array.isArray(entries)) {
            for (const entry of entries) {
                if (!entry || typeof entry.key !== "string") {
                    continue;
                }
                map.set(entry.key, entry);
                keys.push(entry.key);
            }
        }
        let architectureName = null;
        const archEntry = map.get("general.architecture");
        if (archEntry && typeof archEntry.value === "string" && archEntry.value.length > 0) {
            architectureName = archEntry.value.trim();
        }
        return { map, keys, architectureName };
    }

    function getEntryValue(entry) {
        if (!entry) {
            return undefined;
        }
        if (entry.type === "GGUF_TYPE_ARRAY") {
            if (Array.isArray(entry.preview)) {
                return entry.preview;
            }
            return undefined;
        }
        return entry.value;
    }

    function extractNumber(value) {
        if (typeof value === "number" && Number.isFinite(value)) {
            return value;
        }
        if (typeof value === "string" && value.length > 0) {
            const parsed = Number(value);
            if (Number.isFinite(parsed)) {
                return parsed;
            }
        }
        if (Array.isArray(value)) {
            for (const item of value) {
                const candidate = extractNumber(item);
                if (Number.isFinite(candidate)) {
                    return candidate;
                }
            }
        }
        return null;
    }

    function extractBoolean(value) {
        if (typeof value === "boolean") {
            return value;
        }
        if (typeof value === "number") {
            return value !== 0;
        }
        if (typeof value === "string") {
            const normalized = value.trim().toLowerCase();
            if (normalized === "true" || normalized === "yes" || normalized === "on") {
                return true;
            }
            if (normalized === "false" || normalized === "no" || normalized === "off") {
                return false;
            }
        }
        return null;
    }

    function resolveArchValue(index, architectureName, suffix) {
        const { map, keys } = index;
        const tried = new Set();
        const candidates = [];
        if (architectureName) {
            candidates.push(`${architectureName}.${suffix}`);
            const normalized = architectureName.replace(/\s+/g, "");
            if (normalized !== architectureName) {
                candidates.push(`${normalized}.${suffix}`);
            }
            const underscored = architectureName.replace(/[-\.]/g, "_");
            if (underscored !== architectureName) {
                candidates.push(`${underscored}.${suffix}`);
            }
            const dashed = architectureName.replace(/_/g, "-");
            if (dashed !== architectureName) {
                candidates.push(`${dashed}.${suffix}`);
            }
        }
        if (architectureName && !architectureName.startsWith("llama")) {
            candidates.push(`llama.${suffix}`);
        }
        for (const key of keys) {
            if (key.endsWith(`.${suffix}`)) {
                candidates.push(key);
            }
        }
        for (const key of candidates) {
            if (tried.has(key)) {
                continue;
            }
            tried.add(key);
            if (map.has(key)) {
                return getEntryValue(map.get(key));
            }
        }
        return undefined;
    }

    function getArchNumber(index, architectureName, suffix) {
        const value = resolveArchValue(index, architectureName, suffix);
        const number = extractNumber(value);
        return Number.isFinite(number) ? number : null;
    }

    function getArchBoolean(index, architectureName, suffix) {
        const value = resolveArchValue(index, architectureName, suffix);
        const boolean = extractBoolean(value);
        return boolean !== null ? boolean : null;
    }

    function detectOutputWeightSharing(index, architectureName) {
        const suffixes = [
            "tie_embeddings",
            "tie_weights",
            "tie_word_embeddings",
            "share_embeddings",
            "share_token_embeddings",
        ];
        for (const suffix of suffixes) {
            const value = getArchBoolean(index, architectureName, suffix);
            if (value !== null) {
                return value;
            }
        }
        const directKeys = [
            "general.tie_embeddings",
            "general.tie_word_embeddings",
            "llama.tie_embeddings",
            "llama.tie_weights",
            "llama.tie_word_embeddings",
            "granite.tie_embeddings",
        ];
        for (const key of directKeys) {
            if (index.map.has(key)) {
                const boolean = extractBoolean(getEntryValue(index.map.get(key)));
                if (boolean !== null) {
                    return boolean;
                }
            }
        }
        return false;
    }

    function getArchArray(index, architectureName, suffix) {
        const value = resolveArchValue(index, architectureName, suffix);
        if (Array.isArray(value)) {
            return value.slice();
        }
        if (typeof value === "number" || typeof value === "string") {
            const number = extractNumber(value);
            return Number.isFinite(number) ? [number] : [];
        }
        return [];
    }

    function getTensorMap(tensors) {
        const map = new Map();
        if (Array.isArray(tensors)) {
            for (const tensor of tensors) {
                if (tensor && typeof tensor.name === "string") {
                    map.set(tensor.name, tensor);
                }
            }
        }
        return map;
    }

    function getTensorDim(tensorsByName, name, axis) {
        const tensor = tensorsByName.get(name);
        if (!tensor || !Array.isArray(tensor.shape)) {
            return null;
        }
        if (axis < 0 || axis >= tensor.shape.length) {
            return null;
        }
        const value = tensor.shape[axis];
        return Number.isFinite(value) ? value : null;
    }

    function getBaseName(name) {
        if (!name) {
            return "";
        }
        let base = name;
        const lastDot = base.lastIndexOf(".");
        if (lastDot > 0) {
            base = base.slice(0, lastDot);
        }
        return base;
    }

    function getSuffixOrder(base) {
        if (SUFFIX_ORDER.has(base)) {
            return SUFFIX_ORDER.get(base);
        }
        if (base.startsWith("attn_")) {
            return 60;
        }
        if (base.startsWith("ffn_")) {
            return 120;
        }
        return 1000;
    }

    function classifyBlockBase(base) {
        if (!base) {
            return "other";
        }
        if (ATTENTION_NORM_BASES.has(base) || (base.endsWith("_norm") && base.includes("attn"))) {
            return "pre-attn-norm";
        }
        if (base.startsWith("attn_") || base.includes("attn_q") || base.includes("attn_k") || base.includes("attn_v")) {
            return "attention";
        }
        if (FFN_NORM_BASES.has(base) || (base.endsWith("_norm") && base.includes("ffn"))) {
            return "pre-ffn-norm";
        }
        if (base.startsWith("ffn_") && !FFN_NORM_BASES.has(base)) {
            return "ffn";
        }
        for (const keyword of ROUTER_KEYWORDS) {
            if (base.includes(keyword)) {
                return "routing";
            }
        }
        if (base.includes("ssm") || base.startsWith("time_mix") || base.startsWith("conv") || base.includes("mamba")) {
            return "sequence";
        }
        return "other";
    }

    function sortBlockEntries(entries) {
        entries.sort((a, b) => {
            const orderA = getSuffixOrder(a.base);
            const orderB = getSuffixOrder(b.base);
            if (orderA !== orderB) {
                return orderA - orderB;
            }
            return a.suffix.localeCompare(b.suffix);
        });
    }

    function buildBlocks(tensors) {
        const map = new Map();
        const indices = new Set();
        if (Array.isArray(tensors)) {
            for (const tensor of tensors) {
                if (!tensor || typeof tensor.name !== "string") {
                    continue;
                }
                const match = tensor.name.match(/^blk\.(\d+)\.(.+)$/);
                if (!match) {
                    continue;
                }
                const index = Number.parseInt(match[1], 10);
                if (!Number.isFinite(index)) {
                    continue;
                }
                const suffix = match[2];
                const base = getBaseName(suffix);
                indices.add(index);
                if (!map.has(index)) {
                    map.set(index, []);
                }
                map.get(index).push({ tensor, suffix, base });
            }
        }
        for (const [index, entries] of map.entries()) {
            sortBlockEntries(entries);
        }
        const ordered = Array.from(indices);
        ordered.sort((a, b) => a - b);
        return { map, indices: ordered };
    }

    function makeExpected(dimDefinitions, shape) {
        const dims = dimDefinitions.map((def, idx) => {
            const result = { label: def.label, value: null };
            if (Number.isFinite(def.value)) {
                result.value = def.value;
            } else if (Array.isArray(shape) && Number.isFinite(shape[idx])) {
                result.value = shape[idx];
            }
            return result;
        });
        const numbers = dims.map((dim) => (Number.isFinite(dim.value) ? dim.value : null));
        return { dims, numbers };
    }

    function shapesMatch(shape, expected) {
        if (!Array.isArray(shape) || !Array.isArray(expected)) {
            return true;
        }
        const length = Math.min(shape.length, expected.length);
        for (let i = 0; i < length; ++i) {
            const exp = expected[i];
            if (Number.isFinite(exp) && shape[i] !== exp) {
                return false;
            }
        }
        return true;
    }

    function formatShape(shape) {
        if (!Array.isArray(shape) || shape.length === 0) {
            return "[]";
        }
        return `[${shape.map((value) => (Number.isFinite(value) ? value : "?" )).join(", ")}]`;
    }

    function formatExpected(dims) {
        if (!Array.isArray(dims) || dims.length === 0) {
            return null;
        }
        const parts = dims.map((dim) => {
            if (Number.isFinite(dim.value)) {
                return `${dim.label}=${dim.value}`;
            }
            return dim.label;
        });
        return `[${parts.join(", ")}]`;
    }

    function createList() {
        const list = document.createElement("ul");
        list.className = "architecture-list";
        return list;
    }

    function createGroup(title) {
        const item = document.createElement("li");
        item.className = "architecture-group";
        const header = document.createElement("div");
        header.className = "architecture-group__title";
        header.textContent = title;
        item.appendChild(header);
        return { item, header };
    }

    function createBadge(text, className = "") {
        const span = document.createElement("span");
        span.className = className ? `architecture-badge ${className}` : "architecture-badge";
        span.textContent = text;
        return span;
    }

    function createNoteElement(text) {
        const note = document.createElement("div");
        note.className = "architecture-note";
        note.textContent = text;
        return note;
    }

    function describeDimension(ctx, dimension) {
        if (!Number.isFinite(dimension)) {
            return { label: "dimension", value: dimension };
        }
        const checks = [
            { label: "hidden_dim", value: ctx.embeddingLength },
            { label: "head_dim", value: ctx.keyLength },
            { label: "value_dim", value: ctx.valueLength },
            { label: "ff_dim", value: ctx.feedForwardLength },
            { label: "ff_dim_expert", value: ctx.expertFeedForwardLength },
            { label: "shared_ff_dim", value: ctx.expertSharedFeedForwardLength },
            { label: "chunk_ff_dim", value: ctx.expertChunkFeedForwardLength },
            { label: "n_heads", value: ctx.headCount },
            { label: "n_kv_heads", value: ctx.kvHeadCount },
            { label: "num_experts", value: ctx.expertCount },
            { label: "experts_used", value: ctx.expertUsedCount },
        ];
        for (const entry of checks) {
            if (Number.isFinite(entry.value) && dimension === entry.value) {
                return { label: entry.label, value: entry.value };
            }
        }
        if (Number.isFinite(ctx.headCount) && Number.isFinite(ctx.keyLength)) {
            const combined = ctx.headCount * ctx.keyLength;
            if (combined === dimension) {
                return { label: "n_heads x head_dim", value: combined };
            }
        }
        if (Number.isFinite(ctx.kvHeadCount) && Number.isFinite(ctx.keyLength)) {
            const combinedKv = ctx.kvHeadCount * ctx.keyLength;
            if (combinedKv === dimension) {
                return { label: "n_kv_heads x head_dim", value: combinedKv };
            }
        }
        if (Number.isFinite(ctx.kvHeadCount) && Number.isFinite(ctx.valueLength)) {
            const combinedValue = ctx.kvHeadCount * ctx.valueLength;
            if (combinedValue === dimension) {
                return { label: "n_kv_heads x value_dim", value: combinedValue };
            }
        }
        return { label: "dimension", value: dimension };
    }

    function makeExpectedForNorm(ctx, tensor) {
        const shape = Array.isArray(tensor?.shape) ? tensor.shape : [];
        if (shape.length === 1 && Number.isFinite(shape[0])) {
            const described = describeDimension(ctx, shape[0]);
            return makeExpected([described], shape);
        }
        return makeExpected([{ label: "hidden_dim", value: ctx.embeddingLength }], shape);
    }

    function createTensorNode(ctx, tensor, options = {}) {
        const li = document.createElement("li");
        li.className = "architecture-tensor";
        if (!tensor) {
            const variant = options.variant || "missing";
            if (variant === "info") {
                li.classList.add("architecture-tensor--info");
            } else {
                li.classList.add("architecture-tensor--missing");
            }
            const header = document.createElement("div");
            header.className = "architecture-tensor__header";
            const code = document.createElement("code");
            code.textContent = options.displayName || options.name || "(missing tensor)";
            header.appendChild(code);
            const badgeLabel = options.badgeLabel || (variant === "info" ? "note" : "missing");
            const badgeClass = variant === "info" ? "architecture-badge--info" : "architecture-badge--mismatch";
            header.appendChild(createBadge(badgeLabel, badgeClass));
            li.appendChild(header);
            if (options.note) {
                const note = document.createElement("div");
                note.className = "architecture-tensor__note";
                note.textContent = options.note;
                li.appendChild(note);
            }
            return li;
        }
        ctx.used.add(tensor.name);
        const header = document.createElement("div");
        header.className = "architecture-tensor__header";
        const code = document.createElement("code");
        code.textContent = options.displayName || tensor.name;
        header.appendChild(code);
        if (tensor.type) {
            header.appendChild(createBadge(tensor.type));
        }
        li.appendChild(header);
        if (Array.isArray(tensor.shape)) {
            const shape = document.createElement("div");
            shape.className = "architecture-tensor__shape";
            shape.textContent = `shape ${formatShape(tensor.shape)}`;
            li.appendChild(shape);
        }
        const expected = options.expected || null;
        if (expected && Array.isArray(expected.dims) && expected.dims.length > 0) {
            const expectedText = formatExpected(expected.dims);
            if (expectedText) {
                const expectedEl = document.createElement("div");
                expectedEl.className = "architecture-tensor__expected";
                expectedEl.textContent = `expected ${expectedText}`;
                li.appendChild(expectedEl);
            }
            if (!shapesMatch(tensor.shape, expected.numbers)) {
                li.classList.add("architecture-tensor--mismatch");
                header.appendChild(createBadge("mismatch", "architecture-badge--mismatch"));
            }
        }
        if (options.note) {
            const note = document.createElement("div");
            note.className = "architecture-tensor__note";
            note.textContent = options.note;
            li.appendChild(note);
        }
        if (tensor.name) {
            const actions = document.createElement("div");
            actions.className = "architecture-tensor__actions";

            const encodedName = encodeURIComponent(tensor.name);

            const heatmapButton = document.createElement("button");
            heatmapButton.type = "button";
            heatmapButton.dataset.action = "heatmap";
            heatmapButton.dataset.name = encodedName;
            heatmapButton.textContent = "Heatmap";
            actions.appendChild(heatmapButton);

            const statisticsButton = document.createElement("button");
            statisticsButton.type = "button";
            statisticsButton.dataset.action = "statistics";
            statisticsButton.dataset.name = encodedName;
            statisticsButton.textContent = "Statistics";
            actions.appendChild(statisticsButton);

            header.appendChild(actions);
        }
        return li;
    }

    function getExpectedForTensor(ctx, tensor) {
        if (!tensor || typeof tensor.name !== "string") {
            return null;
        }
        const shape = Array.isArray(tensor.shape) ? tensor.shape : [];
        switch (tensor.name) {
            case "token_embd.weight":
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "vocab_size", value: ctx.vocabSize },
                ], shape);
            case "token_embd_norm.weight":
            case "token_embd_norm.bias":
                return makeExpected([{ label: "hidden_dim", value: ctx.embeddingLength }], shape);
            case "output_norm.weight":
            case "output_norm.bias":
                return makeExpected([{ label: "hidden_dim", value: ctx.embeddingLength }], shape);
            case "output.weight":
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "vocab_size", value: ctx.vocabSize },
                ], shape);
            case "output.bias":
                return makeExpected([{ label: "vocab_size", value: ctx.vocabSize }], shape);
            default:
                break;
        }
        const match = tensor.name.match(/^blk\.(\d+)\.(.+)$/);
        if (!match) {
            return null;
        }
        const suffix = match[2];
        const parts = suffix.split('.');
        let suffixType = "";
        let base = suffix;
        if (parts.length > 1) {
            suffixType = parts[parts.length - 1];
            base = parts.slice(0, -1).join('.');
        }
        switch (base) {
            case "attn_norm":
            case "attn_norm_2":
            case "attn_q_norm":
            case "attn_k_norm":
            case "attn_q_a_norm":
            case "attn_kv_a_norm":
            case "attn_output_norm":
            case "attn_post_norm":
            case "post_attention_norm":
            case "ffn_norm":
            case "ffn_norm_exps":
            case "post_ffw_norm":
            case "layer_output_norm":
            case "ffn_pre_norm":
            case "ffn_post_norm":
                return makeExpectedForNorm(ctx, tensor);
            case "attn_q":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "n_heads x head_dim", value: ctx.headCount && ctx.keyLength ? ctx.headCount * ctx.keyLength : null },
                    ], shape);
                }
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "n_heads x head_dim", value: ctx.headCount && ctx.keyLength ? ctx.headCount * ctx.keyLength : null },
                ], shape);
            case "attn_qkv": {
                let proj = null;
                if (ctx.headCount && ctx.keyLength && ctx.kvHeadCount) {
                    proj = ctx.embeddingLength + 2 * ctx.kvHeadCount * ctx.keyLength;
                }
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "qkv_proj", value: proj },
                ], shape);
            }
            case "attn_k":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "n_kv_heads x head_dim", value: ctx.kvHeadCount && ctx.keyLength ? ctx.kvHeadCount * ctx.keyLength : null },
                    ], shape);
                }
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "n_kv_heads x head_dim", value: ctx.kvHeadCount && ctx.keyLength ? ctx.kvHeadCount * ctx.keyLength : null },
                ], shape);
            case "attn_v":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "n_kv_heads x value_dim", value: ctx.kvHeadCount && ctx.valueLength ? ctx.kvHeadCount * ctx.valueLength : null },
                    ], shape);
                }
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "n_kv_heads x value_dim", value: ctx.kvHeadCount && ctx.valueLength ? ctx.kvHeadCount * ctx.valueLength : null },
                ], shape);
            case "attn_output":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "hidden_dim", value: ctx.embeddingLength },
                    ], shape);
                }
                return makeExpected([
                    { label: "n_heads x value_dim", value: ctx.headCount && ctx.valueLength ? ctx.headCount * ctx.valueLength : null },
                    { label: "hidden_dim", value: ctx.embeddingLength },
                ], shape);
            case "ffn_gate":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "ff_dim", value: ctx.feedForwardLength },
                    ], shape);
                }
            case "ffn_up":
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "ff_dim", value: ctx.feedForwardLength },
                ], shape);
            case "ffn_down":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "hidden_dim", value: ctx.embeddingLength },
                    ], shape);
                }
                return makeExpected([
                    { label: "ff_dim", value: ctx.feedForwardLength },
                    { label: "hidden_dim", value: ctx.embeddingLength },
                ], shape);
            case "ffn_gate_inp":
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "num_experts", value: ctx.expertCount },
                ], shape);
            case "ffn_gate_exps":
            case "ffn_up_exps":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "ff_dim_expert", value: ctx.expertFeedForwardLength },
                        { label: "num_experts", value: ctx.expertCount },
                    ], shape);
                }
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "ff_dim_expert", value: ctx.expertFeedForwardLength },
                    { label: "num_experts", value: ctx.expertCount },
                ], shape);
            case "ffn_down_exps":
                if (suffixType === "bias") {
                    return makeExpected([
                        { label: "hidden_dim", value: ctx.embeddingLength },
                        { label: "num_experts", value: ctx.expertCount },
                    ], shape);
                }
                return makeExpected([
                    { label: "ff_dim_expert", value: ctx.expertFeedForwardLength },
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "num_experts", value: ctx.expertCount },
                ], shape);
            case "ffn_gate_shexp":
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "shared_ff_dim", value: ctx.expertSharedFeedForwardLength },
                ], shape);
            case "ffn_gate_chexps":
                return makeExpected([
                    { label: "hidden_dim", value: ctx.embeddingLength },
                    { label: "chunk_ff_dim", value: ctx.expertChunkFeedForwardLength },
                    { label: "num_chunks", value: null },
                ], shape);
            default:
                return null;
        }
    }

    function renderLlama(ctx, target) {
        const tree = document.createElement("ul");
        tree.className = "architecture-tree";

        const embedding = createGroup("Token embedding");
        const embeddingList = createList();
        embedding.item.appendChild(embeddingList);
        embeddingList.appendChild(createTensorNode(ctx, ctx.tensorsByName.get("token_embd.weight"), { expected: getExpectedForTensor(ctx, ctx.tensorsByName.get("token_embd.weight")) }));
        const tokenNorm = ctx.tensorsByName.get("token_embd_norm.weight");
        if (tokenNorm) {
            embeddingList.appendChild(createTensorNode(ctx, tokenNorm, { expected: getExpectedForTensor(ctx, tokenNorm) }));
        }
        const tokenNormBias = ctx.tensorsByName.get("token_embd_norm.bias");
        if (tokenNormBias) {
            embeddingList.appendChild(createTensorNode(ctx, tokenNormBias, { expected: getExpectedForTensor(ctx, tokenNormBias) }));
        }
        tree.appendChild(embedding.item);

        const ropeTensors = Array.from(ctx.tensorsByName.values()).filter((tensor) => typeof tensor.name === "string" && tensor.name.startsWith("rope"));
        if (ropeTensors.length > 0) {
            const ropeGroup = createGroup("Rotary positional parameters");
            const ropeList = createList();
            ropeGroup.item.appendChild(ropeList);
            ropeTensors.sort((a, b) => a.name.localeCompare(b.name));
            for (const tensor of ropeTensors) {
                ropeList.appendChild(createTensorNode(ctx, tensor));
            }
            tree.appendChild(ropeGroup.item);
        }

        const blocksGroup = createGroup(`Transformer blocks x ${ctx.blockOrder.length}`);
        if (ctx.blockOrder.length > 0) {
            const blocksList = createList();
            blocksGroup.item.appendChild(blocksList);

            const navListItem = document.createElement("li");
            const navContainer = document.createElement("div");
            navContainer.className = "architecture-block-nav";
            navListItem.appendChild(navContainer);

            const navSummary = document.createElement("div");
            navSummary.className = "architecture-block-nav__summary";
            navContainer.appendChild(navSummary);

            const navControls = document.createElement("div");
            navControls.className = "architecture-block-nav__controls";
            navContainer.appendChild(navControls);

            const selectLabel = document.createElement("label");
            selectLabel.className = "architecture-block-nav__label";
            selectLabel.textContent = "Block";
            const select = document.createElement("select");
            select.className = "architecture-block-nav__select";
            for (const blockIndex of ctx.blockOrder) {
                const option = document.createElement("option");
                option.value = String(blockIndex);
                option.textContent = String(blockIndex);
                select.appendChild(option);
            }
            selectLabel.appendChild(select);
            navControls.appendChild(selectLabel);

            const prevButton = document.createElement("button");
            prevButton.type = "button";
            prevButton.textContent = "Prev";
            navControls.appendChild(prevButton);

            const nextButton = document.createElement("button");
            nextButton.type = "button";
            nextButton.textContent = "Next";
            navControls.appendChild(nextButton);

            blocksList.appendChild(navListItem);

            const detailListItem = document.createElement("li");
            const blockDetailContainer = document.createElement("div");
            blockDetailContainer.className = "architecture-block";
            detailListItem.appendChild(blockDetailContainer);
            blocksList.appendChild(detailListItem);

            const blockIndices = ctx.blockOrder.slice();
            for (const blockIndex of blockIndices) {
                const entries = ctx.blocks.get(blockIndex) || [];
                for (const entry of entries) {
                    if (entry && entry.tensor && typeof entry.tensor.name === "string") {
                        ctx.used.add(entry.tensor.name);
                    }
                }
            }
            let currentPosition = 0;

            const categoryLabels = {
                "pre-attn-norm": "Attention input normalization",
                "attention": "Multi-head / grouped attention",
                "pre-ffn-norm": "Pre-FFN normalization",
                "ffn": "Feed-forward network",
                "routing": "Routing & experts",
                "sequence": "Sequence mixers / auxiliary modules",
                "other": "Other tensors",
            };

            function renderBlockDetailAt(position) {
                if (position < 0 || position >= blockIndices.length) {
                    return;
                }
                currentPosition = position;
                const blockIndex = blockIndices[position];
                select.value = String(blockIndex);
                prevButton.disabled = position === 0;
                nextButton.disabled = position === blockIndices.length - 1;

                blockDetailContainer.replaceChildren();

                const title = document.createElement("div");
                title.className = "architecture-block__title";
                title.textContent = `Block ${blockIndex}`;
                blockDetailContainer.appendChild(title);

                const sectionList = createList();
                blockDetailContainer.appendChild(sectionList);

                const entries = ctx.blocks.get(blockIndex) || [];
                const categories = new Map([
                    ["pre-attn-norm", []],
                    ["attention", []],
                    ["pre-ffn-norm", []],
                    ["ffn", []],
                    ["routing", []],
                    ["sequence", []],
                    ["other", []],
                ]);

                for (const entry of entries) {
                    const tensor = entry.tensor;
                    const categoryKey = categories.has(classifyBlockBase(entry.base))
                        ? classifyBlockBase(entry.base)
                        : "other";
                    categories.get(categoryKey).push(entry);
                }

                for (const [key, entriesForKey] of categories.entries()) {
                    if (!entriesForKey.length) {
                        continue;
                    }
                    const subgroup = document.createElement("li");
                    subgroup.className = "architecture-subgroup";
                    const subgroupTitle = document.createElement("div");
                    subgroupTitle.className = "architecture-subgroup__title";
                    subgroupTitle.textContent = categoryLabels[key];
                    subgroup.appendChild(subgroupTitle);
                    const tensorsList = createList();
                    subgroup.appendChild(tensorsList);
                    for (const entry of entriesForKey) {
                        const tensor = entry.tensor;
                        tensorsList.appendChild(
                            createTensorNode(ctx, tensor, {
                                expected: getExpectedForTensor(ctx, tensor),
                            }),
                        );
                    }
                    sectionList.appendChild(subgroup);
                }

                const residualNote = createNoteElement("Residual connections reuse the block input (no dedicated tensor).");
                blockDetailContainer.appendChild(residualNote);
            }

            prevButton.addEventListener("click", () => {
                if (currentPosition > 0) {
                    renderBlockDetailAt(currentPosition - 1);
                }
            });

            nextButton.addEventListener("click", () => {
                if (currentPosition < blockIndices.length - 1) {
                    renderBlockDetailAt(currentPosition + 1);
                }
            });

            select.addEventListener("change", () => {
                const value = Number(select.value);
                const newPosition = blockIndices.indexOf(value);
                if (newPosition >= 0) {
                    renderBlockDetailAt(newPosition);
                }
            });

            renderBlockDetailAt(0);
        } else {
            blocksGroup.item.appendChild(
                createNoteElement("No transformer blocks exported in GGUF payload."),
            );
        }
        tree.appendChild(blocksGroup.item);

        const finalNorm = ctx.tensorsByName.get("output_norm.weight");
        if (finalNorm || ctx.tensorsByName.get("output_norm.bias")) {
            const normGroup = createGroup("Final RMSNorm");
            const normList = createList();
            normGroup.item.appendChild(normList);
            if (finalNorm) {
                normList.appendChild(createTensorNode(ctx, finalNorm, { expected: getExpectedForTensor(ctx, finalNorm) }));
            }
            const normBias = ctx.tensorsByName.get("output_norm.bias");
            if (normBias) {
                normList.appendChild(createTensorNode(ctx, normBias, { expected: getExpectedForTensor(ctx, normBias) }));
            }
            tree.appendChild(normGroup.item);
        }

        const lmHeadGroup = createGroup("Language modeling head");
        const lmHeadList = createList();
        lmHeadGroup.item.appendChild(lmHeadList);
        const lmWeight = ctx.tensorsByName.get("output.weight");
        if (lmWeight) {
            lmHeadList.appendChild(createTensorNode(ctx, lmWeight, { expected: getExpectedForTensor(ctx, lmWeight) }));
        } else if (ctx.outputWeightShared && ctx.tensorsByName.has("token_embd.weight")) {
            const shared = ctx.tensorsByName.get("token_embd.weight");
            const expected = makeExpected([
                { label: "hidden_dim", value: ctx.embeddingLength },
                { label: "vocab_size", value: ctx.vocabSize },
            ], Array.isArray(shared.shape) ? shared.shape : []);
            lmHeadList.appendChild(
                createTensorNode(ctx, shared, {
                    displayName: "output.weight (shared with token_embd.weight)",
                    expected,
                    note: "Logits reuse token_embd.weight (tied embeddings).",
                }),
            );
        } else {
            lmHeadList.appendChild(
                createTensorNode(ctx, null, {
                    name: "output.weight",
                    displayName: "output.weight",
                    variant: "info",
                    badgeLabel: "info",
                    note: "Output projection not found. Logits are likely computed via tied embeddings (token_embd.weight) or integrated into the final transformer block.",
                }),
            );
        }
        const lmBias = ctx.tensorsByName.get("output.bias");
        if (lmBias) {
            lmHeadList.appendChild(createTensorNode(ctx, lmBias, { expected: getExpectedForTensor(ctx, lmBias) }));
        }
        tree.appendChild(lmHeadGroup.item);

        const unused = [];
        for (const tensor of ctx.tensors) {
            if (!ctx.used.has(tensor.name)) {
                unused.push(tensor);
            }
        }
        if (unused.length > 0) {
            const miscGroup = createGroup("Additional tensors");
            const miscList = createList();
            miscGroup.item.appendChild(miscList);
            unused.sort((a, b) => a.name.localeCompare(b.name));
            for (const tensor of unused) {
                miscList.appendChild(createTensorNode(ctx, tensor));
            }
            tree.appendChild(miscGroup.item);
        }

        target.appendChild(tree);

        const statusParts = [];
        if (ctx.architectureName) {
            statusParts.push(ctx.architectureName);
        }
        if (Number.isFinite(ctx.blockCount)) {
            statusParts.push(`${ctx.blockCount} blocks`);
        } else {
            statusParts.push(`${ctx.blockOrder.length} blocks`);
        }
        if (Number.isFinite(ctx.headCount)) {
            if (Number.isFinite(ctx.kvHeadCount) && ctx.kvHeadCount !== ctx.headCount) {
                statusParts.push(`${ctx.headCount} heads (${ctx.kvHeadCount} KV)`);
            } else {
                statusParts.push(`${ctx.headCount} heads`);
            }
        }
        if (Number.isFinite(ctx.keyLength)) {
            statusParts.push(`head_dim=${ctx.keyLength}`);
        }
        return statusParts.join(" - ");
    }

    function renderFallback(ctx, target) {
        const list = document.createElement("ul");
        list.className = "architecture-tree";
        const group = createGroup("Tensors");
        const tensorList = createList();
        group.item.appendChild(tensorList);
        const sorted = ctx.tensors.slice().sort((a, b) => a.name.localeCompare(b.name));
        for (const tensor of sorted) {
            tensorList.appendChild(createTensorNode(ctx, tensor));
        }
        list.appendChild(group.item);
        target.appendChild(list);
        if (ctx.architectureName) {
            return `Architecture ${ctx.architectureName} not yet documented.`;
        }
        return "Architecture metadata unavailable.";
    }

    function buildContext(metadataEntries, tensors) {
        const metadataIndex = buildMetadataIndex(metadataEntries);
        const tensorsByName = getTensorMap(tensors);
        const { map: blockMap, indices } = buildBlocks(tensors);
        const architectureName = metadataIndex.architectureName;

        const embeddingLength = getArchNumber(metadataIndex, architectureName, "embedding_length")
            ?? getTensorDim(tensorsByName, "token_embd.weight", 0)
            ?? getTensorDim(tensorsByName, "output.weight", 0);
        const vocabSize = getArchNumber(metadataIndex, architectureName, "vocab_size")
            ?? getTensorDim(tensorsByName, "token_embd.weight", 1)
            ?? getTensorDim(tensorsByName, "output.weight", 1);
        const blockCount = getArchNumber(metadataIndex, architectureName, "block_count");
        const headCount = getArchNumber(metadataIndex, architectureName, "attention.head_count");
        const kvHeadCount = getArchNumber(metadataIndex, architectureName, "attention.head_count_kv") ?? headCount;
        const keyLength = getArchNumber(metadataIndex, architectureName, "attention.key_length");
        const valueLength = getArchNumber(metadataIndex, architectureName, "attention.value_length") ?? keyLength;
        const ffArray = getArchArray(metadataIndex, architectureName, "feed_forward_length");
        const feedForwardLength = extractNumber(ffArray);
        const expertCount = getArchNumber(metadataIndex, architectureName, "expert_count");
        const expertUsedCount = getArchNumber(metadataIndex, architectureName, "expert_used_count");
        const expertFeedForwardLength = getArchNumber(metadataIndex, architectureName, "expert_feed_forward_length");
        const expertSharedFeedForwardLength = getArchNumber(metadataIndex, architectureName, "expert_shared_feed_forward_length");
        const expertChunkFeedForwardLength = getArchNumber(metadataIndex, architectureName, "expert_chunk_feed_forward_length");
        const leadingDenseBlockCount = getArchNumber(metadataIndex, architectureName, "leading_dense_block_count");
        const outputWeightShared = detectOutputWeightSharing(metadataIndex, architectureName);

        const isLlamaLike = indices.length > 0 || (architectureName && architectureName.toLowerCase().includes("llama"));

        const ctx = {
            type: isLlamaLike ? "llama" : "fallback",
            architectureName,
            metadataIndex,
            tensorsByName,
            tensors: Array.isArray(tensors) ? tensors : [],
            blocks: blockMap,
            blockOrder: [],
            blockCount,
            embeddingLength,
            vocabSize,
            headCount,
            kvHeadCount,
            keyLength,
            valueLength,
            feedForwardLength,
            expertCount,
            expertUsedCount,
            expertFeedForwardLength,
            expertSharedFeedForwardLength,
            expertChunkFeedForwardLength,
            leadingDenseBlockCount,
            outputWeightShared,
            used: new Set(),
        };

        if (isLlamaLike) {
            if (Number.isInteger(blockCount) && blockCount > 0) {
                const orderSet = new Set();
                for (let i = 0; i < blockCount; ++i) {
                    orderSet.add(i);
                }
                for (const idx of indices) {
                    orderSet.add(idx);
                }
                ctx.blockOrder = Array.from(orderSet).sort((a, b) => a - b);
            } else {
                ctx.blockOrder = indices;
            }
        }

        return ctx;
    }

    function render(params) {
        const { metadata, tensors, target } = params;
        if (!target) {
            return { statusMessage: "" };
        }
        target.innerHTML = "";
        const ctx = buildContext(metadata, tensors);
        if (ctx.type === "llama") {
            const statusMessage = renderLlama(ctx, target);
            return { statusMessage };
        }
        const statusMessage = renderFallback(ctx, target);
        return { statusMessage };
    }

    window.llamaViewerArchitecture = {
        render,
    };
})();
