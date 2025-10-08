function setTokenizerMessage(message) {
    setSectionStatus("tokenizer", message);
    renderEmptyNote(tokenizerContent, message);
}

function formatTokenForDisplay(token) {
    if (!token) {
        return "∅";
    }
    let output = "";
    for (const ch of token) {
        switch (ch) {
            case " ":
                output += "␠";
                break;
            case "\n":
                output += "⏎";
                break;
            case "\r":
                output += "␍";
                break;
            case "\t":
                output += "⇥";
                break;
            default: {
                const code = ch.codePointAt(0);
                if (code !== undefined && (code < 0x20 || code === 0x7f)) {
                    output += `\\u{${code.toString(16)}}`;
                } else {
                    output += ch;
                }
            }
        }
    }
    return output;
}

function tokenToHex(token) {
    if (!token) {
        return "";
    }
    return Array.from(textEncoder.encode(token))
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join(" ");
}

function formatTokenType(value) {
    if (value === undefined) {
        return "N/A";
    }
    const label = TOKEN_TYPE_LABELS[value];
    return label ? `${value} (${label})` : String(value);
}

function hideTokenTooltip() {
    tokenTooltipState.activeCell = null;
    tokenTooltipState.host = null;
    hideTooltipElement();
}

function showTokenTooltip(event, cell, item, host) {
    if (!heatmapTooltip) {
        return;
    }
    setTooltipHost(host);
    const displayText = formatTokenForDisplay(item.token);
    const hex = tokenToHex(item.token);
    const rows = [
        { label: "Text", value: displayText, wrap: true },
        { label: "Hex", value: hex ? hex : "(empty)", wrap: true },
        { label: "Score", value: typeof item.score === "number" ? formatTooltipFloat(item.score) : "N/A", align: "left" },
        { label: "Type", value: formatTokenType(item.tokenType), align: "left" },
    ];
    renderTooltipContent(`Token ${formatTooltipInteger(item.index)}`, rows);
    positionTooltipWithinHost(host, event.clientX, event.clientY);
    tokenTooltipState.activeCell = cell;
    tokenTooltipState.host = host;
}

function updateTokenTooltipPosition(event) {
    if (!tokenTooltipState.host) {
        return;
    }
    positionTooltipWithinHost(tokenTooltipState.host, event.clientX, event.clientY);
}

function renderTokenizerGrid(state, data) {
    const container = document.createElement("div");

    hideTokenTooltip();

    const controls = document.createElement("div");
    controls.className = "controls tokenizer-controls";

    const offsetGroup = document.createElement("div");
    offsetGroup.className = "control-group";
    const offsetLabel = document.createElement("label");
    const offsetText = document.createElement("span");
    offsetText.textContent = "Offset";
    const offsetInput = document.createElement("input");
    offsetInput.type = "number";
    offsetInput.min = "0";
    offsetInput.value = state.offset;
    offsetLabel.appendChild(offsetText);
    offsetLabel.appendChild(offsetInput);
    offsetGroup.appendChild(offsetLabel);

    const gridGroup = document.createElement("div");
    gridGroup.className = "control-group";
    const gridLabel = document.createElement("label");
    const gridText = document.createElement("span");
    gridText.textContent = "Grid size";
    const gridInput = document.createElement("input");
    gridInput.type = "number";
    gridInput.min = "1";
    gridInput.max = "64";
    gridInput.value = state.gridSize;
    gridLabel.appendChild(gridText);
    gridLabel.appendChild(gridInput);
    gridGroup.appendChild(gridLabel);

    const badge = document.createElement("span");
    badge.className = "badge";
    if (data.items.length > 0) {
        const start = data.items[0].index;
        const end = data.items[data.items.length - 1].index;
        badge.textContent = `${start} - ${end} / ${data.total}`;
    } else {
        badge.textContent = `0 / ${data.total}`;
    }

    const prev = document.createElement("button");
    prev.textContent = "Prev";

    const next = document.createElement("button");
    next.textContent = "Next";

    controls.appendChild(offsetGroup);
    controls.appendChild(gridGroup);
    controls.appendChild(prev);
    controls.appendChild(next);
    controls.appendChild(badge);

    const grid = document.createElement("div");
    grid.className = "token-grid";
    grid.style.gridTemplateColumns = `repeat(${state.gridSize}, minmax(0, 1fr))`;

    const totalCells = state.gridSize * state.gridSize;
    for (let i = 0; i < totalCells; ++i) {
        const item = data.items[i];
        const cell = document.createElement("div");
        cell.className = "token-cell";

        if (item) {
            const span = document.createElement("span");
            span.className = "token-text";
            span.textContent = formatTokenForDisplay(item.token);
            cell.appendChild(span);
            cell.addEventListener("mouseenter", (event) => {
                showTokenTooltip(event, cell, item, grid);
            });
            cell.addEventListener("mousemove", (event) => {
                if (tokenTooltipState.activeCell !== cell) {
                    showTokenTooltip(event, cell, item, grid);
                    return;
                }
                updateTokenTooltipPosition(event);
            });
            cell.addEventListener("mouseleave", () => {
                if (tokenTooltipState.activeCell === cell) {
                    hideTokenTooltip();
                }
            });
        } else {
            cell.classList.add("token-cell--empty");
            cell.textContent = "";
        }

        grid.appendChild(cell);
    }

    grid.addEventListener("mouseleave", () => {
        if (tokenTooltipState.host === grid) {
            hideTokenTooltip();
        }
    });

    container.appendChild(controls);
    container.appendChild(grid);

    return { container, offsetInput, gridInput, prev, next, badge };
}

async function loadTokenizer(offset = tokenizerState.offset) {
    try {
        hideTokenTooltip();
        const gridSize = Math.max(1, tokenizerState.gridSize);
        const requestedOffset = Math.max(0, Math.floor(Number.isFinite(offset) ? offset : 0));
        const limit = gridSize * gridSize;
        const data = await fetchJSON(withModel(`api/tokenizer?offset=${requestedOffset}&limit=${limit}`));
        tokenizerContent.innerHTML = "";
        if (!data || typeof data !== "object" || !data.hasTokenizer) {
            setTokenizerMessage("Tokenizer metadata not found in this model.");
            return;
        }
        tokenizerState = {
            ...tokenizerState,
            offset: Number.isFinite(data.offset) ? data.offset : requestedOffset,
            total: Number.isFinite(data.total) ? data.total : 0,
        };

        if (!Array.isArray(data.items)) {
            data.items = [];
        }

        if (data.total > 0 && data.items.length === 0 && data.offset >= data.total) {
            const fallbackOffset = Math.max(0, data.total - limit);
            if (fallbackOffset !== data.offset) {
                loadTokenizer(fallbackOffset);
                return;
            }
        }

        const layout = renderTokenizerGrid(tokenizerState, data);
        tokenizerContent.appendChild(layout.container);
        setSectionStatus("tokenizer", null);

        const step = tokenizerState.gridSize * tokenizerState.gridSize;
        const nextOffset = tokenizerState.offset + step;

        layout.offsetInput.value = tokenizerState.offset;
        layout.gridInput.value = tokenizerState.gridSize;

        layout.prev.disabled = tokenizerState.offset === 0;
        layout.next.disabled = tokenizerState.total === 0 || nextOffset >= tokenizerState.total;

        layout.prev.addEventListener("click", () => {
            if (tokenizerState.offset === 0) {
                return;
            }
            const newOffset = Math.max(0, tokenizerState.offset - step);
            loadTokenizer(newOffset);
        });

        layout.next.addEventListener("click", () => {
            if (tokenizerState.total === 0) {
                return;
            }
            const target = tokenizerState.offset + step;
            if (target >= tokenizerState.total) {
                return;
            }
            loadTokenizer(target);
        });

        const commitOffsetChange = () => {
            if (tokenizerState.total === 0) {
                layout.offsetInput.value = 0;
                return;
            }
            let value = Math.floor(Number(layout.offsetInput.value));
            if (!Number.isFinite(value) || value < 0) {
                value = tokenizerState.offset;
            }
            value = Math.min(value, Math.max(0, tokenizerState.total - 1));
            layout.offsetInput.value = value;
            if (value !== tokenizerState.offset) {
                loadTokenizer(value);
            }
        };

        layout.offsetInput.addEventListener("change", commitOffsetChange);
        layout.offsetInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                commitOffsetChange();
            }
        });

        layout.gridInput.addEventListener("change", () => {
            let value = Math.floor(Number(layout.gridInput.value));
            if (!Number.isFinite(value) || value < 1) {
                value = tokenizerState.gridSize;
            }
            value = Math.min(value, 64);
            if (value !== tokenizerState.gridSize) {
                tokenizerState.gridSize = value;
                loadTokenizer(tokenizerState.offset);
            } else {
                layout.gridInput.value = tokenizerState.gridSize;
            }
        });
    } catch (err) {
        setTokenizerMessage(err.status === 409 ? NO_MODEL_MESSAGE : err.message);
    }
}

