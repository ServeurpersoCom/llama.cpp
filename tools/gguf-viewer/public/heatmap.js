function setHeatmapOverlay(message, mode = "idle") {
    if (!message) {
        heatmapOverlay.hidden = true;
        heatmapOverlay.textContent = "";
        heatmapOverlay.className = "heatmap-overlay";
        return;
    }
    heatmapOverlay.hidden = false;
    heatmapOverlay.textContent = message;
    heatmapOverlay.className = `heatmap-overlay${mode !== "idle" ? ` heatmap-overlay--${mode}` : ""}`;
}

function hideHeatmapTooltip() {
    if (heatmapHoverState.controller) {
        heatmapHoverState.controller.abort();
        heatmapHoverState.controller = null;
    }
    heatmapHoverState.canvasX = null;
    heatmapHoverState.canvasY = null;
    heatmapHoverState.globalX = null;
    heatmapHoverState.globalY = null;
    heatmapHoverState.slice = null;
    heatmapHoverState.requestX = null;
    heatmapHoverState.requestY = null;
    heatmapHoverState.requestSlice = null;
    heatmapHoverState.data = null;
    heatmapHoverState.clientX = null;
    heatmapHoverState.clientY = null;
    hideTooltipElement();
}

function updateHeatmapTooltipPosition() {
    if (!heatmapTooltip || !heatmapCanvasWrapper) {
        return;
    }
    if (heatmapTooltip.hidden) {
        return;
    }
    if (heatmapHoverState.clientX === null || heatmapHoverState.clientY === null) {
        return;
    }
    positionTooltipWithinHost(heatmapCanvasWrapper, heatmapHoverState.clientX, heatmapHoverState.clientY);
}

function renderHeatmapTooltip() {
    if (!heatmapTooltip || !heatmapHoverState.data) {
        return;
    }
    setTooltipHost(heatmapCanvasWrapper);
    const data = heatmapHoverState.data;
    const coord = data.coordinate || {};
    const xLabel = formatTooltipInteger(coord.x ?? heatmapHoverState.globalX ?? 0);
    const yLabel = formatTooltipInteger(coord.y ?? heatmapHoverState.globalY ?? 0);
    let titleText = `Weight [${xLabel}, ${yLabel}]`;
    const depth = heatmapState.layout && typeof heatmapState.layout.depth === "number" ? heatmapState.layout.depth : 1;
    if (depth > 1) {
        const sliceValue = coord.slice ?? heatmapHoverState.slice ?? 0;
        titleText += ` - slice ${formatTooltipInteger(sliceValue)}`;
    }
    const rows = [];
    rows.push({ label: "Element", value: `${formatTooltipInteger(data.index)} / ${formatTooltipInteger(data.count)}` });
    if (typeof data.tensorOffset === "number") {
        rows.push({ label: "Tensor offset", value: formatTooltipInteger(data.tensorOffset) });
    }
    if (typeof data.fileOffset === "number") {
        rows.push({ label: "File offset", value: formatTooltipInteger(data.fileOffset) });
    }
    if (typeof data.value === "number") {
        rows.push({ label: "Value", value: formatTooltipFloat(data.value) });
    }
    renderTooltipContent(titleText, rows);
    updateHeatmapTooltipPosition();
}

function showHeatmapTooltipMessage(message) {
    if (!heatmapTooltip) {
        return;
    }
    setTooltipHost(heatmapCanvasWrapper);
    renderTooltipContent(message);
    updateHeatmapTooltipPosition();
}

function requestHeatmapValue(globalX, globalY, slice) {
    if (!heatmapState.tensor) {
        return;
    }
    if (!currentModelPath) {
        return;
    }
    if (heatmapHoverState.controller) {
        heatmapHoverState.controller.abort();
        heatmapHoverState.controller = null;
    }
    const baseSlice = Number.isFinite(slice) ? slice : heatmapState.slice;
    const requestSlice = clampSlice(baseSlice);
    heatmapHoverState.requestX = globalX;
    heatmapHoverState.requestY = globalY;
    heatmapHoverState.requestSlice = requestSlice;
    heatmapHoverState.data = null;

    const controller = new AbortController();
    heatmapHoverState.controller = controller;

    const params = new URLSearchParams();
    params.set("x", String(globalX));
    params.set("y", String(globalY));
    params.set("slice", String(requestSlice));
    appendModelParam(params);

    const tensorName = encodeURIComponent(heatmapState.tensor.name);
    fetch(`api/tensors/${tensorName}/value?${params.toString()}`, { signal: controller.signal })
        .then((res) => {
            if (!res.ok) {
                return res.text().then((text) => {
                    throw new Error(text || `Request failed: ${res.status}`);
                });
            }
            return res.json();
        })
        .then((data) => {
            if (controller.signal.aborted) {
                return;
            }
            heatmapHoverState.controller = null;
            if (heatmapHoverState.requestX !== globalX || heatmapHoverState.requestY !== globalY || heatmapHoverState.requestSlice !== requestSlice) {
                return;
            }
            heatmapHoverState.data = data;
            renderHeatmapTooltip();
        })
        .catch((err) => {
            if (controller.signal.aborted) {
                return;
            }
            heatmapHoverState.controller = null;
            if (heatmapHoverState.requestX !== globalX || heatmapHoverState.requestY !== globalY || heatmapHoverState.requestSlice !== requestSlice) {
                return;
            }
            heatmapHoverState.data = null;
            showHeatmapTooltipMessage(`Error: ${err.message}`);
        });
}

function resetHeatmap(message = HEATMAP_DEFAULT_STREAM_MESSAGE) {
    const isDefaultStreamMessage = message === HEATMAP_DEFAULT_STREAM_MESSAGE;
    const histogramMessage = isDefaultStreamMessage
        ? STATISTICS_DEFAULT_HEADER_MESSAGE
        : message;
    heatmapHeaderMessage = isDefaultStreamMessage ? HEATMAP_DEFAULT_HEADER_MESSAGE : message;
    resetHistogram(histogramMessage);
    if (heatmapState.controller) {
        heatmapState.controller.abort();
        heatmapState.controller = null;
    }
    if (heatmapState.sliceController) {
        heatmapState.sliceController.abort();
        heatmapState.sliceController = null;
    }
    hideHeatmapTooltip();
    const preserveGrid = !!heatmapState.gridVisible;
    heatmapState.tensor = null;
    heatmapState.layout = { width: 1, height: 1, depth: 1 };
    heatmapState.windowX = 0;
    heatmapState.windowY = 0;
    heatmapState.slice = 0;
    heatmapState.viewMin = undefined;
    heatmapState.viewMax = undefined;
    heatmapState.sliceMin = undefined;
    heatmapState.sliceMax = undefined;
    heatmapState.scaleMin = -1;
    heatmapState.scaleMax = 1;
    heatmapState.scaleInitialized = false;
    heatmapState.scaleStep = HEATMAP_STEP_FALLBACK;
    heatmapState.offsetMode = false;
    heatmapState.blockSize = 1;
    heatmapState.gridVisible = preserveGrid;
    heatmapState.valid = 0;
    heatmapState.fetching = false;
    heatmapState.imageReady = false;
    heatmapState.values = [];
    heatmapState.viewWidth = heatmapCanvas.width;
    heatmapState.viewHeight = heatmapCanvas.height;
    heatmapState.pendingScale = null;
    heatmapState.sliceRequestId = 0;
    heatmapDragState.active = false;
    heatmapDragState.touchId = null;
    heatmapDragState.offsetX = 0;
    heatmapDragState.offsetY = 0;
    heatmapDragState.moved = false;
    if (heatmapCanvas) {
        heatmapCanvas.classList.remove("heatmap-canvas--dragging");
    }
    if (heatmapGainOffsetToggle) {
        heatmapGainOffsetToggle.checked = false;
    }
    if (heatmapGridToggle) {
        heatmapGridToggle.checked = preserveGrid;
    }
    syncHeatmapControls();
    clearHeatmapCanvas();
    updateHeatmapHeader();
    setHeatmapOverlay(message);
    updateHeatmapUrlState();
    syncHeatmapToViewport(false);
}

function formatShape(shape) {
    if (!Array.isArray(shape) || shape.length === 0) {
        return "[]";
    }
    const parts = shape.map((value) => {
        if (Number.isFinite(value)) {
            return String(value);
        }
        return typeof value === "string" && value.length > 0 ? value : "?";
    });
    return `[${parts.join(", ")}]`;
}

function layoutFromTensor(tensor) {
    const layout = tensor && typeof tensor.layout === "object" ? tensor.layout : {};
    let width = typeof layout.width === "number" ? layout.width : undefined;
    let height = typeof layout.height === "number" ? layout.height : undefined;
    let depth = typeof layout.depth === "number" ? layout.depth : undefined;

    if (!Number.isFinite(width) || width <= 0) {
        if (Array.isArray(tensor.shape) && tensor.shape.length > 0) {
            width = Number(tensor.shape[0]);
        }
    }
    if (!Number.isFinite(height) || height <= 0) {
        if (Array.isArray(tensor.shape) && tensor.shape.length > 1) {
            height = Number(tensor.shape[1]);
        } else {
            height = 1;
        }
    }
    if (!Number.isFinite(depth) || depth <= 0) {
        depth = 1;
    }

    width = Math.max(1, Math.floor(width || 1));
    height = Math.max(1, Math.floor(height || 1));
    depth = Math.max(1, Math.floor(depth || 1));

    return { width, height, depth };
}

function getTensorBlockSize(tensor) {
    if (!tensor || typeof tensor !== "object") {
        return 1;
    }
    const raw = Number(tensor.blockSize);
    if (!Number.isFinite(raw) || raw <= 0) {
        return 1;
    }
    return Math.max(1, Math.floor(raw));
}

function updateHeatmapHeader() {
    if (!heatmapStatus) {
        return;
    }

    if (!heatmapState.tensor) {
        const message = typeof heatmapHeaderMessage === "string" && heatmapHeaderMessage.length > 0
            ? heatmapHeaderMessage
            : HEATMAP_DEFAULT_HEADER_MESSAGE;
        setSectionStatus("heatmap", message);
        heatmapStatus.textContent = "";
        heatmapStatus.hidden = true;
        return;
    }

    const layout = heatmapState.layout || { width: heatmapState.viewWidth, height: heatmapState.viewHeight, depth: 1 };
    const layoutWidth = layout && typeof layout.width === "number" && layout.width > 0 ? layout.width : heatmapState.viewWidth;
    const layoutHeight = layout && typeof layout.height === "number" && layout.height > 0 ? layout.height : heatmapState.viewHeight;

    setSectionStatus("heatmap", null);
    heatmapHeaderMessage = HEATMAP_DEFAULT_HEADER_MESSAGE;

    const x0 = heatmapState.windowX;
    const y0 = heatmapState.windowY;
    const x1 = Math.min(layoutWidth - 1, x0 + heatmapState.viewWidth - 1);
    const y1 = Math.min(layoutHeight - 1, y0 + heatmapState.viewHeight - 1);
    const coverageTotal = heatmapState.viewWidth * heatmapState.viewHeight;
    const coverageText = coverageTotal > 0 ? `${heatmapState.valid}/${coverageTotal}` : "0/0";
    const statusParts = [
        `Layout ${layoutWidth} x ${layoutHeight}`,
        `X ${x0} - ${x1}`,
        `Y ${y0} - ${y1}`,
        `Cells ${coverageText}`,
    ];
    heatmapStatus.textContent = statusParts.join(" | ");
    heatmapStatus.hidden = false;
}

function ensureHeatmapBuffer(width, height) {
    const targetWidth = Math.max(1, Math.floor(width));
    const targetHeight = Math.max(1, Math.floor(height));
    if (heatmapCanvas.width !== targetWidth || heatmapCanvas.height !== targetHeight) {
        heatmapCanvas.width = targetWidth;
        heatmapCanvas.height = targetHeight;
        if (heatmapCtx) {
            heatmapCtx.imageSmoothingEnabled = false;
        }
        heatmapImageData = null;
    }
    if (!heatmapImageData && heatmapCtx) {
        heatmapImageData = heatmapCtx.createImageData(heatmapCanvas.width, heatmapCanvas.height);
    }
}

function getPaddingBottomPx(element) {
    if (!element || typeof window.getComputedStyle !== "function") {
        return 0;
    }
    const styles = window.getComputedStyle(element);
    if (!styles) {
        return 0;
    }
    const value = Number.parseFloat(styles.paddingBottom);
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.max(0, value);
}

function syncHeatmapToViewport(fetchAfterResize) {
    if (heatmapCanvasWrapper && heatmapCanvasWrapper.offsetParent === null) {
        return;
    }

    const viewport = getViewportDimensions();
    const viewportHeight = Math.max(1, Math.floor(viewport.height));
    let availableHeight = viewportHeight;
    if (heatmapCanvasWrapper) {
        const wrapperRect = heatmapCanvasWrapper.getBoundingClientRect();
        const topOffset = Math.max(0, Math.floor(wrapperRect.top));
        let paddingOffset = 1;
        const parentSection = heatmapCanvasWrapper.closest("section");
        if (parentSection) {
            paddingOffset += Math.ceil(getPaddingBottomPx(parentSection));
        }
        const mainElement = heatmapCanvasWrapper.closest("main");
        if (mainElement) {
            paddingOffset += Math.ceil(getPaddingBottomPx(mainElement));
        }
        availableHeight = Math.max(1, viewportHeight - topOffset - paddingOffset);
        heatmapCanvasWrapper.style.height = `${availableHeight}px`;
    }

    const wrapperWidth = heatmapCanvasWrapper
        ? Math.max(1, Math.floor(heatmapCanvasWrapper.clientWidth || heatmapCanvasWrapper.offsetWidth || 1))
        : Math.max(1, viewport.width);

    const hasTensor = !!heatmapState.tensor;
    const layoutWidth = hasTensor && heatmapState.layout && heatmapState.layout.width > 0
        ? heatmapState.layout.width
        : wrapperWidth;
    const layoutHeight = hasTensor && heatmapState.layout && heatmapState.layout.height > 0
        ? heatmapState.layout.height
        : availableHeight;

    const targetWidth = Math.max(1, Math.min(layoutWidth, wrapperWidth));
    const targetHeight = Math.max(1, Math.min(layoutHeight, availableHeight));

    const widthChanged = targetWidth !== heatmapState.viewWidth;
    const heightChanged = targetHeight !== heatmapState.viewHeight;

    if (widthChanged || heightChanged) {
        heatmapState.viewWidth = targetWidth;
        heatmapState.viewHeight = targetHeight;
        ensureHeatmapBuffer(targetWidth, targetHeight);
        heatmapDragState.offsetX = 0;
        heatmapDragState.offsetY = 0;
        heatmapDragState.moved = false;
        heatmapDragState.active = false;
        heatmapDragState.touchId = null;
        if (heatmapCanvas) {
            heatmapCanvas.classList.remove("heatmap-canvas--dragging");
        }
        if (hasTensor) {
            heatmapState.windowX = clampWindowX(heatmapState.windowX);
            heatmapState.windowY = clampWindowY(heatmapState.windowY);
            heatmapState.imageReady = false;
            syncHeatmapControls();
            if (fetchAfterResize) {
                void fetchHeatmapWindow();
            } else {
                clearHeatmapCanvas();
            }
        } else {
            clearHeatmapCanvas();
        }
    } else {
        ensureHeatmapBuffer(targetWidth, targetHeight);
        if (heatmapState.imageReady) {
            drawHeatmap();
        } else {
            clearHeatmapCanvas();
        }
    }

    updateHeatmapHeader();
}

function clearHeatmapCanvas() {
    if (!heatmapCtx) {
        return;
    }
    heatmapCtx.fillStyle = `rgb(${HEATMAP_BACKGROUND.r}, ${HEATMAP_BACKGROUND.g}, ${HEATMAP_BACKGROUND.b})`;
    heatmapCtx.fillRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
}

function getHeatmapGridStep(rawBlockSize) {
    const normalized = Number.isFinite(rawBlockSize) ? rawBlockSize : 1;
    if (normalized <= 1) {
        return 1;
    }
    const halved = Math.floor(normalized / 2);
    return halved > 1 ? halved : Math.floor(normalized);
}

function drawHeatmapGrid(translateX = 0, translateY = 0) {
    if (!heatmapCtx || !heatmapState.imageReady || !heatmapState.gridVisible) {
        return;
    }

    const rawBlockSize = Number.isFinite(heatmapState.blockSize) ? heatmapState.blockSize : 1;
    const blockSize = Math.max(1, getHeatmapGridStep(rawBlockSize));
    if (blockSize <= 1) {
        return;
    }

    const fallbackWidth = heatmapCanvas ? heatmapCanvas.width : 0;
    const fallbackHeight = heatmapCanvas ? heatmapCanvas.height : 0;
    const layout = heatmapState.layout || { width: heatmapState.viewWidth, height: heatmapState.viewHeight };
    const layoutWidthRaw = typeof layout.width === "number" ? layout.width : heatmapState.viewWidth;
    const layoutHeightRaw = typeof layout.height === "number" ? layout.height : heatmapState.viewHeight;
    const layoutWidth = Math.max(0, Math.floor(Number.isFinite(layoutWidthRaw) ? layoutWidthRaw : fallbackWidth));
    const layoutHeight = Math.max(0, Math.floor(Number.isFinite(layoutHeightRaw) ? layoutHeightRaw : fallbackHeight));
    const viewWidthRaw = Number.isFinite(heatmapState.viewWidth) ? heatmapState.viewWidth : fallbackWidth;
    const viewHeightRaw = Number.isFinite(heatmapState.viewHeight) ? heatmapState.viewHeight : fallbackHeight;
    const viewWidth = Math.max(0, Math.floor(Number.isFinite(viewWidthRaw) ? viewWidthRaw : 0));
    const viewHeight = Math.max(0, Math.floor(Number.isFinite(viewHeightRaw) ? viewHeightRaw : 0));
    if (viewWidth <= 0 || viewHeight <= 0 || layoutWidth <= 0 || layoutHeight <= 0) {
        return;
    }

    const windowX = Math.max(0, Math.floor(Number.isFinite(heatmapState.windowX) ? heatmapState.windowX : 0));
    const windowY = Math.max(0, Math.floor(Number.isFinite(heatmapState.windowY) ? heatmapState.windowY : 0));
    if (windowX >= layoutWidth || windowY >= layoutHeight) {
        return;
    }

    const widthLimit = Math.max(0, Math.min(viewWidth, layoutWidth - windowX));
    const heightLimit = Math.max(0, Math.min(viewHeight, layoutHeight - windowY));
    if (widthLimit <= 0 || heightLimit <= 0) {
        return;
    }

    const tx = Math.round(Number.isFinite(translateX) ? translateX : 0);
    const ty = Math.round(Number.isFinite(translateY) ? translateY : 0);
    const remainderX = windowX % blockSize;
    const remainderY = windowY % blockSize;
    const firstVertical = remainderX === 0 ? 0 : blockSize - remainderX;
    const firstHorizontal = remainderY === 0 ? 0 : blockSize - remainderY;

    heatmapCtx.save();
    heatmapCtx.translate(tx, ty);
    heatmapCtx.lineWidth = 1;
    heatmapCtx.lineCap = "butt";
    heatmapCtx.strokeStyle = "#fff";
    heatmapCtx.beginPath();

    let drew = false;

    for (let x = firstVertical; x < widthLimit; x += blockSize) {
        const pos = x + 0.5;
        heatmapCtx.moveTo(pos, 0);
        heatmapCtx.lineTo(pos, heightLimit);
        drew = true;
    }

    for (let y = firstHorizontal; y < heightLimit; y += blockSize) {
        const pos = y + 0.5;
        heatmapCtx.moveTo(0, pos);
        heatmapCtx.lineTo(widthLimit, pos);
        drew = true;
    }

    if (drew) {
        heatmapCtx.stroke();
    }
    heatmapCtx.restore();
}

function drawHeatmap(translateX = 0, translateY = 0) {
    clearHeatmapCanvas();
    if (!heatmapCtx || !heatmapImageData) {
        return;
    }
    const tx = Math.round(translateX);
    const ty = Math.round(translateY);
    heatmapCtx.putImageData(heatmapImageData, tx, ty);
    drawHeatmapGrid(tx, ty);
}

function clampWindowX(value) {
    const layoutWidth = heatmapState.layout && typeof heatmapState.layout.width === "number" ? heatmapState.layout.width : heatmapState.viewWidth;
    const max = Math.max(0, layoutWidth - heatmapState.viewWidth);
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.min(Math.max(0, Math.floor(value)), max);
}

function clampWindowY(value) {
    const layoutHeight = heatmapState.layout && typeof heatmapState.layout.height === "number" ? heatmapState.layout.height : heatmapState.viewHeight;
    const max = Math.max(0, layoutHeight - heatmapState.viewHeight);
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.min(Math.max(0, Math.floor(value)), max);
}

function clampSlice(value) {
    const depth = heatmapState.layout && typeof heatmapState.layout.depth === "number" && heatmapState.layout.depth > 0
        ? heatmapState.layout.depth
        : 1;
    if (!Number.isFinite(value)) {
        return 0;
    }
    return Math.min(Math.max(0, Math.floor(value)), depth - 1);
}

function clampDragOffsets() {
    const layoutWidth = heatmapState.layout && typeof heatmapState.layout.width === "number" ? heatmapState.layout.width : heatmapState.viewWidth;
    const layoutHeight = heatmapState.layout && typeof heatmapState.layout.height === "number" ? heatmapState.layout.height : heatmapState.viewHeight;
    const maxWindowX = Math.max(0, layoutWidth - heatmapState.viewWidth);
    const maxWindowY = Math.max(0, layoutHeight - heatmapState.viewHeight);

    const maxLeft = Math.min(heatmapState.windowX, heatmapState.viewWidth);
    const maxRight = Math.min(Math.max(0, maxWindowX - heatmapState.windowX), heatmapState.viewWidth);
    const maxUp = Math.min(heatmapState.windowY, heatmapState.viewHeight);
    const maxDown = Math.min(Math.max(0, maxWindowY - heatmapState.windowY), heatmapState.viewHeight);

    const clampedX = Math.min(Math.max(heatmapDragState.offsetX, -maxRight), maxLeft);
    const clampedY = Math.min(Math.max(heatmapDragState.offsetY, -maxDown), maxUp);

    const changed = clampedX !== heatmapDragState.offsetX || clampedY !== heatmapDragState.offsetY;
    heatmapDragState.offsetX = clampedX;
    heatmapDragState.offsetY = clampedY;
    return changed;
}

async function fetchHeatmapWindow() {
    if (!heatmapState.tensor) {
        return;
    }
    if (!currentModelPath) {
        return;
    }

    if (heatmapState.controller) {
        heatmapState.controller.abort();
    }

    const controller = new AbortController();
    heatmapState.controller = controller;
    heatmapState.fetching = true;
    if (!heatmapState.imageReady) {
        setHeatmapOverlay("Loading heatmap…", "loading");
    }

    const tensor = heatmapState.tensor;
    const layout = heatmapState.layout || { width: heatmapState.viewWidth, height: heatmapState.viewHeight, depth: 1 };
    heatmapState.slice = clampSlice(heatmapState.slice);
    const requestedSlice = heatmapState.slice;

    const params = new URLSearchParams();
    params.set("x", String(heatmapState.windowX));
    params.set("y", String(heatmapState.windowY));
    params.set("width", String(heatmapState.viewWidth));
    params.set("height", String(heatmapState.viewHeight));
    params.set("slice", String(requestedSlice));
    appendModelParam(params);

    try {
        const res = await fetch(`api/tensors/${encodeURIComponent(tensor.name)}/raw?${params.toString()}`, {
            signal: controller.signal,
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(text || `Request failed: ${res.status}`);
        }
        const data = await res.json();
        if (controller.signal.aborted) {
            return;
        }
        const layoutNode = data.layout || {};
        if (typeof layoutNode.width === "number" && layoutNode.width > 0) {
            heatmapState.layout.width = layoutNode.width;
        }
        if (typeof layoutNode.height === "number" && layoutNode.height > 0) {
            heatmapState.layout.height = layoutNode.height;
        }
        if (typeof layoutNode.depth === "number" && layoutNode.depth > 0) {
            heatmapState.layout.depth = layoutNode.depth;
        }

        const origin = data.origin || {};
        if (typeof origin.x === "number") {
            heatmapState.windowX = clampWindowX(origin.x);
        } else {
            heatmapState.windowX = clampWindowX(heatmapState.windowX);
        }
        if (typeof origin.y === "number") {
            heatmapState.windowY = clampWindowY(origin.y);
        } else {
            heatmapState.windowY = clampWindowY(heatmapState.windowY);
        }
        let nextSlice = heatmapState.slice;
        if (typeof origin.slice === "number") {
            nextSlice = clampSlice(origin.slice);
        }

        const viewport = data.viewport || {};
        let viewWidth = typeof viewport.width === "number" ? viewport.width : heatmapState.viewWidth;
        let viewHeight = typeof viewport.height === "number" ? viewport.height : heatmapState.viewHeight;
        viewWidth = Math.max(1, Math.floor(viewWidth));
        viewHeight = Math.max(1, Math.floor(viewHeight));
        viewWidth = Math.min(viewWidth, heatmapState.layout.width);
        viewHeight = Math.min(viewHeight, heatmapState.layout.height);

        if (heatmapState.viewWidth !== viewWidth || heatmapState.viewHeight !== viewHeight) {
            heatmapState.viewWidth = viewWidth;
            heatmapState.viewHeight = viewHeight;
            ensureHeatmapBuffer(viewWidth, viewHeight);
        }

        heatmapState.windowX = clampWindowX(heatmapState.windowX);
        heatmapState.windowY = clampWindowY(heatmapState.windowY);
        nextSlice = clampSlice(nextSlice);
        const previousSlice = heatmapState.slice;
        heatmapState.slice = nextSlice;
        const sliceChanged = heatmapState.slice !== previousSlice;
        if (sliceChanged) {
            if (heatmapState.sliceController) {
                heatmapState.sliceController.abort();
                heatmapState.sliceController = null;
            }
            heatmapState.sliceRequestId = (heatmapState.sliceRequestId || 0) + 1;
            heatmapState.sliceMin = undefined;
            heatmapState.sliceMax = undefined;
        }
        if (histogramState.slice !== heatmapState.slice && heatmapState.tensor) {
            histogramState.slice = heatmapState.slice;
            void fetchHistogram();
        }
        if (sliceChanged) {
            const shouldApplyScale = !heatmapState.scaleInitialized;
            void fetchSliceProperties({ applyScale: shouldApplyScale });
        }

        heatmapState.viewMin = typeof data.min === "number" ? data.min : undefined;
        heatmapState.viewMax = typeof data.max === "number" ? data.max : undefined;
        if (!heatmapState.scaleInitialized) {
            setHeatmapScale(heatmapState.viewMin, heatmapState.viewMax, { reapply: false, sync: false });
        } else {
            const clamped = sanitizeScale(heatmapState.scaleMin, heatmapState.scaleMax);
            heatmapState.scaleMin = clamped.min;
            heatmapState.scaleMax = clamped.max;
        }

        if (heatmapState.pendingScale && Number.isFinite(heatmapState.pendingScale.min) && Number.isFinite(heatmapState.pendingScale.max)) {
            const pending = heatmapState.pendingScale;
            heatmapState.pendingScale = null;
            setHeatmapScale(pending.min, pending.max);
        }

        const values = Array.isArray(data.values) ? data.values : [];
        heatmapState.values = values;
        updateHeatmapImage(values, heatmapState.scaleMin, heatmapState.scaleMax);
        syncHeatmapControls();
        updateHeatmapHeader();
        if (!heatmapState.imageReady) {
            setHeatmapOverlay("Reached tensor boundary.", "idle");
        } else {
            setHeatmapOverlay("", "idle");
        }
        if (
            heatmapHoverState.globalX !== null
            && heatmapHoverState.globalY !== null
            && heatmapHoverState.clientX !== null
            && heatmapHoverState.clientY !== null
        ) {
            requestHeatmapValue(heatmapHoverState.globalX, heatmapHoverState.globalY, heatmapState.slice);
        }
        updateHeatmapUrlState({ syncPending: true });
        syncHeatmapToViewport(false);
    } catch (err) {
        if (controller.signal.aborted) {
            return;
        }
        setHeatmapOverlay(`Failed to load heatmap: ${err.message}`, "error");
    } finally {
        if (heatmapState.controller === controller) {
            heatmapState.controller = null;
        }
        heatmapState.fetching = false;
        updateHeatmapUrlState({ syncPending: true });
    }
}

async function fetchSliceProperties(options = {}) {
    if (!heatmapState.tensor) {
        return null;
    }
    if (!currentModelPath) {
        return null;
    }

    if (heatmapState.sliceController) {
        heatmapState.sliceController.abort();
    }

    const controller = new AbortController();
    const requestId = (heatmapState.sliceRequestId || 0) + 1;
    heatmapState.sliceController = controller;
    heatmapState.sliceRequestId = requestId;

    const params = new URLSearchParams();
    const requestedSlice = Number.isInteger(options.slice)
        ? clampSlice(options.slice)
        : clampSlice(heatmapState.slice);
    params.set("slice", String(requestedSlice));
    appendModelParam(params);

    try {
        const res = await fetch(`api/tensors/${encodeURIComponent(heatmapState.tensor.name)}/slice/properties?${params.toString()}`, {
            signal: controller.signal,
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(text || `Request failed: ${res.status}`);
        }
        const data = await res.json();
        if (controller.signal.aborted || heatmapState.sliceRequestId !== requestId) {
            return null;
        }

        if (heatmapState.sliceController === controller) {
            heatmapState.sliceController = null;
        }

        const hasMin = typeof data.min === "number" && Number.isFinite(data.min);
        const hasMax = typeof data.max === "number" && Number.isFinite(data.max);
        heatmapState.sliceMin = hasMin ? data.min : undefined;
        heatmapState.sliceMax = hasMax ? data.max : undefined;

        const valid = typeof data.valid === "number" && data.valid > 0 ? data.valid : 0;

        if (options.applyScale) {
            if (hasMin && hasMax && valid > 0) {
                setHeatmapScale(data.min, data.max);
            } else {
                const reason = valid > 0
                    ? "Slice scale unavailable."
                    : "Slice has no finite values to scale.";
                heatmapHeaderMessage = reason;
                updateHeatmapHeader();
            }
        }

        syncHeatmapControls();
        return {
            slice: typeof data.slice === "number" ? data.slice : requestedSlice,
            min: heatmapState.sliceMin,
            max: heatmapState.sliceMax,
            valid,
        };
    } catch (err) {
        if (controller.signal.aborted || heatmapState.sliceRequestId !== requestId) {
            return null;
        }
        if (heatmapState.sliceController === controller) {
            heatmapState.sliceController = null;
        }
        if (options.applyScale) {
            const message = err instanceof Error ? err.message : String(err);
            heatmapHeaderMessage = `Slice scale request failed: ${message}`;
            updateHeatmapHeader();
        }
        console.error("Failed to fetch slice properties", err);
        syncHeatmapControls();
        return null;
    }
}

function updateHeatmapImage(values, minValue, maxValue) {
    ensureHeatmapBuffer(heatmapState.viewWidth, heatmapState.viewHeight);

    if (!heatmapCtx || !heatmapImageData) {
        heatmapState.imageReady = false;
        heatmapState.valid = 0;
        return;
    }

    const totalPixels = heatmapCanvas.width * heatmapCanvas.height;
    const buffer = heatmapImageData.data;
    const min = Number.isFinite(minValue) ? minValue : 0;
    const max = Number.isFinite(maxValue) ? maxValue : min;
    const span = max - min;

    for (let i = 0; i < totalPixels; ++i) {
        const idx = i * 4;
        buffer[idx] = HEATMAP_BACKGROUND.r;
        buffer[idx + 1] = HEATMAP_BACKGROUND.g;
        buffer[idx + 2] = HEATMAP_BACKGROUND.b;
        buffer[idx + 3] = 255;
    }

    let valid = 0;
    const limit = Math.min(totalPixels, Array.isArray(values) ? values.length : 0);
    for (let i = 0; i < limit; ++i) {
        const value = values[i];
        if (typeof value !== "number" || !Number.isFinite(value)) {
            continue;
        }

        let norm = span !== 0 ? (value - min) / span : 0;
        if (!Number.isFinite(norm)) {
            norm = 0;
        }
        norm = Math.min(1, Math.max(0, norm));
        const hue = norm * HEATMAP_HUE_MAX;
        const color = hslToRgb(hue, 1, 0.5);

        const idx = i * 4;
        buffer[idx] = color.r;
        buffer[idx + 1] = color.g;
        buffer[idx + 2] = color.b;
        buffer[idx + 3] = 255;
        valid += 1;
    }

    heatmapState.valid = valid;
    heatmapState.imageReady = valid > 0;
    drawHeatmap();
}

function openHeatmap(nameEncoded, options = {}) {
    const name = decodeURIComponent(nameEncoded);
    const tensor = tensorData.find((item) => item.name === name);
    if (!tensor) {
        setHeatmapOverlay("Tensor not found.", "error");
        return;
    }

    resetHeatmap();
    heatmapState.tensor = tensor;
    heatmapState.layout = layoutFromTensor(tensor);
    heatmapState.blockSize = getTensorBlockSize(tensor);
    if (typeof options.initialGrid === "boolean") {
        heatmapState.gridVisible = options.initialGrid;
        if (heatmapGridToggle) {
            heatmapGridToggle.checked = options.initialGrid;
        }
    }
    if (Number.isInteger(options.initialSlice) && options.initialSlice >= 0) {
        heatmapState.slice = clampSlice(options.initialSlice);
        histogramState.slice = heatmapState.slice;
        if (heatmapSliceInput) {
            heatmapSliceInput.value = String(heatmapState.slice + 1);
        }
    } else {
        histogramState.slice = heatmapState.slice;
    }
    if (options.initialWindow && typeof options.initialWindow === "object") {
        if (Number.isInteger(options.initialWindow.x) && options.initialWindow.x >= 0) {
            heatmapState.windowX = clampWindowX(options.initialWindow.x);
        }
        if (Number.isInteger(options.initialWindow.y) && options.initialWindow.y >= 0) {
            heatmapState.windowY = clampWindowY(options.initialWindow.y);
        }
    }
    if (options.initialScale && Number.isFinite(options.initialScale.min) && Number.isFinite(options.initialScale.max)) {
        heatmapState.pendingScale = { min: options.initialScale.min, max: options.initialScale.max };
    }
    histogramState.tensor = tensor;
    histogramState.slice = heatmapState.slice;
    syncHistogramToViewport(false);
    setHistogramOverlay("Generating histogram…", "loading");
    void fetchHistogram();
    syncHeatmapToViewport(false);
    syncHeatmapControls();
    setHeatmapOverlay("Preparing heatmap…", "loading");
    void fetchHeatmapWindow();
    const shouldApplySliceScale = !heatmapState.pendingScale;
    void fetchSliceProperties({ slice: heatmapState.slice, applyScale: shouldApplySliceScale });
    updateHeatmapUrlState({ syncPending: true });

    pendingHeatmapState = {
        tensor: tensor.name,
        slice: heatmapState.slice,
        min: Number.isFinite(heatmapState.scaleMin) ? heatmapState.scaleMin : null,
        max: Number.isFinite(heatmapState.scaleMax) ? heatmapState.scaleMax : null,
        grid: typeof heatmapState.gridVisible === "boolean" ? heatmapState.gridVisible : null,
        x: Number.isFinite(heatmapState.windowX) ? Math.max(0, Math.floor(heatmapState.windowX)) : null,
        y: Number.isFinite(heatmapState.windowY) ? Math.max(0, Math.floor(heatmapState.windowY)) : null,
        applied: true,
    };
}


function handleTensorAction(target) {
    if (!target || target.tagName !== "BUTTON" || !target.dataset.name) {
        return;
    }

    const action = target.dataset.action || "heatmap";
    const destination = action === "statistics" ? "statistics" : "heatmap";
    const current = normalizePageId(window.location.hash.slice(1));
    if (current !== destination) {
        window.location.hash = destination;
    }
    openHeatmap(target.dataset.name);
}

tensorBody.addEventListener("click", (event) => {
    const target = event.target;
    handleTensorAction(target);
});

if (architectureContent) {
    architectureContent.addEventListener("click", (event) => {
        const target = event.target;
        handleTensorAction(target);
    });
}

function commitSliceInput() {
    if (!heatmapSliceInput || !heatmapState.tensor) {
        return;
    }
    const depth = heatmapState.layout && typeof heatmapState.layout.depth === "number" && heatmapState.layout.depth > 0
        ? heatmapState.layout.depth
        : 1;
    let value = Math.floor(Number(heatmapSliceInput.value));
    if (!Number.isFinite(value)) {
        value = heatmapState.slice + 1;
    }
    value = Math.min(Math.max(1, value), depth);
    heatmapSliceInput.value = String(value);
    const nextSlice = clampSlice(value - 1);
    if (nextSlice !== heatmapState.slice) {
        if (heatmapState.sliceController) {
            heatmapState.sliceController.abort();
            heatmapState.sliceController = null;
        }
        heatmapState.sliceRequestId = (heatmapState.sliceRequestId || 0) + 1;
        heatmapState.sliceMin = undefined;
        heatmapState.sliceMax = undefined;
        heatmapState.slice = nextSlice;
        histogramState.slice = nextSlice;
        if (heatmapState.tensor) {
            void fetchHistogram();
        }
        updateHeatmapHeader();
        void fetchHeatmapWindow();
        updateHeatmapUrlState({ syncPending: true });
    }
}

function commitScaleInputs() {
    if (!heatmapState.tensor) {
        return;
    }
    let minValue = heatmapState.scaleMin;
    if (heatmapMinInput) {
        const parsedMin = Number.parseFloat(heatmapMinInput.value);
        if (Number.isFinite(parsedMin)) {
            minValue = parsedMin;
        }
    }
    let maxValue = heatmapState.scaleMax;
    if (heatmapMaxInput) {
        const parsedMax = Number.parseFloat(heatmapMaxInput.value);
        if (Number.isFinite(parsedMax)) {
            maxValue = parsedMax;
        }
    }
    setHeatmapScale(minValue, maxValue);
}

if (heatmapTensorSelect) {
    heatmapTensorSelect.addEventListener("change", () => {
        const value = heatmapTensorSelect.value;
        if (!value) {
            return;
        }
        if (heatmapState.tensor && encodeURIComponent(heatmapState.tensor.name) === value) {
            return;
        }
        openHeatmap(value);
    });
}

if (heatmapSliceInput) {
    heatmapSliceInput.addEventListener("change", commitSliceInput);
    heatmapSliceInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            commitSliceInput();
        }
    });
}

if (heatmapMinInput) {
    heatmapMinInput.addEventListener("change", commitScaleInputs);
    heatmapMinInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            commitScaleInputs();
        }
    });
}

if (heatmapMaxInput) {
    heatmapMaxInput.addEventListener("change", commitScaleInputs);
    heatmapMaxInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            commitScaleInputs();
        }
    });
}

if (heatmapSliceButton) {
    heatmapSliceButton.addEventListener("click", () => {
        if (!heatmapState.tensor) {
            return;
        }
        const previousSlice = heatmapState.slice;
        commitSliceInput();
        const targetSlice = heatmapState.slice;
        const sliceChanged = targetSlice !== previousSlice;
        if (!sliceChanged && Number.isFinite(heatmapState.sliceMin) && Number.isFinite(heatmapState.sliceMax)) {
            setHeatmapScale(heatmapState.sliceMin, heatmapState.sliceMax);
            return;
        }
        void fetchSliceProperties({ slice: targetSlice, applyScale: true });
    });
}

if (heatmapP5Button) {
    heatmapP5Button.addEventListener("click", () => {
        applyHeatmapPercentileRange(5);
    });
}

if (heatmapP10Button) {
    heatmapP10Button.addEventListener("click", () => {
        applyHeatmapPercentileRange(10);
    });
}

if (heatmapP20Button) {
    heatmapP20Button.addEventListener("click", () => {
        applyHeatmapPercentileRange(20);
    });
}

if (heatmapStepInput) {
    const commitStepInput = () => {
        commitHeatmapStep(heatmapStepInput.value);
    };
    heatmapStepInput.addEventListener("change", commitStepInput);
    heatmapStepInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            commitStepInput();
        }
    });
}

if (heatmapGainOffsetToggle) {
    const commitGainOffsetMode = () => {
        heatmapState.offsetMode = heatmapGainOffsetToggle.checked;
        syncHeatmapControls();
    };
    heatmapGainOffsetToggle.addEventListener("change", commitGainOffsetMode);
    heatmapGainOffsetToggle.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            heatmapGainOffsetToggle.checked = !heatmapGainOffsetToggle.checked;
            commitGainOffsetMode();
        }
    });
}

if (heatmapGridToggle) {
    const commitGridVisibility = () => {
        const next = heatmapGridToggle.checked;
        if (heatmapState.gridVisible !== next) {
            heatmapState.gridVisible = next;
            if (heatmapState.imageReady) {
                drawHeatmap();
            }
        } else {
            heatmapState.gridVisible = next;
        }
        syncHeatmapControls();
        updateHeatmapUrlState({ syncPending: true });
    };
    heatmapGridToggle.addEventListener("change", commitGridVisibility);
    heatmapGridToggle.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            heatmapGridToggle.checked = !heatmapGridToggle.checked;
            commitGridVisibility();
        }
    });
}

if (heatmapTightenButton) {
    heatmapTightenButton.addEventListener("click", () => {
        adjustHeatmapContrast(1);
    });
}

if (heatmapWidenButton) {
    heatmapWidenButton.addEventListener("click", () => {
        adjustHeatmapContrast(-1);
    });
}

function heatmapCanvasScale() {
    const rect = heatmapCanvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? heatmapCanvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? heatmapCanvas.height / rect.height : 1;
    return { scaleX, scaleY };
}

function beginHeatmapDrag(clientX, clientY, options = {}) {
    if (!heatmapState.tensor || heatmapDragState.active) {
        return false;
    }
    hideHeatmapTooltip();
    heatmapDragState.active = true;
    heatmapDragState.lastX = clientX;
    heatmapDragState.lastY = clientY;
    heatmapDragState.offsetX = 0;
    heatmapDragState.offsetY = 0;
    heatmapDragState.moved = false;
    heatmapDragState.touchId = typeof options.touchId === "number" ? options.touchId : null;
    heatmapCanvas.classList.add("heatmap-canvas--dragging");
    return true;
}

function updateHeatmapDragPosition(clientX, clientY) {
    if (!heatmapDragState.active) {
        return;
    }
    const { scaleX, scaleY } = heatmapCanvasScale();
    const dx = (clientX - heatmapDragState.lastX) * scaleX;
    const dy = (clientY - heatmapDragState.lastY) * scaleY;
    heatmapDragState.lastX = clientX;
    heatmapDragState.lastY = clientY;

    if (dx === 0 && dy === 0) {
        return;
    }

    const beforeX = heatmapDragState.offsetX;
    const beforeY = heatmapDragState.offsetY;
    heatmapDragState.offsetX += dx;
    heatmapDragState.offsetY += dy;
    clampDragOffsets();
    if (heatmapDragState.offsetX !== beforeX || heatmapDragState.offsetY !== beforeY) {
        heatmapDragState.moved = true;
    }

    if (heatmapState.imageReady) {
        drawHeatmap(heatmapDragState.offsetX, heatmapDragState.offsetY);
    }
}

function findTouchById(touchList, identifier) {
    if (!touchList || typeof touchList.length !== "number") {
        return null;
    }
    for (let i = 0; i < touchList.length; i += 1) {
        const touch = typeof touchList.item === "function" ? touchList.item(i) : touchList[i];
        if (touch && touch.identifier === identifier) {
            return touch;
        }
    }
    return null;
}

function getTrackedTouch(event) {
    if (heatmapDragState.touchId === null) {
        return null;
    }
    const { touchId } = heatmapDragState;
    return findTouchById(event.touches, touchId) || findTouchById(event.changedTouches, touchId);
}

heatmapCanvas.addEventListener("mousedown", (event) => {
    if (!beginHeatmapDrag(event.clientX, event.clientY)) {
        return;
    }
    event.preventDefault();
});

heatmapCanvas.addEventListener("touchstart", (event) => {
    const touch = event.changedTouches && event.changedTouches.length > 0
        ? event.changedTouches[0]
        : event.touches && event.touches.length > 0
            ? event.touches[0]
            : null;
    if (!touch) {
        return;
    }
    if (!beginHeatmapDrag(touch.clientX, touch.clientY, { touchId: touch.identifier })) {
        return;
    }
    event.preventDefault();
}, { passive: false });

heatmapCanvas.addEventListener("mousemove", (event) => {
    if (!heatmapState.tensor || !heatmapState.imageReady || heatmapDragState.active) {
        return;
    }
    const rect = heatmapCanvas.getBoundingClientRect();
    const { scaleX, scaleY } = heatmapCanvasScale();
    const canvasX = Math.floor((event.clientX - rect.left) * scaleX);
    const canvasY = Math.floor((event.clientY - rect.top) * scaleY);

    heatmapHoverState.clientX = event.clientX;
    heatmapHoverState.clientY = event.clientY;

    if (canvasX < 0 || canvasY < 0 || canvasX >= heatmapState.viewWidth || canvasY >= heatmapState.viewHeight) {
        hideHeatmapTooltip();
        return;
    }

    const globalX = heatmapState.windowX + canvasX;
    const globalY = heatmapState.windowY + canvasY;
    const slice = heatmapState.slice;
    const sameCoordinate = heatmapHoverState.globalX === globalX
        && heatmapHoverState.globalY === globalY
        && heatmapHoverState.slice === slice;

    heatmapHoverState.canvasX = canvasX;
    heatmapHoverState.canvasY = canvasY;
    heatmapHoverState.globalX = globalX;
    heatmapHoverState.globalY = globalY;
    heatmapHoverState.slice = slice;

    if (sameCoordinate) {
        if (heatmapHoverState.data) {
            renderHeatmapTooltip();
        } else if (heatmapTooltip && !heatmapTooltip.hidden) {
            updateHeatmapTooltipPosition();
        }
        return;
    }

    requestHeatmapValue(globalX, globalY, slice);
});

window.addEventListener("mousemove", (event) => {
    updateHeatmapDragPosition(event.clientX, event.clientY);
});

window.addEventListener("touchmove", (event) => {
    if (!heatmapDragState.active) {
        return;
    }
    const touch = getTrackedTouch(event);
    if (!touch) {
        return;
    }
    event.preventDefault();
    updateHeatmapDragPosition(touch.clientX, touch.clientY);
}, { passive: false });

function endHeatmapDrag() {
    if (!heatmapDragState.active) {
        return;
    }
    heatmapDragState.active = false;
    heatmapDragState.touchId = null;
    heatmapCanvas.classList.remove("heatmap-canvas--dragging");
    clampDragOffsets();
    if (!heatmapDragState.moved) {
        drawHeatmap();
        return;
    }

    const deltaX = -Math.round(heatmapDragState.offsetX);
    const deltaY = -Math.round(heatmapDragState.offsetY);
    heatmapDragState.offsetX = 0;
    heatmapDragState.offsetY = 0;
    heatmapDragState.moved = false;

    const nextX = clampWindowX(heatmapState.windowX + deltaX);
    const nextY = clampWindowY(heatmapState.windowY + deltaY);

    if (nextX !== heatmapState.windowX || nextY !== heatmapState.windowY) {
        heatmapState.windowX = nextX;
        heatmapState.windowY = nextY;
        updateHeatmapHeader();
        updateHeatmapUrlState({ syncPending: true });
        void fetchHeatmapWindow();
    } else {
        drawHeatmap();
    }
}

window.addEventListener("mouseup", endHeatmapDrag);
window.addEventListener("touchend", (event) => {
    if (!heatmapDragState.active) {
        return;
    }
    const touch = getTrackedTouch(event);
    if (!touch) {
        return;
    }
    event.preventDefault();
    endHeatmapDrag();
}, { passive: false });
window.addEventListener("touchcancel", (event) => {
    if (!heatmapDragState.active) {
        return;
    }
    const touch = getTrackedTouch(event);
    if (!touch) {
        return;
    }
    endHeatmapDrag();
});
heatmapCanvas.addEventListener("mouseleave", () => {
    hideHeatmapTooltip();
    if (heatmapDragState.active) {
        endHeatmapDrag();
    }
});

heatmapCanvas.addEventListener("wheel", (event) => {
    if (!heatmapState.tensor || !heatmapState.imageReady) {
        return;
    }
    if (event.deltaY === 0) {
        return;
    }
    event.preventDefault();
    const magnitude = Math.max(1, Math.round(Math.abs(event.deltaY) / 120));
    adjustHeatmapContrast(event.deltaY < 0 ? 1 : -1, magnitude);
}, { passive: false });

