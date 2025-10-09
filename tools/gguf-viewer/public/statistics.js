function clearHistogramCanvas() {
    if (!histogramCtx || !histogramCanvas) {
        return;
    }
    histogramCtx.fillStyle = "#000";
    histogramCtx.fillRect(0, 0, histogramCanvas.width, histogramCanvas.height);
}

function setHistogramOverlay(message, mode = "idle") {
    if (!histogramOverlay) {
        return;
    }
    if (!message) {
        histogramOverlay.hidden = true;
        histogramOverlay.textContent = "";
        histogramOverlay.className = "histogram-overlay";
        return;
    }
    histogramOverlay.hidden = false;
    histogramOverlay.textContent = message;
    histogramOverlay.className = `histogram-overlay${mode !== "idle" ? ` histogram-overlay--${mode}` : ""}`;
}

function drawHistogram() {
    if (!histogramCtx || !histogramCanvas) {
        return;
    }

    const width = histogramCanvas.width;
    const height = histogramCanvas.height;
    histogramCtx.fillStyle = "#000";
    histogramCtx.fillRect(0, 0, width, height);

    const bins = Array.isArray(histogramState.bins) ? histogramState.bins : [];
    const maxCount = histogramState.maxCount;

    if (!Array.isArray(bins) || bins.length === 0 || !Number.isFinite(maxCount) || maxCount <= 0) {
        return;
    }

    const binWidth = width / bins.length;
    histogramCtx.fillStyle = "#fff";
    for (let i = 0; i < bins.length; ++i) {
        const raw = bins[i];
        const count = Number.isFinite(raw) ? Number(raw) : 0;
        if (count <= 0) {
            continue;
        }
        const scale = count / maxCount;
        const barHeight = Math.max(1, Math.round(scale * height));
        const x0 = Math.floor(i * binWidth);
        const x1 = Math.floor((i + 1) * binWidth);
        const w = Math.max(1, x1 - x0);
        const y = height - barHeight;
        histogramCtx.fillRect(x0, y, w, barHeight);
    }

    const rangeMin = Number.isFinite(histogramState.rangeMin) ? histogramState.rangeMin : Number.NaN;
    const rangeMax = Number.isFinite(histogramState.rangeMax) ? histogramState.rangeMax : Number.NaN;
    if (Number.isFinite(rangeMin) && Number.isFinite(rangeMax) && rangeMax > rangeMin) {
        if (rangeMin <= 0 && rangeMax >= 0) {
            const span = rangeMax - rangeMin;
            const zeroOffset = -rangeMin;
            const zeroRatio = zeroOffset / span;
            if (Number.isFinite(zeroRatio)) {
                const x = Math.round(zeroRatio * width);
                const clampedX = Math.max(0, Math.min(width - 1, x));
                histogramCtx.fillStyle = "#f87171";
                histogramCtx.fillRect(clampedX, 0, 1, height);
            }
        }
    }
}

function updateStatisticsHeader() {
    if (!statisticsStatus) {
        return;
    }

    const hasHistogram = !!(heatmapState.tensor
        && Array.isArray(histogramState.bins)
        && histogramState.bins.length > 0
        && Number.isFinite(histogramState.maxCount)
        && histogramState.maxCount > 0);

    if (!hasHistogram) {
        const message = typeof statisticsHeaderMessage === "string" && statisticsHeaderMessage.length > 0
            ? statisticsHeaderMessage
            : STATISTICS_DEFAULT_HEADER_MESSAGE;
        setSectionStatus("statistics", message);
        statisticsStatus.textContent = "";
        statisticsStatus.hidden = true;
        return;
    }

    setSectionStatus("statistics", null);
    statisticsHeaderMessage = STATISTICS_DEFAULT_HEADER_MESSAGE;

    const parts = [];
    parts.push(`Bins ${formatTooltipInteger(histogramState.bins.length)}`);

    if (Number.isFinite(histogramState.total)) {
        parts.push(`Total ${formatTooltipInteger(histogramState.total)}`);
    }

    if (Number.isFinite(histogramState.zeroCount) && histogramState.zeroCount > 0) {
        parts.push(`Zeros ${formatTooltipInteger(histogramState.zeroCount)}`);
    }

    const clippedLow = Number.isFinite(histogramState.clippedLow) ? histogramState.clippedLow : 0;
    const clippedHigh = Number.isFinite(histogramState.clippedHigh) ? histogramState.clippedHigh : 0;
    if (clippedLow > 0 || clippedHigh > 0) {
        const lowLabel = formatTooltipInteger(clippedLow);
        const highLabel = formatTooltipInteger(clippedHigh);
        parts.push(`Outliers ${lowLabel} low / ${highLabel} high`);
    }

    if (Number.isFinite(histogramState.rangeMin)
        && Number.isFinite(histogramState.rangeMax)
        && histogramState.rangeMax >= histogramState.rangeMin) {
        const minLabel = formatTooltipFloat(histogramState.rangeMin);
        const maxLabel = formatTooltipFloat(histogramState.rangeMax);
        parts.push(`Range ${minLabel} - ${maxLabel}`);
    }

    statisticsStatus.textContent = parts.join(" | ");
    statisticsStatus.hidden = parts.length === 0;
}

function syncHistogramToViewport(fetchAfterResize) {
    if (!histogramCanvas || !histogramCanvasWrapper) {
        return;
    }
    if (histogramCanvasWrapper.offsetParent === null) {
        return;
    }

    const targetWidth = Math.max(1, Math.floor(histogramCanvasWrapper.clientWidth || histogramCanvasWrapper.offsetWidth || 1));
    const targetHeight = HISTOGRAM_CANVAS_HEIGHT;

    const widthChanged = targetWidth !== histogramState.viewWidth;
    const heightChanged = targetHeight !== histogramState.viewHeight;

    if (widthChanged || heightChanged) {
        histogramState.viewWidth = targetWidth;
        histogramState.viewHeight = targetHeight;
        histogramCanvas.width = targetWidth;
        histogramCanvas.height = targetHeight;
        if (histogramCtx) {
            histogramCtx.imageSmoothingEnabled = false;
        }
        histogramState.bins = [];
        histogramState.maxCount = 0;
        histogramState.total = 0;
        histogramState.clippedLow = 0;
        histogramState.clippedHigh = 0;
        histogramState.zeroCount = 0;
        clearHistogramCanvas();
        if (heatmapState.tensor && fetchAfterResize) {
            void fetchHistogram();
        }
        return;
    }

    if (histogramState.bins.length > 0 && histogramState.maxCount > 0) {
        drawHistogram();
    }
}

async function fetchHistogram() {
    if (!heatmapState.tensor || !histogramCanvas) {
        return;
    }
    if (!currentModelPath) {
        return;
    }

    if (histogramState.controller) {
        histogramState.controller.abort();
        histogramState.controller = null;
    }

    const width = Math.max(1, Math.floor(histogramState.viewWidth || histogramCanvas.width || 1));
    const height = Math.max(1, Math.floor(histogramState.viewHeight || histogramCanvas.height || 1));

    const controller = new AbortController();
    histogramState.controller = controller;
    histogramState.fetching = true;
    const loadingMessage = "Generating histogram…";
    statisticsHeaderMessage = loadingMessage;
    updateStatisticsHeader();
    setHistogramOverlay(loadingMessage, "loading");

    try {
        const name = encodeURIComponent(heatmapState.tensor.name);
        const params = new URLSearchParams();
        params.set("width", String(width));
        params.set("height", String(height));
        params.set("slice", String(heatmapState.slice));
        appendModelParam(params);
        const response = await fetch(`api/tensors/${name}/histogram?${params.toString()}`, { signal: controller.signal });
        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`.trim());
        }
        const data = await response.json();
        const bins = Array.isArray(data.bins)
            ? data.bins.map((value) => {
                const num = Number(value);
                return Number.isFinite(num) && num > 0 ? num : 0;
            })
            : [];
        const maxCount = Number.isFinite(data.maxCount) ? Number(data.maxCount) : 0;
        const rangeMin = Number.isFinite(data.min) ? Number(data.min) : Number.NaN;
        const rangeMax = Number.isFinite(data.max) ? Number(data.max) : Number.NaN;
        const clippedLow = Number.isFinite(data.clippedLow) ? Number(data.clippedLow) : 0;
        const clippedHigh = Number.isFinite(data.clippedHigh) ? Number(data.clippedHigh) : 0;
        const zeroCount = Number.isFinite(data.zeroCount) ? Number(data.zeroCount) : 0;
        histogramState.tensor = heatmapState.tensor;
        histogramState.slice = typeof data.slice === "number" ? data.slice : heatmapState.slice;
        histogramState.bins = bins;
        histogramState.maxCount = maxCount;
        histogramState.rangeMin = rangeMin;
        histogramState.rangeMax = rangeMax;
        histogramState.clippedLow = clippedLow;
        histogramState.clippedHigh = clippedHigh;
        histogramState.zeroCount = zeroCount;
        if (Number.isFinite(data.total)) {
            histogramState.total = Number(data.total);
        } else {
            histogramState.total = bins.reduce((sum, value) => sum + value, 0);
        }

        if (bins.length === 0 || maxCount <= 0) {
            const emptyMessage = "No histogram data available for this slice.";
            clearHistogramCanvas();
            setHistogramOverlay(emptyMessage);
            statisticsHeaderMessage = emptyMessage;
            updateStatisticsHeader();
            return;
        }

        setHistogramOverlay("", "idle");
        drawHistogram();
        statisticsHeaderMessage = STATISTICS_DEFAULT_HEADER_MESSAGE;
        updateStatisticsHeader();
    } catch (err) {
        if (controller.signal.aborted) {
            return;
        }
        clearHistogramCanvas();
        const errorMessage = `Failed to load histogram: ${err.message}`;
        setHistogramOverlay(errorMessage, "error");
        statisticsHeaderMessage = errorMessage;
        updateStatisticsHeader();
    } finally {
        if (histogramState.controller === controller) {
            histogramState.controller = null;
        }
        histogramState.fetching = false;
    }
}

function resetHistogram(message = STATISTICS_DEFAULT_HEADER_MESSAGE) {
    if (histogramState.controller) {
        histogramState.controller.abort();
        histogramState.controller = null;
    }
    histogramState.tensor = null;
    histogramState.slice = 0;
    histogramState.bins = [];
    histogramState.maxCount = 0;
    histogramState.total = 0;
    histogramState.clippedLow = 0;
    histogramState.clippedHigh = 0;
    histogramState.zeroCount = 0;
    histogramState.rangeMin = Number.NaN;
    histogramState.rangeMax = Number.NaN;
    histogramState.fetching = false;
    clearHistogramCanvas();
    setHistogramOverlay(message);
    statisticsHeaderMessage = message;
    updateStatisticsHeader();
    syncHistogramToViewport(false);
}
