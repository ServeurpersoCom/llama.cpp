
const infoCards = document.getElementById("info-cards");
const kvBody = document.getElementById("kv-body");
const tokenizerContent = document.getElementById("tokenizer-content");
const tensorBody = document.getElementById("tensor-body");
const architectureContent = document.getElementById("architecture-content");
const histogramCanvasWrapper = document.querySelector(".histogram-canvas-wrapper");
const histogramCanvas = document.getElementById("histogram-canvas");
const histogramOverlay = document.getElementById("histogram-overlay");
const HISTOGRAM_CANVAS_HEIGHT = 512;
const STATISTICS_DEFAULT_HEADER_MESSAGE = "Select a tensor to generate its histogram.";
let statisticsHeaderMessage = STATISTICS_DEFAULT_HEADER_MESSAGE;
const histogramCtx = histogramCanvas ? histogramCanvas.getContext("2d") : null;
if (histogramCtx) {
    histogramCtx.imageSmoothingEnabled = false;
}

const heatmapCanvas = document.getElementById("heatmap-canvas");
const heatmapCanvasWrapper = document.querySelector(".heatmap-canvas-wrapper");
const heatmapTooltip = document.getElementById("heatmap-tooltip");
const heatmapStatus = document.getElementById("heatmap-status");
const statisticsStatus = document.getElementById("statistics-status");
const heatmapOverlay = document.getElementById("heatmap-overlay");
const heatmapControls = document.getElementById("heatmap-controls");
const heatmapTensorSelect = document.getElementById("heatmap-tensor-select");
const heatmapSliceInput = document.getElementById("heatmap-slice-input");
const heatmapMinInput = document.getElementById("heatmap-min-input");
const heatmapMaxInput = document.getElementById("heatmap-max-input");
const heatmapSliceButton = document.getElementById("heatmap-slice-button");
const heatmapP1Button = document.getElementById("heatmap-p1-button");
const heatmapP5Button = document.getElementById("heatmap-p5-button");
const heatmapP10Button = document.getElementById("heatmap-p10-button");
const heatmapStepInput = document.getElementById("heatmap-step-input");
const heatmapTightenButton = document.getElementById("heatmap-tighten-button");
const heatmapWidenButton = document.getElementById("heatmap-widen-button");
const heatmapGainOffsetToggle = document.getElementById("heatmap-gain-offset-toggle");
const heatmapGridToggle = document.getElementById("heatmap-grid-toggle");
const tabButtons = document.querySelectorAll(".tab-button");
const pageContainers = document.querySelectorAll(".page");
const modelBrowserContainer = document.getElementById("model-browser");
const modelBrowserRoot = document.getElementById("model-browser-root");
const modelTitleLabel = document.getElementById("model-title-label");
const tokenizerTitleLabel = document.getElementById("tokenizer-title-label");
const tensorsTitleLabel = document.getElementById("tensors-title-label");
const architectureTitleLabel = document.getElementById("architecture-title-label");
const heatmapTitleLabel = document.getElementById("heatmap-title-label");
const statisticsTitleLabel = document.getElementById("statistics-title-label");
const modelStatusIndicator = document.getElementById("model-status");
const tokenizerStatusIndicator = document.getElementById("tokenizer-status");
const tensorsStatusIndicator = document.getElementById("tensors-status");
const architectureStatusIndicator = document.getElementById("architecture-status");
const heatmapSectionStatus = document.getElementById("heatmap-section-status");
const statisticsSectionStatus = document.getElementById("statistics-section-status");
const sectionTitles = [
    { key: "model", label: "Model overview", labelElement: modelTitleLabel, statusElement: modelStatusIndicator },
    { key: "tokenizer", label: "Tokenizer", labelElement: tokenizerTitleLabel, statusElement: tokenizerStatusIndicator },
    { key: "tensors", label: "Tensors", labelElement: tensorsTitleLabel, statusElement: tensorsStatusIndicator },
    { key: "architecture", label: "Architecture", labelElement: architectureTitleLabel, statusElement: architectureStatusIndicator },
    { key: "heatmap", label: "Heatmap", labelElement: heatmapTitleLabel, statusElement: heatmapSectionStatus },
    { key: "statistics", label: "Statistics", labelElement: statisticsTitleLabel, statusElement: statisticsSectionStatus },
];
const sectionStatusMessages = new Map();
let currentModelDisplayPath = null;
let currentModelFullPath = null;
const heatmapCtx = heatmapCanvas.getContext("2d");
if (heatmapCtx) {
    heatmapCtx.imageSmoothingEnabled = false;
}

function readUrlState() {
    const params = new URLSearchParams(window.location.search);
    const getString = (key) => {
        const value = params.get(key);
        return value && value.length > 0 ? value : null;
    };
    const getInt = (key) => {
        const value = params.get(key);
        if (value === null) {
            return null;
        }
        const parsed = Number.parseInt(value, 10);
        return Number.isFinite(parsed) ? parsed : null;
    };
    const getFloat = (key) => {
        const value = params.get(key);
        if (value === null) {
            return null;
        }
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
    };
    const getBool = (key) => {
        const value = params.get(key);
        if (value === null) {
            return null;
        }
        const lowered = value.toLowerCase();
        if (value === "1" || lowered === "true") {
            return true;
        }
        if (value === "0" || lowered === "false") {
            return false;
        }
        return null;
    };
    return {
        model: getString("model"),
        tensor: getString("tensor"),
        slice: getInt("slice"),
        min: getFloat("min"),
        max: getFloat("max"),
        grid: getBool("grid"),
        x: getInt("x"),
        y: getInt("y"),
    };
}

function buildPendingHeatmapState(state) {
    return {
        tensor: state.tensor,
        slice: Number.isInteger(state.slice) ? Math.max(0, state.slice - 1) : null,
        min: Number.isFinite(state.min) ? state.min : null,
        max: Number.isFinite(state.max) ? state.max : null,
        grid: typeof state.grid === "boolean" ? state.grid : null,
        x: Number.isFinite(state.x) ? Math.max(0, state.x) : null,
        y: Number.isFinite(state.y) ? Math.max(0, state.y) : null,
        applied: false,
    };
}

let currentUrlState = readUrlState();
let pendingHeatmapState = buildPendingHeatmapState(currentUrlState);

function applyPendingHeatmapPreferences() {
    if (heatmapGridToggle && typeof pendingHeatmapState.grid === "boolean") {
        heatmapGridToggle.checked = pendingHeatmapState.grid;
    }
}

applyPendingHeatmapPreferences();

let urlSyncReady = false;
let currentModelPath = currentUrlState.model;
let backendSelectedModel = null;

async function syncStateFromSearchParams() {
    const nextState = readUrlState();
    const modelChanged = nextState.model !== currentModelPath;
    currentUrlState = nextState;
    pendingHeatmapState = buildPendingHeatmapState(nextState);
    applyPendingHeatmapPreferences();

    if (modelChanged) {
        backendSelectedModel = null;
        setCurrentModel(nextState.model, { updateBrowser: true, updateUrl: false });
        if (nextState.model) {
            pendingHeatmapState.applied = false;
            try {
                await refreshAll();
            } catch (err) {
                console.error(err);
            }
        } else {
            resetViewer();
        }
        return;
    }

    applyPendingHeatmapStateIfPossible();
}

function updateUrlParams(updates, options = {}) {
    const { force = false } = options;
    if (!force && !urlSyncReady) {
        return;
    }
    const params = new URLSearchParams(window.location.search);
    let changed = false;
    for (const [key, value] of Object.entries(updates)) {
        if (value === null || value === undefined || value === "") {
            if (params.has(key)) {
                params.delete(key);
                changed = true;
            }
        } else {
            const stringValue = String(value);
            if (params.get(key) !== stringValue) {
                params.set(key, stringValue);
                changed = true;
            }
        }
    }
    if (!changed && !force) {
        return;
    }
    const query = params.toString();
    const newUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`;
    window.history.replaceState(null, "", newUrl);
    currentUrlState = readUrlState();
}

function withModel(url, model = currentModelPath) {
    if (!model) {
        return url;
    }
    const [base, hashFragment] = url.split("#", 2);
    const separator = base.includes("?") ? "&" : "?";
    const next = `${base}${separator}model=${encodeURIComponent(model)}`;
    return hashFragment ? `${next}#${hashFragment}` : next;
}

function appendModelParam(params, model = currentModelPath) {
    if (!model) {
        return;
    }
    params.set("model", model);
}

function updateModelUrlState(options = {}) {
    const { force = false } = options;
    const value = typeof currentModelPath === "string" && currentModelPath.length > 0 ? currentModelPath : null;
    updateUrlParams({ model: value }, { force });
}

function updateHeatmapUrlState(options = {}) {
    const { force = false, syncPending = false } = options;
    if (!force && !urlSyncReady) {
        return;
    }
    const updates = {
        tensor: null,
        slice: null,
        min: null,
        max: null,
        grid: heatmapState.gridVisible ? "1" : "0",
        x: null,
        y: null,
    };
    if (currentModelPath && heatmapState.tensor) {
        updates.tensor = heatmapState.tensor.name || null;
        const sliceValue = Number.isInteger(heatmapState.slice) ? heatmapState.slice + 1 : null;
        updates.slice = sliceValue && sliceValue > 0 ? sliceValue : null;
        updates.min = Number.isFinite(heatmapState.scaleMin) ? heatmapState.scaleMin : null;
        updates.max = Number.isFinite(heatmapState.scaleMax) ? heatmapState.scaleMax : null;
        updates.grid = heatmapState.gridVisible ? "1" : "0";
        updates.x = Number.isFinite(heatmapState.windowX) ? Math.max(0, Math.floor(heatmapState.windowX)) : null;
        updates.y = Number.isFinite(heatmapState.windowY) ? Math.max(0, Math.floor(heatmapState.windowY)) : null;
    }
    updateUrlParams(updates, { force });
    if (!syncPending) {
        return;
    }
    if (currentModelPath && heatmapState.tensor) {
        pendingHeatmapState = {
            tensor: heatmapState.tensor.name || null,
            slice: Number.isInteger(heatmapState.slice) ? heatmapState.slice : null,
            min: Number.isFinite(heatmapState.scaleMin) ? heatmapState.scaleMin : null,
            max: Number.isFinite(heatmapState.scaleMax) ? heatmapState.scaleMax : null,
            grid: typeof heatmapState.gridVisible === "boolean" ? heatmapState.gridVisible : null,
            x: Number.isFinite(heatmapState.windowX) ? Math.max(0, Math.floor(heatmapState.windowX)) : null,
            y: Number.isFinite(heatmapState.windowY) ? Math.max(0, Math.floor(heatmapState.windowY)) : null,
            applied: true,
        };
    } else {
        pendingHeatmapState = {
            tensor: null,
            slice: null,
            min: null,
            max: null,
            grid: typeof heatmapState.gridVisible === "boolean" ? heatmapState.gridVisible : null,
            x: null,
            y: null,
            applied: true,
        };
    }
}

function syncFullUrlState(options = {}) {
    updateModelUrlState(options);
    updateHeatmapUrlState(options);
}

const tooltipIntegerFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const tooltipFloatFormatter = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 });

let activeTooltipHost = heatmapCanvasWrapper || null;
if (activeTooltipHost) {
    activeTooltipHost.classList.add("tooltip-host");
}

function setTooltipHost(host) {
    if (!heatmapTooltip) {
        activeTooltipHost = null;
        return;
    }
    if (!host) {
        activeTooltipHost = null;
        return;
    }
    if (!host.classList.contains("tooltip-host")) {
        host.classList.add("tooltip-host");
    }
    if (heatmapTooltip.parentElement !== host) {
        host.appendChild(heatmapTooltip);
    }
    activeTooltipHost = host;
}

function hideTooltipElement() {
    if (!heatmapTooltip) {
        return;
    }
    heatmapTooltip.hidden = true;
    heatmapTooltip.textContent = "";
}

function renderTooltipContent(titleText, rows = []) {
    if (!heatmapTooltip) {
        return;
    }
    heatmapTooltip.innerHTML = "";

    const normalizedRows = Array.isArray(rows)
        ? rows.filter((row) => row && row.value !== null && row.value !== undefined && row.value !== "")
        : [];

    if (!titleText && normalizedRows.length === 0) {
        hideTooltipElement();
        return;
    }

    if (titleText) {
        const title = document.createElement("div");
        title.className = "heatmap-tooltip__title";
        title.textContent = titleText;
        heatmapTooltip.appendChild(title);
    }

    if (normalizedRows.length > 0) {
        const grid = document.createElement("div");
        grid.className = "heatmap-tooltip__grid";

        normalizedRows.forEach((row) => {
            const labelEl = document.createElement("div");
            labelEl.className = "heatmap-tooltip__label";
            labelEl.textContent = row.label || "";

            const valueEl = document.createElement("div");
            let valueClass = "heatmap-tooltip__value";
            if (row.wrap) {
                valueClass += " heatmap-tooltip__value--wrap";
            }
            if (row.align === "left") {
                valueClass += " heatmap-tooltip__value--left";
            }
            valueEl.className = valueClass;
            valueEl.textContent = typeof row.value === "string" ? row.value : String(row.value);

            grid.appendChild(labelEl);
            grid.appendChild(valueEl);
        });

        heatmapTooltip.appendChild(grid);
    }

    heatmapTooltip.hidden = false;
}

function positionTooltipWithinHost(host, clientX, clientY) {
    if (!heatmapTooltip) {
        return;
    }
    const targetHost = host || activeTooltipHost;
    if (!targetHost) {
        return;
    }
    if (heatmapTooltip.hidden) {
        return;
    }
    const hostRect = targetHost.getBoundingClientRect();
    const tooltipRect = heatmapTooltip.getBoundingClientRect();
    const padding = 8;
    let left = clientX - hostRect.left + 16;
    let top = clientY - hostRect.top + 16;
    const width = tooltipRect.width || heatmapTooltip.offsetWidth;
    const height = tooltipRect.height || heatmapTooltip.offsetHeight;
    if (left + width + padding > hostRect.width) {
        left = Math.max(padding, hostRect.width - width - padding);
    }
    if (top + height + padding > hostRect.height) {
        top = Math.max(padding, hostRect.height - height - padding);
    }
    if (left < padding) {
        left = padding;
    }
    if (top < padding) {
        top = padding;
    }
    heatmapTooltip.style.left = `${left}px`;
    heatmapTooltip.style.top = `${top}px`;
}

const tokenTooltipState = {
    activeCell: null,
    host: null,
};

const DEFAULT_PAGE = "browser";
const validPages = new Set(Array.from(pageContainers, (page) => page.dataset.pageId));
let activePage = null;

function normalizePageId(pageId) {
    return validPages.has(pageId) ? pageId : DEFAULT_PAGE;
}

function renderActivePage(pageId) {
    const targetPage = normalizePageId(pageId);
    const previousPage = activePage;
    activePage = targetPage;

    pageContainers.forEach((page) => {
        const isActive = page.dataset.pageId === targetPage;
        page.classList.toggle("page--active", isActive);
    });
    tabButtons.forEach((button) => {
        const isActive = button.dataset.targetPage === targetPage;
        button.classList.toggle("tab-button--active", isActive);
        if (isActive) {
            button.setAttribute("aria-current", "page");
        } else {
            button.removeAttribute("aria-current");
        }
    });

    if (targetPage === "browser" && previousPage !== "browser") {
        void loadModelList();
    }

    if (previousPage === "tokenizer" && targetPage !== "tokenizer") {
        hideTokenTooltip();
    }

    const heatmapActive = targetPage === "heatmap";
    const heatmapWasActive = previousPage === "heatmap";
    if (heatmapActive) {
        syncHeatmapToViewport(!!heatmapState.tensor);
    } else if (heatmapWasActive) {
        syncHeatmapToViewport(false);
    }

    const statisticsActive = targetPage === "statistics";
    const statisticsWasActive = previousPage === "statistics";
    if (statisticsActive) {
        syncHistogramToViewport(!!heatmapState.tensor);
    } else if (statisticsWasActive) {
        syncHistogramToViewport(false);
    }

    if (heatmapActive) {
        applyPendingHeatmapStateIfPossible();
    }
}

function syncPageFromLocation() {
    const hash = window.location.hash.slice(1);
    const target = hash && hash.length > 0 ? hash : DEFAULT_PAGE;
    renderActivePage(target);
}

tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
        const target = button.dataset.targetPage;
        if (!target) {
            return;
        }
        const normalized = normalizePageId(target);
        if (window.location.hash.slice(1) !== normalized) {
            window.location.hash = normalized;
        } else {
            renderActivePage(normalized);
        }
    });
});

window.addEventListener("hashchange", () => {
    syncPageFromLocation();
});

window.addEventListener("popstate", () => {
    void syncStateFromSearchParams();
});

suppressNativeValidation(heatmapMinInput);
suppressNativeValidation(heatmapMaxInput);
suppressNativeValidation(heatmapStepInput);
const heatmapControlElements = [
    heatmapTensorSelect,
    heatmapSliceInput,
    heatmapMinInput,
    heatmapMaxInput,
    heatmapSliceButton,
    heatmapStepInput,
    heatmapTightenButton,
    heatmapWidenButton,
    heatmapGainOffsetToggle,
    heatmapGridToggle,
];
heatmapControlElements.forEach((element) => {
    if (element) {
        element.removeAttribute("title");
    }
});
let heatmapImageData = null;

const histogramState = {
    tensor: null,
    slice: 0,
    viewWidth: histogramCanvas ? histogramCanvas.width : 0,
    viewHeight: histogramCanvas ? histogramCanvas.height : 0,
    bins: [],
    maxCount: 0,
    total: 0,
    rangeMin: Number.NaN,
    rangeMax: Number.NaN,
    fetching: false,
    controller: null,
};

const HEATMAP_BACKGROUND = { r: 15, g: 23, b: 42 };
const HEATMAP_HUE_MAX = 270;
const HEATMAP_SCALE_EPSILON = 1e-5;
const HEATMAP_STEP_FALLBACK = 0.001;
const HEATMAP_MIN_STEP = 1e-6;
const HEATMAP_SCALE_HTML_STEP = 0.0001;
const HEATMAP_SCALE_DECIMALS = 4;
const VIEWPORT_MARGIN = 64;

let tensorData = [];
let modelMetadataEntries = [];
let architectureMetadataLoaded = false;
let architectureTensorsLoaded = false;
let tokenizerState = { offset: 0, total: 0, gridSize: 16 };
const textEncoder = new TextEncoder();
const TOKEN_TYPE_LABELS = Object.freeze({
    0: "normal",
    1: "unknown",
    2: "control",
    3: "user_defined",
    4: "unused",
    5: "byte",
});
const NO_MODEL_MESSAGE = "Select a GGUF model to inspect.";
const LOADING_MESSAGE = "Loading…";
const BROWSER_EMPTY_MESSAGE = "No GGUF files found in the root directory.";
const HEATMAP_DEFAULT_HEADER_MESSAGE = "Select a tensor to open the heatmap viewer.";
const HEATMAP_DEFAULT_STREAM_MESSAGE = "Select a tensor to stream its weights.";
let modelBrowserState = { root: "", items: [], selected: currentModelPath || null };
let selectingModel = false;

const heatmapState = {
    tensor: null,
    layout: { width: 1, height: 1, depth: 1 },
    windowX: 0,
    windowY: 0,
    slice: 0,
    viewMin: undefined,
    viewMax: undefined,
    sliceMin: undefined,
    sliceMax: undefined,
    scaleMin: -1,
    scaleMax: 1,
    scaleInitialized: false,
    scaleStep: HEATMAP_STEP_FALLBACK,
    offsetMode: false,
    blockSize: 1,
    gridVisible: typeof pendingHeatmapState.grid === "boolean" ? pendingHeatmapState.grid : true,
    valid: 0,
    viewWidth: heatmapCanvas ? heatmapCanvas.width : 0,
    viewHeight: heatmapCanvas ? heatmapCanvas.height : 0,
    controller: null,
    fetching: false,
    imageReady: false,
    values: [],
    pendingScale: null,
};

let heatmapHeaderMessage = HEATMAP_DEFAULT_HEADER_MESSAGE;

const heatmapHoverState = {
    canvasX: null,
    canvasY: null,
    globalX: null,
    globalY: null,
    slice: null,
    requestX: null,
    requestY: null,
    requestSlice: null,
    controller: null,
    data: null,
    clientX: null,
    clientY: null,
};

const heatmapDragState = {
    active: false,
    lastX: 0,
    lastY: 0,
    offsetX: 0,
    offsetY: 0,
    moved: false,
};

syncPageFromLocation();

function getSliceBounds() {
    let minBound = Number.isFinite(heatmapState.sliceMin) ? heatmapState.sliceMin : undefined;
    let maxBound = Number.isFinite(heatmapState.sliceMax) ? heatmapState.sliceMax : undefined;

    if (minBound !== undefined && maxBound !== undefined && minBound > maxBound) {
        const tmp = minBound;
        minBound = maxBound;
        maxBound = tmp;
    }

    return { minBound, maxBound };
}

function clampToBounds(value, minBound, maxBound) {
    let result = value;
    if (minBound !== undefined && result < minBound) {
        result = minBound;
    }
    if (maxBound !== undefined && result > maxBound) {
        result = maxBound;
    }
    return result;
}

function roundScaleValue(value) {
    if (!Number.isFinite(value)) {
        return value;
    }
    const factor = 1 / HEATMAP_SCALE_HTML_STEP;
    const rounded = Math.round(value * factor) / factor;
    return Object.is(rounded, -0) ? 0 : rounded;
}

function formatScaleInputValue(value) {
    if (!Number.isFinite(value)) {
        return "";
    }
    const rounded = roundScaleValue(value);
    if (Object.is(rounded, -0)) {
        return "0";
    }
    const fixed = rounded.toFixed(HEATMAP_SCALE_DECIMALS);
    return fixed
        .replace(/(\.\d*?[1-9])0+$/, "$1")
        .replace(/\.0+$/, "");
}

function suppressNativeValidation(input) {
    if (!input) {
        return;
    }
    input.addEventListener("invalid", (event) => {
        event.preventDefault();
    });
}

function sanitizeScale(minValue, maxValue) {
    const { minBound, maxBound } = getSliceBounds();

    let min = Number.isFinite(minValue) ? minValue : undefined;
    if (!Number.isFinite(min)) {
        min = Number.isFinite(heatmapState.scaleMin) ? heatmapState.scaleMin : undefined;
    }
    if (!Number.isFinite(min)) {
        min = Number.isFinite(heatmapState.viewMin) ? heatmapState.viewMin : undefined;
    }
    if (!Number.isFinite(min)) {
        min = Number.isFinite(heatmapState.sliceMin) ? heatmapState.sliceMin : undefined;
    }
    if (!Number.isFinite(min)) {
        min = minBound;
    }
    if (!Number.isFinite(min)) {
        min = -1;
    }

    let max = Number.isFinite(maxValue) ? maxValue : undefined;
    if (!Number.isFinite(max)) {
        max = Number.isFinite(heatmapState.scaleMax) ? heatmapState.scaleMax : undefined;
    }
    if (!Number.isFinite(max)) {
        max = Number.isFinite(heatmapState.viewMax) ? heatmapState.viewMax : undefined;
    }
    if (!Number.isFinite(max)) {
        max = Number.isFinite(heatmapState.sliceMax) ? heatmapState.sliceMax : undefined;
    }
    if (!Number.isFinite(max)) {
        max = maxBound;
    }
    if (!Number.isFinite(max)) {
        max = 1;
    }

    min = clampToBounds(min, minBound, maxBound);
    max = clampToBounds(max, minBound, maxBound);

    if (!Number.isFinite(min)) {
        min = -1;
    }
    if (!Number.isFinite(max)) {
        max = 1;
    }

    return { min, max };
}

function setHeatmapScale(minValue, maxValue, options = {}) {
    const { reapply = true, sync = true } = options;
    const { min, max } = sanitizeScale(minValue, maxValue);
    const roundedMin = roundScaleValue(min);
    const roundedMax = roundScaleValue(max);
    if (heatmapState.scaleInitialized) {
        const prevMin = Number.isFinite(heatmapState.scaleMin) ? heatmapState.scaleMin : -1;
        const prevMax = Number.isFinite(heatmapState.scaleMax) ? heatmapState.scaleMax : 1;
        if (Math.abs(prevMin - roundedMin) < HEATMAP_SCALE_EPSILON && Math.abs(prevMax - roundedMax) < HEATMAP_SCALE_EPSILON) {
            if (sync) {
                updateHeatmapHeader();
                syncHeatmapControls();
            }
            return;
        }
    }
    heatmapState.scaleMin = roundedMin;
    heatmapState.scaleMax = roundedMax;
    heatmapState.scaleInitialized = true;

    if (reapply && heatmapState.imageReady && Array.isArray(heatmapState.values) && heatmapState.values.length > 0) {
        updateHeatmapImage(heatmapState.values, heatmapState.scaleMin, heatmapState.scaleMax);
    }

    if (sync) {
        updateHeatmapHeader();
        syncHeatmapControls();
    }
    updateHeatmapUrlState({ syncPending: true });
}

function percentile(sortedValues, percentile) {
    if (!Array.isArray(sortedValues) || sortedValues.length === 0) {
        return null;
    }
    if (sortedValues.length === 1) {
        return sortedValues[0];
    }
    const clampedPercentile = Math.min(100, Math.max(0, percentile));
    const rank = (clampedPercentile / 100) * (sortedValues.length - 1);
    const lowerIndex = Math.floor(rank);
    const upperIndex = Math.ceil(rank);
    if (lowerIndex === upperIndex) {
        return sortedValues[lowerIndex];
    }
    const weight = rank - lowerIndex;
    const lowerValue = sortedValues[lowerIndex];
    const upperValue = sortedValues[upperIndex];
    return lowerValue + (upperValue - lowerValue) * weight;
}

function getHeatmapPercentileRange(lowerPercent) {
    if (!Array.isArray(heatmapState.values) || heatmapState.values.length === 0) {
        return null;
    }
    const finiteValues = heatmapState.values.filter((value) => typeof value === "number" && Number.isFinite(value));
    if (finiteValues.length === 0) {
        return null;
    }
    const sortedValues = finiteValues.slice().sort((a, b) => a - b);
    const lower = percentile(sortedValues, lowerPercent);
    const upper = percentile(sortedValues, 100 - lowerPercent);
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
        return null;
    }
    return { min: lower, max: upper };
}

function applyHeatmapPercentileRange(lowerPercent) {
    if (!heatmapState.tensor || !heatmapState.imageReady) {
        return;
    }
    const range = getHeatmapPercentileRange(lowerPercent);
    if (!range) {
        return;
    }
    setHeatmapScale(range.min, range.max);
}

function sanitizeStepValue(value) {
    if (!Number.isFinite(value) || value <= 0) {
        return HEATMAP_STEP_FALLBACK;
    }
    return Math.max(HEATMAP_MIN_STEP, Math.abs(value));
}

function ensureHeatmapStep() {
    const next = sanitizeStepValue(heatmapState.scaleStep);
    heatmapState.scaleStep = next;
    return next;
}

function commitHeatmapStep(rawValue) {
    const parsed = Number.parseFloat(rawValue);
    const next = Number.isFinite(parsed) ? sanitizeStepValue(parsed) : ensureHeatmapStep();
    heatmapState.scaleStep = next;
    if (heatmapStepInput) {
        heatmapStepInput.value = String(next);
    }
    return next;
}

function adjustHeatmapContrast(direction, magnitude = 1) {
    if (!heatmapState.tensor || !heatmapState.imageReady) {
        return;
    }
    if (direction === 0) {
        return;
    }
    const step = ensureHeatmapStep();
    const appliedMagnitude = Math.max(1, Math.floor(Math.abs(magnitude)));
    const delta = step * appliedMagnitude;
    let nextMin = heatmapState.scaleMin;
    let nextMax = heatmapState.scaleMax;
    if (heatmapState.offsetMode) {
        const offsetDelta = direction > 0 ? delta : -delta;
        nextMin += offsetDelta;
        nextMax += offsetDelta;
    } else if (direction > 0) {
        nextMin += delta;
        nextMax -= delta;
    } else {
        nextMin -= delta;
        nextMax += delta;
    }
    setHeatmapScale(nextMin, nextMax);
}

function syncHeatmapControls() {
    if (!heatmapControls) {
        return;
    }

    const hasTensor = !!heatmapState.tensor;
    heatmapControls.hidden = !hasTensor;

    if (heatmapTensorSelect) {
        const selected = hasTensor && heatmapState.tensor
            ? encodeURIComponent(heatmapState.tensor.name)
            : "";
        if (heatmapTensorSelect.value !== selected) {
            heatmapTensorSelect.value = selected;
        }
        heatmapTensorSelect.disabled = tensorData.length === 0;
    }

    if (heatmapStepInput) {
        const step = ensureHeatmapStep();
        heatmapStepInput.value = String(step);
        heatmapStepInput.disabled = !hasTensor;
    }

    if (heatmapTightenButton) {
        heatmapTightenButton.disabled = !(hasTensor && heatmapState.imageReady);
    }
    if (heatmapWidenButton) {
        heatmapWidenButton.disabled = !(hasTensor && heatmapState.imageReady);
    }
    if (heatmapGainOffsetToggle) {
        heatmapGainOffsetToggle.checked = !!heatmapState.offsetMode;
        heatmapGainOffsetToggle.disabled = !hasTensor;
    }
    if (heatmapGridToggle) {
        const gridStep = getHeatmapGridStep(heatmapState.blockSize);
        const canShowGrid = hasTensor && gridStep > 1;
        heatmapGridToggle.disabled = !canShowGrid;
        heatmapGridToggle.checked = canShowGrid && !!heatmapState.gridVisible;
    }

    if (!hasTensor) {
        if (heatmapSliceInput) {
            heatmapSliceInput.value = "1";
            heatmapSliceInput.disabled = true;
        }
        if (heatmapMinInput) {
            heatmapMinInput.value = formatScaleInputValue(heatmapState.scaleMin);
            heatmapMinInput.disabled = true;
        }
        if (heatmapMaxInput) {
            heatmapMaxInput.value = formatScaleInputValue(heatmapState.scaleMax);
            heatmapMaxInput.disabled = true;
        }
        return;
    }

    const depth = heatmapState.layout && typeof heatmapState.layout.depth === "number" && heatmapState.layout.depth > 0
        ? heatmapState.layout.depth
        : 1;

    if (heatmapSliceInput) {
        heatmapSliceInput.min = "1";
        heatmapSliceInput.max = String(depth);
        heatmapSliceInput.value = String(Math.min(depth, heatmapState.slice + 1));
        heatmapSliceInput.disabled = depth <= 1;
    }

    const ready = heatmapState.imageReady;

    if (heatmapMinInput) {
        heatmapMinInput.value = formatScaleInputValue(heatmapState.scaleMin);
        heatmapMinInput.disabled = !ready;
    }

    if (heatmapMaxInput) {
        heatmapMaxInput.value = formatScaleInputValue(heatmapState.scaleMax);
        heatmapMaxInput.disabled = !ready;
    }

    if (heatmapSliceButton) {
        const hasSliceScale = Number.isFinite(heatmapState.sliceMin) && Number.isFinite(heatmapState.sliceMax);
        heatmapSliceButton.disabled = !ready || !hasSliceScale;
    }

    const hasPercentiles = ready && heatmapState.valid > 0;
    if (heatmapP1Button) {
        heatmapP1Button.disabled = !hasPercentiles;
    }
    if (heatmapP5Button) {
        heatmapP5Button.disabled = !hasPercentiles;
    }
    if (heatmapP10Button) {
        heatmapP10Button.disabled = !hasPercentiles;
    }
}

function getViewportDimensions() {
    const doc = document.documentElement;
    const width = Math.max(1, Math.floor(doc ? doc.clientWidth : window.innerWidth || 1));
    const height = Math.max(1, Math.floor(window.innerHeight || (doc ? doc.clientHeight : 1)));
    return { width, height };
}

function hslToRgb(h, s, l) {
    const hue = ((h % 360) + 360) % 360;
    const chroma = (1 - Math.abs(2 * l - 1)) * s;
    const hp = hue / 60;
    const x = chroma * (1 - Math.abs((hp % 2) - 1));
    let r1 = 0;
    let g1 = 0;
    let b1 = 0;

    if (hp >= 0 && hp < 1) {
        r1 = chroma;
        g1 = x;
    } else if (hp >= 1 && hp < 2) {
        r1 = x;
        g1 = chroma;
    } else if (hp >= 2 && hp < 3) {
        g1 = chroma;
        b1 = x;
    } else if (hp >= 3 && hp < 4) {
        g1 = x;
        b1 = chroma;
    } else if (hp >= 4 && hp < 5) {
        r1 = x;
        b1 = chroma;
    } else {
        r1 = chroma;
        b1 = x;
    }

    const m = l - chroma / 2;
    return {
        r: Math.round((r1 + m) * 255),
        g: Math.round((g1 + m) * 255),
        b: Math.round((b1 + m) * 255),
    };
}

function renderEmptyNote(target, message) {
    if (!target) {
        return;
    }
    target.innerHTML = "";
    const note = document.createElement("div");
    note.className = "empty-note";
    note.textContent = message;
    target.appendChild(note);
}

function renderTableMessage(body, span, message) {
    if (!body) {
        return;
    }
    const row = document.createElement("tr");
    row.className = "table-empty";
    const cell = document.createElement("td");
    cell.colSpan = span;
    cell.textContent = message;
    row.appendChild(cell);
    body.appendChild(row);
}

function updateSectionHeaderPaths(displayPath, fullPath) {
    const resolvedDisplay = typeof displayPath === "string" && displayPath.length > 0 ? displayPath : null;
    const resolvedFull = typeof fullPath === "string" && fullPath.length > 0 ? fullPath : null;
    sectionTitles.forEach(({ key, label, labelElement, statusElement }) => {
        if (labelElement) {
            labelElement.textContent = label;
        }
        if (!statusElement) {
            return;
        }
        const message = sectionStatusMessages.get(key);
        statusElement.classList.remove("model-status--message");
        statusElement.innerHTML = "";
        if (typeof message === "string" && message.length > 0) {
            statusElement.textContent = message;
            statusElement.classList.add("model-status--message");
            statusElement.hidden = false;
            return;
        }
        if (!resolvedDisplay) {
            statusElement.hidden = true;
            return;
        }
        const pathBadge = document.createElement("span");
        pathBadge.className = "model-path";
        if (resolvedFull) {
            pathBadge.title = resolvedFull;
        }
        const pathCode = document.createElement("code");
        pathCode.textContent = resolvedDisplay;
        pathBadge.appendChild(pathCode);
        statusElement.appendChild(pathBadge);
        statusElement.hidden = false;
    });
}

function setSectionStatus(key, message) {
    if (!key) {
        return;
    }
    if (typeof message === "string" && message.length > 0) {
        sectionStatusMessages.set(key, message);
    } else {
        sectionStatusMessages.delete(key);
    }
    updateSectionHeaderPaths(currentModelDisplayPath, currentModelFullPath);
}

async function fetchJSON(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
        const message = await res.text();
        const error = new Error(message || `Request failed: ${res.status}`);
        error.status = res.status;
        throw error;
    }
    if (res.status === 204) {
        return null;
    }
    try {
        return await res.json();
    } catch (err) {
        const error = new Error(`Failed to parse response from ${url}: ${err.message}`);
        error.status = res.status;
        throw error;
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return "0";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / Math.pow(1024, idx);
    return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[idx]}`;
}

function formatTooltipInteger(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return String(value);
    }
    return tooltipIntegerFormatter.format(value);
}

function formatTooltipFloat(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return String(value);
    }
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) {
        return value.toExponential(4);
    }
    return tooltipFloatFormatter.format(value);
}

