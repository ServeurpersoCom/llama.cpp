function setInfoMessage(message) {
    setSectionStatus("model", message);
    renderEmptyNote(infoCards, message);
}

function setArchitectureMessage(message) {
    if (!architectureContent) {
        return;
    }
    if (typeof message === "string" && message.length > 0) {
        renderEmptyNote(architectureContent, message);
    } else {
        architectureContent.innerHTML = "";
    }
    setSectionStatus("architecture", message);
}

function tryRenderArchitecture() {
    if (!architectureContent) {
        return;
    }
    if (!architectureMetadataLoaded || !architectureTensorsLoaded) {
        return;
    }
    if (!Array.isArray(modelMetadataEntries) || modelMetadataEntries.length === 0) {
        setArchitectureMessage("No metadata entries available.");
        return;
    }
    if (!Array.isArray(tensorData) || tensorData.length === 0) {
        setArchitectureMessage("No tensors available.");
        return;
    }
    const module = window.llamaViewerArchitecture;
    if (!module || typeof module.render !== "function") {
        setArchitectureMessage("Architecture module unavailable.");
        return;
    }
    architectureContent.innerHTML = "";
    const result = module.render({
        metadata: modelMetadataEntries,
        tensors: tensorData,
        target: architectureContent,
    });
    const statusMessage = result && typeof result.statusMessage === "string" && result.statusMessage.length > 0
        ? result.statusMessage
        : null;
    setSectionStatus("architecture", statusMessage);
}

function createInfoCard(title, value) {
    const div = document.createElement("div");
    div.className = "info-card";
    const heading = document.createElement("h3");
    heading.textContent = title;
    const para = document.createElement("p");
    para.textContent = value;
    div.appendChild(heading);
    div.appendChild(para);
    return div;
}

function resetViewer(message = NO_MODEL_MESSAGE) {
    currentModelDisplayPath = null;
    currentModelFullPath = null;
    setInfoMessage(message);
    kvBody.innerHTML = "";
    renderTableMessage(kvBody, 3, message);
    setTokenizerMessage(message);
    setTensorsMessage(message);
    setArchitectureMessage(message);
    tensorData = [];
    modelMetadataEntries = [];
    architectureMetadataLoaded = false;
    architectureTensorsLoaded = false;
    tokenizerState = { ...tokenizerState, offset: 0, total: 0 };
    refreshHeatmapTensorSelect();
    resetHistogram(message);
    resetHeatmap(message);
}

async function refreshAll() {
    if (!currentModelPath) {
        resetViewer();
        return { ok: false, message: NO_MODEL_MESSAGE };
    }

    currentModelDisplayPath = null;
    currentModelFullPath = null;
    setInfoMessage(LOADING_MESSAGE);
    kvBody.innerHTML = "";
    renderTableMessage(kvBody, 3, LOADING_MESSAGE);
    setTokenizerMessage(LOADING_MESSAGE);
    setTensorsMessage(LOADING_MESSAGE);
    setArchitectureMessage(LOADING_MESSAGE);
    tokenizerState = { ...tokenizerState, offset: 0, total: 0 };
    modelMetadataEntries = [];
    architectureMetadataLoaded = false;
    architectureTensorsLoaded = false;
    resetHeatmap();

    const infoResult = await loadInfo();
    if (!infoResult.ok) {
        const message = infoResult.message || NO_MODEL_MESSAGE;
        setTokenizerMessage(message);
        setTensorsMessage(message);
        setArchitectureMessage(message);
        tokenizerState = { ...tokenizerState, offset: 0, total: 0 };
        tensorData = [];
        modelMetadataEntries = [];
        architectureMetadataLoaded = false;
        architectureTensorsLoaded = false;
        refreshHeatmapTensorSelect();
        resetHistogram(message);
        resetHeatmap(message);
        pendingHeatmapState.applied = true;
        pendingHeatmapState.tensor = null;
        pendingHeatmapState.slice = null;
        pendingHeatmapState.min = null;
        pendingHeatmapState.max = null;
        pendingHeatmapState.x = null;
        pendingHeatmapState.y = null;
        if (typeof heatmapState.gridVisible === "boolean") {
            pendingHeatmapState.grid = heatmapState.gridVisible;
        }
        updateModelUrlState();
        return { ok: false, message };
    }

    updateModelUrlState();
    await loadKv();
    await loadTokenizer(0);
    await loadTensors();
    return { ok: true };
}

async function loadInfo() {
    try {
        const info = await fetchJSON(withModel("api/info"));
        infoCards.innerHTML = "";

        if (info && typeof info === "object") {
            const canonicalPath = typeof info.relativePath === "string" && info.relativePath.length > 0
                ? info.relativePath
                : null;
            if (canonicalPath && canonicalPath !== currentModelPath) {
                setCurrentModel(canonicalPath, { updateBrowser: true });
            }
            const displayPath = canonicalPath && canonicalPath.length > 0
                ? canonicalPath
                : info.modelPath;
            currentModelDisplayPath = displayPath && displayPath.length > 0 ? displayPath : null;
            currentModelFullPath = typeof info.modelPath === "string" && info.modelPath.length > 0
                ? info.modelPath
                : null;
            setSectionStatus("model", null);

            const cards = [
                ["File size", Number.isFinite(info.fileSize) ? formatBytes(info.fileSize) : "N/A"],
                ["GGUF version", info.ggufVersion ?? "N/A"],
                ["KV entries", info.nKv ?? "N/A"],
                ["Tensors", info.nTensors ?? "N/A"],
                ["Alignment", Number.isFinite(info.alignment) ? `${info.alignment} bytes` : "N/A"],
                ["Data offset", Number.isFinite(info.dataOffset) ? String(info.dataOffset) : "N/A"],
            ];

            cards.forEach(([title, value]) => {
                infoCards.appendChild(createInfoCard(title, value));
            });
            return { ok: true };
        }

        currentModelDisplayPath = null;
        currentModelFullPath = null;
        const message = "Model info unavailable.";
        setInfoMessage(message);
        return { ok: false, message };
    } catch (err) {
        currentModelDisplayPath = null;
        currentModelFullPath = null;
        const message = isModelUnavailableError(err) ? NO_MODEL_MESSAGE : err.message;
        setInfoMessage(message);
        return { ok: false, message };
    }
}

function renderKvRow(entry) {
    const tr = document.createElement("tr");
    const typeText = entry.type === "GGUF_TYPE_ARRAY"
        ? `${entry.type}<br/><span class="badge">${entry.arrayType} x ${entry.length}</span>`
        : entry.type;
    tr.innerHTML = `
        <td><code>${entry.key}</code></td>
        <td>${typeText}</td>
        <td class="kv-value"></td>
    `;
    const details = tr.querySelector(".kv-value");
    if (entry.type === "GGUF_TYPE_ARRAY") {
        const preview = Array.isArray(entry.preview) ? entry.preview : [];
        details.textContent = preview.length ? JSON.stringify(preview, null, 2) : "[]";
        if (entry.previewTruncated) {
            const span = document.createElement("span");
            span.className = "badge";
            span.textContent = `first ${preview.length} of ${entry.length}`;
            details.appendChild(document.createElement("br"));
            details.appendChild(span);
        }
    } else {
        details.textContent = entry.value === null ? "null" : entry.value;
    }
    return tr;
}

async function loadKv() {
    try {
        const entries = await fetchJSON(withModel("api/kv"));
        kvBody.innerHTML = "";
        modelMetadataEntries = Array.isArray(entries) ? entries : [];
        architectureMetadataLoaded = true;
        if (!Array.isArray(entries) || entries.length === 0) {
            renderTableMessage(kvBody, 3, "No metadata entries found.");
            setArchitectureMessage("No metadata entries available.");
            return;
        }
        entries.forEach(entry => kvBody.appendChild(renderKvRow(entry)));
        tryRenderArchitecture();
    } catch (err) {
        kvBody.innerHTML = "";
        modelMetadataEntries = [];
        architectureMetadataLoaded = false;
        const message = isModelUnavailableError(err) ? NO_MODEL_MESSAGE : err.message;
        renderTableMessage(kvBody, 3, message);
        setArchitectureMessage(message);
    }
}

