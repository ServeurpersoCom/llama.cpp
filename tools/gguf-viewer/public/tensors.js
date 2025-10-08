function setTensorsMessage(message) {
    if (!tensorBody) {
        return;
    }
    tensorBody.innerHTML = "";
    renderTableMessage(tensorBody, 6, message);
    setSectionStatus("tensors", message);
}

function renderTensorRow(tensor) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
        <td><code>${tensor.name}</code></td>
        <td>${tensor.type}</td>
        <td><code>${formatShape(tensor.shape)}</code></td>
        <td>${tensor.nElements}</td>
        <td>${tensor.fileOffset}</td>
        <td><button data-name="${encodeURIComponent(tensor.name)}">Heatmap</button></td>
    `;
    return tr;
}

function formatHeatmapTensorLabel(tensor) {
    if (!tensor || typeof tensor !== "object") {
        return "";
    }
    const parts = [];
    if (tensor.name) {
        parts.push(tensor.name);
    }
    if (tensor.type) {
        parts.push(tensor.type);
    }
    parts.push(formatShape(tensor.shape));
    return parts.filter(Boolean).join(" ");
}

function refreshHeatmapTensorSelect() {
    if (!heatmapTensorSelect) {
        return;
    }

    const fragment = document.createDocumentFragment();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = tensorData.length > 0 ? "Select tensor…" : "No tensors available";
    placeholder.disabled = tensorData.length > 0;
    placeholder.selected = !heatmapState.tensor;
    fragment.appendChild(placeholder);

    tensorData.forEach((tensor) => {
        const option = document.createElement("option");
        const encodedName = encodeURIComponent(tensor.name);
        option.value = encodedName;
        option.textContent = formatHeatmapTensorLabel(tensor);
        if (heatmapState.tensor && tensor.name === heatmapState.tensor.name) {
            option.selected = true;
        }
        fragment.appendChild(option);
    });

    heatmapTensorSelect.innerHTML = "";
    heatmapTensorSelect.appendChild(fragment);
    if (heatmapState.tensor) {
        heatmapTensorSelect.value = encodeURIComponent(heatmapState.tensor.name);
    }
    heatmapTensorSelect.disabled = tensorData.length === 0;
}

function applyPendingHeatmapStateIfPossible() {
    if (pendingHeatmapState.applied) {
        return;
    }
    if (normalizePageId(activePage || DEFAULT_PAGE) !== "heatmap") {
        return;
    }
    if (!Array.isArray(tensorData) || tensorData.length === 0) {
        return;
    }

    pendingHeatmapState.applied = true;

    if (!pendingHeatmapState.tensor) {
        return;
    }

    const tensor = tensorData.find((item) => item.name === pendingHeatmapState.tensor);
    if (!tensor) {
        return;
    }

    const options = {};
    if (Number.isInteger(pendingHeatmapState.slice) && pendingHeatmapState.slice >= 0) {
        options.initialSlice = pendingHeatmapState.slice;
    }

    const windowOptions = {};
    if (Number.isInteger(pendingHeatmapState.x) && pendingHeatmapState.x >= 0) {
        windowOptions.x = pendingHeatmapState.x;
    }
    if (Number.isInteger(pendingHeatmapState.y) && pendingHeatmapState.y >= 0) {
        windowOptions.y = pendingHeatmapState.y;
    }
    if (Object.keys(windowOptions).length > 0) {
        options.initialWindow = windowOptions;
    }

    if (Number.isFinite(pendingHeatmapState.min) && Number.isFinite(pendingHeatmapState.max)) {
        options.initialScale = { min: pendingHeatmapState.min, max: pendingHeatmapState.max };
    }

    if (typeof pendingHeatmapState.grid === "boolean") {
        options.initialGrid = pendingHeatmapState.grid;
    }

    openHeatmap(encodeURIComponent(tensor.name), options);
}

async function loadTensors() {
    try {
        const data = await fetchJSON(withModel("api/tensors"));
        tensorData = Array.isArray(data) ? data : [];
        architectureTensorsLoaded = true;
        if (tensorData.length === 0) {
            setTensorsMessage("No tensors available.");
            setArchitectureMessage("No tensors available.");
        } else {
            tensorBody.innerHTML = "";
            setSectionStatus("tensors", null);
            tensorData.forEach(tensor => tensorBody.appendChild(renderTensorRow(tensor)));
            tryRenderArchitecture();
        }
        refreshHeatmapTensorSelect();
        applyPendingHeatmapStateIfPossible();
    } catch (err) {
        tensorData = [];
        architectureTensorsLoaded = false;
        setTensorsMessage(err.status === 409 ? NO_MODEL_MESSAGE : err.message);
        setArchitectureMessage(err.status === 409 ? NO_MODEL_MESSAGE : err.message);
        refreshHeatmapTensorSelect();
        pendingHeatmapState.applied = true;
    }
}

