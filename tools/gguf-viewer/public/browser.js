async function requestModelSelection(path) {
    if (!path) {
        throw new Error("Model path is required");
    }
    const response = await fetchJSON(`api/models/select?model=${encodeURIComponent(path)}`, { method: "POST" });
    const selected = typeof response?.selected === "string" && response.selected.length > 0
        ? response.selected
        : path;
    backendSelectedModel = selected;
    return selected;
}

function setCurrentModel(path, options = {}) {
    const { updateBrowser = true, updateUrl = true } = options;
    const normalized = typeof path === "string" && path.length > 0 ? path : null;
    currentModelPath = normalized;
    if (!normalized) {
        backendSelectedModel = null;
    }
    const items = Array.isArray(modelBrowserState.items) ? modelBrowserState.items : [];
    const updatedItems = items.map((item) => ({ ...item, selected: !!normalized && item.path === normalized }));
    modelBrowserState = { ...modelBrowserState, selected: normalized, items: updatedItems };
    if (updateBrowser) {
        renderModelBrowser();
    }
    if (updateUrl) {
        updateModelUrlState();
    }
}

function renderModelBrowser() {
    if (!modelBrowserContainer) {
        return;
    }

    modelBrowserContainer.innerHTML = "";

    if (modelBrowserRoot) {
        modelBrowserRoot.innerHTML = "";
        const hasRoot = typeof modelBrowserState.root === "string" && modelBrowserState.root.length > 0;
        modelBrowserRoot.hidden = !hasRoot;
        if (hasRoot) {
            const code = document.createElement("code");
            code.textContent = modelBrowserState.root;
            modelBrowserRoot.appendChild(code);
        }
    }

    const items = Array.isArray(modelBrowserState.items) ? modelBrowserState.items : [];
    if (items.length === 0) {
        renderEmptyNote(modelBrowserContainer, BROWSER_EMPTY_MESSAGE);
        return;
    }

    const list = document.createElement("ul");
    list.className = "model-list";

    items.forEach((item) => {
        const li = document.createElement("li");
        li.className = "model-item";

        const button = document.createElement("button");
        button.type = "button";
        button.className = "model-button";
        const isSelected = !!item.selected;
        button.dataset.selected = isSelected ? "true" : "false";
        button.disabled = selectingModel;

        const pathLabel = document.createElement("span");
        pathLabel.className = "model-path-entry";
        pathLabel.textContent = item.path;

        const size = document.createElement("span");
        size.className = "model-meta";
        size.textContent = Number.isFinite(item.size) ? formatBytes(item.size) : "N/A";

        button.appendChild(pathLabel);
        button.appendChild(size);

        button.addEventListener("click", () => {
            if (!selectingModel) {
                selectModel(item.path);
            }
        });

        li.appendChild(button);
        list.appendChild(li);
    });

    modelBrowserContainer.appendChild(list);
}

async function loadModelList() {
    if (modelBrowserContainer) {
        renderEmptyNote(modelBrowserContainer, LOADING_MESSAGE);
    }

    try {
        const data = await fetchJSON(withModel("api/models"));
        const rawItems = Array.isArray(data?.items) ? data.items : [];
        const rootPath = typeof data?.root === "string" ? data.root : "";
        let selected = typeof data?.selected === "string" && data.selected.length > 0 ? data.selected : null;
        if (currentModelPath && currentModelPath.length > 0) {
            selected = currentModelPath;
        }
        const items = rawItems.map((item) => ({
            path: item.path,
            size: item.size,
        }));
        if (selected && !items.some((item) => item.path === selected)) {
            selected = null;
        }
        modelBrowserState = {
            root: rootPath,
            items: items.map((item) => ({ ...item, selected: !!selected && item.path === selected })),
            selected,
        };
        currentModelPath = selected;
        updateModelUrlState();
        renderModelBrowser();
    } catch (err) {
        modelBrowserState = { root: "", items: [], selected: null };
        if (modelBrowserRoot) {
            modelBrowserRoot.textContent = "";
            modelBrowserRoot.hidden = true;
        }
        if (modelBrowserContainer) {
            renderEmptyNote(modelBrowserContainer, err.message);
        }
    }
}

async function selectModel(path) {
    if (!path || selectingModel) {
        return;
    }
    if (currentModelPath && currentModelPath === path) {
        return;
    }

    selectingModel = true;
    let errorMessage = null;
    try {
        const selectedPath = await requestModelSelection(path);
        setCurrentModel(selectedPath, { updateBrowser: false });
        pendingHeatmapState.applied = true;
        pendingHeatmapState.tensor = null;
        pendingHeatmapState.slice = null;
        pendingHeatmapState.min = null;
        pendingHeatmapState.max = null;
        pendingHeatmapState.x = null;
        pendingHeatmapState.y = null;
        await refreshAll({ ensureSelected: false });
    } catch (err) {
        errorMessage = err.status === 409 ? NO_MODEL_MESSAGE : err.message;
    } finally {
        selectingModel = false;
        renderModelBrowser();
        if (errorMessage && modelBrowserContainer) {
            const note = document.createElement("div");
            note.className = "empty-note";
            note.textContent = errorMessage;
            modelBrowserContainer.appendChild(note);
        }
    }
}
