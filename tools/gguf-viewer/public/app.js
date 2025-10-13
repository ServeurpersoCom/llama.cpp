window.addEventListener("resize", () => {
    syncHeatmapToViewport(true);
    const statisticsVisible = normalizePageId(activePage || DEFAULT_PAGE) === "statistics";
    syncHistogramToViewport(statisticsVisible && !!heatmapState.tensor);
});

syncHeatmapToViewport(false);
resetViewer();

(async function init() {
    await loadModelList();
    if (currentModelPath) {
        await refreshAll();
    } else {
        resetViewer();
    }
    urlSyncReady = true;
    syncFullUrlState({ force: true });
})();
