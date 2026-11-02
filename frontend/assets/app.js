const RECENT_LIMIT = 20;
const PARAM_LABELS = {
  kpca_dim: "K",
  tree_size: "M",
  tree_num: "q",
  para: "area",
  n_dim: "PCA",
};
const PARAM_HINTS = {
  kpca_dim: "KPCA components",
  tree_size: "3% of rows × columns",
  tree_num: "trees",
  para: "attribute threshold",
  n_dim: "components",
};

function paramLabel(key) {
  return PARAM_LABELS[key] || key;
}

const DETECTOR_NOTES = {
  kifd: "M is how many pixels each isolation tree picks at random. Auto means 3% of the image (rows × columns). Example: a 100×100 scene uses M = 300. Type a number to use your own value.",
  aed: "area (para) is the attribute-filter threshold from the original demo.m. Default 25; the demo uses 64 or 5 on a few scenes. PCA components default to 3. Gaussian kernel is not used here — AED is an attribute + edge-preserving filter.",
};

function paramDefaultText(key, value) {
  if (key === "tree_size" && (value === "auto" || value === "" || value === 0)) {
    return "auto — 3% of image pixels";
  }
  return value;
}

function defaultTreeSize() {
  const shape = state.session && state.session.shape;
  if (!shape) return "";
  return Math.max(2, Math.floor((3 * shape.height * shape.width) / 100));
}

const DR_HINTS = {
  none: "Use the original spectral feature space.",
  pca: "Project the cube onto principal components.",
  ica: "Recover statistically independent spectral sources.",
  svd: "Keep the leading singular vectors of the cube.",
  factor_analysis: "Model shared latent factors across bands.",
  isomap: "Preserve geodesic neighborhood structure.",
};
const state = {
  session: null,
  catalog: null,
  sceneView: "rgb",
  resultView: "heatmap",
  roc: null,
  spec: null,
  sessions: [],
  recentResults: [],
  detailId: null,
  threshold: 99,
  thresholdTimer: null,
  viewZoom: 1,
  viewSyncing: false,
  selectedDetector: null,
};

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

function setBusy(on, text) {
  $("pageBusy").classList.toggle("hidden", !on);
  $("busyText").textContent = text || "Working…";
}

async function api(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function showView(name) {
  document.querySelectorAll(".view").forEach((el) => el.classList.toggle("hidden", el.id !== `view-${name}`));
  document.querySelectorAll(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
  const labels = { workspace: "Analysis workspace", datasets: "Datasets", models: "Model registry", experiments: "Experiments" };
  $("crumbCurrent").textContent = labels[name] || name;
}

function setClearVisible(on) {
  ["clearBtn", "clearBtn2"].forEach((id) => $(id).classList.toggle("hidden", !on));
}

function clearRunReadouts() {
  $("detValue").textContent = "—";
  $("timeValue").textContent = "—";
  $("aucValue").textContent = "—";
  if ($("aucNote")) $("aucNote").textContent = "Requires reference mask";
  $("resultStatus").textContent = "Awaiting analysis";
  drawROC(null);
  if (state.session) state.session.last_result = null;
}

function resetAnalysisUI() {
  state.threshold = 99;
  resetViewers();
  state.resultView = "heatmap";
  state.sceneView = "rgb";
  document.querySelectorAll("#resultModes button").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === "heatmap"));
  document.querySelectorAll("#sceneModes button").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === "rgb"));
  clearRunReadouts();
  $("pixelReadout").textContent = "—";
  $("thresholdInfo").textContent = "Adjust the percentile to inspect overlay detections.";
  $("prepMethod").value = "none";
  $("removeBands").value = "";
  $("cropWindow").value = "";
  $("medianSize").value = "0";
  $("drMethod").value = "none";
  $("nComp").value = "10";
  updatePrepExtras();
  updateDrExtras();
  enableThreshold(false);
  drawSpectrum([]);
  renderCompare();
}

function stripResultPreview(session) {
  if (!session.preview) return session;
  return {
    ...session,
    last_result: null,
    results: [],
    preview: {
      rgb: session.preview.rgb,
      gt: session.preview.gt,
      shape: session.preview.shape,
    },
  };
}

function recommendedExcludeBands(session) {
  if (!session) return "";
  if (session.recommended_remove_bands) return String(session.recommended_remove_bands).replace(/[–—]/g, "-");
  const id = session.catalog_id || session.id || "";
  const spec = ((state.catalog && state.catalog.datasets) || []).find((item) => item.id === id);
  if (spec && spec.recommended_remove_bands) return String(spec.recommended_remove_bands).replace(/[–—]/g, "-");
  if (id === "aviris-i") return "0-5, 32-34, 93-96, 106-112, 152-165, 220-223";
  return "";
}

function applyExcludeDefaults(session) {
  $("removeBands").value = recommendedExcludeBands(session);
}

function applySession(session, { fresh = false } = {}) {
  const switched = Boolean(state.session && state.session.id !== session.id);
  if (switched || fresh) {
    resetAnalysisUI();
    session = stripResultPreview(session);
  }
  state.session = session;
  $("sessionName").textContent = session.name;
  $("sessionMeta").textContent = `${session.shape.height} × ${session.shape.width} × ${session.shape.bands}  · src ${session.original_bands} bands`;
  $("sceneFoot").textContent = session.has_gt ? "Ground truth available" : "No ground-truth mask";
  $("runHelp").textContent = "Pipeline ready.";
  setClearVisible(true);
  if (switched || fresh) applyExcludeDefaults(session);
  rememberSession(session);
  rememberResults(session.results || [], session.name);
  renderLoaded();
  renderDatasetCatalog();
  renderCompare();
  fillParams();
  updateViews();
  const last = session.last_result;
  if (last) {
    $("detValue").textContent = last.name;
    $("timeValue").textContent = `${last.elapsed}s`;
    $("aucValue").textContent = last.metrics && last.metrics.auc != null ? last.metrics.auc.toFixed(4) : "—";
    if ($("aucNote")) {
      $("aucNote").textContent = last.metrics && last.metrics.roc === "matlab-demo"
        ? "MATLAB demo.m · 5000 thresholds"
        : "Requires reference mask";
    }
    $("resultStatus").textContent = "Completed";
    drawROC(last.metrics);
    enableThreshold(true);
    if (state.resultView === "overlay" || state.resultView === "binary") applyThreshold();
  } else {
    enableThreshold(false);
  }
}

async function loadFreshSession(session) {
  const switched = !state.session || state.session.id !== session.id;
  if (switched) {
    session = await api(`/api/session/${encodeURIComponent(session.id)}/reset`, { method: "POST" });
    applySession(session, { fresh: true });
    return;
  }
  applySession(session);
}

function enableThreshold(on) {
  $("thresholdBox").classList.toggle("hidden", !on);
  $("threshold").disabled = !on;
  $("threshold").value = String(state.threshold);
  $("thresholdLabel").textContent = `${Number(state.threshold).toFixed(2)}th percentile`;
}

function applyThreshold() {
  if (!state.session || !state.session.last_result) return;
  const percentile = Number($("threshold").value);
  state.threshold = percentile;
  $("thresholdLabel").textContent = `${percentile.toFixed(2)}th percentile`;
  clearTimeout(state.thresholdTimer);
  state.thresholdTimer = setTimeout(async () => {
    try {
      const views = await api("/api/threshold", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.session.id, percentile }),
      });
      state.session.preview = { ...(state.session.preview || {}), overlay: views.overlay, binary: views.binary };
      $("thresholdInfo").textContent = `${(views.fraction * 100).toFixed(2)}% flagged · score > ${views.threshold.toPrecision(4)}`;
      if (state.resultView === "overlay" || state.resultView === "binary") updateViews();
    } catch (err) {
      $("thresholdInfo").textContent = err.message;
    }
  }, 120);
}

function rememberSession(session) {
  const slim = {
    id: session.id,
    name: session.name,
    shape: session.shape,
    original_bands: session.original_bands,
    has_gt: session.has_gt,
    origin: session.origin,
    catalog_id: session.catalog_id,
    preview: session.preview && session.preview.rgb
      ? { rgb: session.preview.rgb }
      : { rgb: `/api/library/${session.id}/preview.png` },
  };
  state.sessions = [slim, ...state.sessions.filter((item) => item.id !== session.id && item.catalog_id !== session.id)];
}

function sceneRgb(item) {
  return (item.preview && item.preview.rgb) || "";
}

function sceneOriginLabel(item) {
  if (item.origin === "catalog") return "Research catalogue";
  if (item.origin === "demo" || /demo|synthetic/i.test(item.name || "")) return "Synthetic demo";
  return "Loaded locally";
}

function isSceneLoaded(id) {
  return state.sessions.some((item) => item.id === id || item.catalog_id === id);
}

function rememberResults(results, dataset) {
  const incoming = [...results].reverse().map((item) => ({
    ...item,
    dataset: item.dataset || dataset || "—",
  }));
  const merged = [...incoming, ...state.recentResults];
  const seen = new Set();
  state.recentResults = merged.filter((item) => {
    if (!item.id || seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  }).slice(0, RECENT_LIMIT);
}

function renderLoaded() {
  $("loadedCount").textContent = state.sessions.length;
  $("navDataCount").textContent = state.sessions.length;
  $("datasetSelect").innerHTML = `<option value="">Select a dataset</option>` + state.sessions.map((item) =>
    `<option value="${esc(item.id)}">${esc(item.name)}</option>`
  ).join("");
  if (state.session) $("datasetSelect").value = state.session.id;
  $("loadedDatasets").innerHTML = state.sessions.length
    ? state.sessions.map((item) => `
      <article class="loaded-card">
        <div class="loaded-preview">
          ${sceneRgb(item) ? `<img src="${sceneRgb(item)}" alt="">` : ""}
          <span class="pill">${esc(sceneOriginLabel(item))}</span>
        </div>
        <div class="loaded-body">
          <h3>${esc(item.name)}</h3>
          <p>${item.shape.height} × ${item.shape.width} × ${item.shape.bands} · height × width × bands</p>
          <div class="card-actions">
            <span>${item.has_gt ? "Reference mask available" : "No reference mask"}</span>
            <button class="btn small" type="button" data-open-scene="${esc(item.id)}">Open scene</button>
          </div>
        </div>
      </article>`).join("")
    : `<div class="section-empty"><h3>Bring your first scene into focus</h3><p>Import a cube or start with the bundled synthetic scene to explore the full workflow.</p><button class="btn" type="button" id="emptyLoadedDemo">Load demo scene</button></div>`;
  const emptyDemo = $("emptyLoadedDemo");
  if (emptyDemo) emptyDemo.onclick = loadDemo;
}

function renderCompare() {
  const results = state.recentResults.slice(0, RECENT_LIMIT);
  const rows = results.length
    ? results.map((r) => `<tr><td>${esc(r.dataset || "—")}</td><td>${esc(r.name)}</td><td>${r.auc == null ? "—" : Number(r.auc).toFixed(4)}</td><td>${r.elapsed}s</td></tr>`).join("")
    : `<tr><td colspan="4">No results yet</td></tr>`;
  $("compare").innerHTML = `<table><thead><tr><th>Dataset</th><th>Detector</th><th>AUC</th><th>Time</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function setImage(img, src, empty) {
  if (!src) {
    img.removeAttribute("src");
    img.classList.add("hidden");
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  img.classList.remove("hidden");
  img.src = src;
}

function updateResultFoot() {
  const view = state.resultView;
  if (view === "binary") {
    $("resultFoot").textContent = "Higher score = more anomalous";
    $("resultLegend").classList.remove("hidden");
    $("resultLegend").innerHTML = "<i></i> Low → High";
  } else if (view === "overlay") {
    $("resultFoot").textContent = "Magenta marks pixels above the percentile";
    $("resultLegend").classList.remove("hidden");
    $("resultLegend").innerHTML = "<i class='overlay-swatch'></i> Anomaly";
  } else {
    $("resultFoot").textContent = "Higher score = more anomalous";
    $("resultLegend").classList.remove("hidden");
    $("resultLegend").innerHTML = "<i></i> Low → High";
  }
}

function resetViewers() {
  state.viewZoom = 1;
  applyViewerZoom(0, 0);
}

function applyViewerZoom(left, top) {
  ["sceneViewport", "resultViewport"].forEach((id) => {
    const pane = $(id);
    pane.style.setProperty("--zoom", String(state.viewZoom));
    pane.classList.toggle("is-scrollable", state.viewZoom > 1);
    if (left != null) pane.scrollLeft = left;
    if (top != null) pane.scrollTop = top;
  });
}

function syncViewerScroll(source) {
  if (state.viewSyncing || state.viewZoom <= 1) return;
  const target = source.id === "sceneViewport" ? $("resultViewport") : $("sceneViewport");
  state.viewSyncing = true;
  const sx = source.scrollWidth - source.clientWidth;
  const sy = source.scrollHeight - source.clientHeight;
  const tx = target.scrollWidth - target.clientWidth;
  const ty = target.scrollHeight - target.clientHeight;
  if (sx > 0) target.scrollLeft = source.scrollLeft * (tx / sx);
  else target.scrollLeft = 0;
  if (sy > 0) target.scrollTop = source.scrollTop * (ty / sy);
  else target.scrollTop = 0;
  requestAnimationFrame(() => { state.viewSyncing = false; });
}

function bindPairedViewers() {
  const panes = ["sceneViewport", "resultViewport"].map((id) => $(id));
  panes.forEach((pane) => {
    pane.addEventListener("scroll", () => syncViewerScroll(pane), { passive: true });
    pane.addEventListener("wheel", (event) => {
      if (!state.session || !state.session.preview) return;
      event.preventDefault();
      const before = state.viewZoom;
      const next = Math.round((before + (event.deltaY < 0 ? 0.04 : -0.04)) * 100) / 100;
      state.viewZoom = Math.min(2, Math.max(1, next));
      if (state.viewZoom === before) return;
      const ratio = state.viewZoom / before;
      const box = pane.getBoundingClientRect();
      const x = event.clientX - box.left + pane.scrollLeft;
      const y = event.clientY - box.top + pane.scrollTop;
      applyViewerZoom(Math.max(0, x * ratio - (event.clientX - box.left)), Math.max(0, y * ratio - (event.clientY - box.top)));
    }, { passive: false });
    let drag = null;
    pane.addEventListener("pointerdown", (event) => {
      if (event.target.tagName !== "IMG") return;
      drag = { x: event.clientX, y: event.clientY, left: pane.scrollLeft, top: pane.scrollTop, moved: false, imageId: event.target.id };
      if (state.viewZoom > 1) {
        pane.classList.add("is-panning");
        pane.setPointerCapture(event.pointerId);
      }
    });
    pane.addEventListener("pointermove", (event) => {
      if (!drag) return;
      if (Math.abs(event.clientX - drag.x) + Math.abs(event.clientY - drag.y) > 4) drag.moved = true;
      if (state.viewZoom <= 1) return;
      pane.scrollLeft = drag.left - (event.clientX - drag.x);
      pane.scrollTop = drag.top - (event.clientY - drag.y);
    });
    const endDrag = (event) => {
      const start = drag;
      drag = null;
      pane.classList.remove("is-panning");
      if (!start || start.moved || !event || !start.imageId) return;
      inspectPixel(event, start.imageId);
    };
    pane.addEventListener("pointerup", endDrag);
    pane.addEventListener("pointercancel", () => { drag = null; pane.classList.remove("is-panning"); });
  });
}

function updateViews() {
  const preview = state.session && state.session.preview;
  setImage($("sceneImage"), preview && (preview[state.sceneView] || preview.rgb), $("emptyState"));
  const result = preview && (preview[state.resultView] || preview.heatmap || preview.overlay);
  setImage($("resultImage"), result, $("resultEmpty"));
  updateResultFoot();
  applyViewerZoom();
}

function drawROC(metrics) {
  const ctx = $("rocChart");
  if (state.roc) state.roc.destroy();
  const points = metrics && metrics.fpr && metrics.fpr.length
    ? metrics.fpr.map((x, i) => ({ x, y: metrics.tpr[i] }))
    : [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  state.roc = new Chart(ctx, {
    type: "line",
    data: { datasets: [{ data: points, borderColor: "#2fd3d0", pointRadius: 0, fill: false, tension: 0.15 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { type: "linear", min: 0, max: 1, beginAtZero: true, ticks: { maxTicksLimit: 6 } },
        y: { type: "linear", min: 0, max: 1, beginAtZero: true, ticks: { maxTicksLimit: 6 } },
      },
    },
  });
}

function drawSpectrum(values) {
  const ctx = $("specChart");
  if (state.spec) state.spec.destroy();
  state.spec = new Chart(ctx, {
    type: "line",
    data: { labels: values.map((_, i) => i + 1), datasets: [{ data: values, borderColor: "#8b7cff", pointRadius: 0, tension: 0.2 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { beginAtZero: true } },
    },
  });
}

function showInspectorTab(tab) {
  document.querySelectorAll(".inspector .tabs button").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tab));
  $("specChart").classList.toggle("is-off", tab !== "spectrum");
  $("rocChart").classList.toggle("is-off", tab !== "roc");
  const chart = tab === "roc" ? state.roc : state.spec;
  if (chart) requestAnimationFrame(() => chart.resize());
}

function fillFamilies() {
  $("familySelect").innerHTML = Object.entries(state.catalog.families).map(([id, name]) => `<option value="${id}">${name}</option>`).join("");
  $("modelFamilyFilter").innerHTML = `<option value="all">All method families</option>` + Object.values(state.catalog.families).map((name) => `<option value="${esc(name)}">${esc(name)}</option>`).join("");
  fillDetectors();
}

function fillDetectors(preferred) {
  const dets = state.catalog.detectors.filter((d) => d.family === $("familySelect").value);
  $("detectorSelect").innerHTML = dets.map((d) => `<option value="${d.id}">${d.name}</option>`).join("");
  if (preferred && dets.some((d) => d.id === preferred)) $("detectorSelect").value = preferred;
  fillParams();
}

function fillParams() {
  if (!state.catalog) return;
  const det = state.catalog.detectors.find((d) => d.id === $("detectorSelect").value);
  if (!det) return;
  if (state.selectedDetector && state.selectedDetector !== det.id) clearRunReadouts();
  state.selectedDetector = det.id;
  $("detectorMeta").textContent = `${det.reference} · ${det.source}`;
  const skip = new Set(["ridge", "cov", "kpca_train", "kernel_param"]);
  const entries = Object.entries(det.defaults).filter(([k]) => !skip.has(k));
  $("paramBox").innerHTML = entries.map(([k, v]) => {
    let value = v;
    if (k === "tree_size" && (v === "auto" || v === "" || v === 0)) value = defaultTreeSize();
    const numeric = value === "" || typeof value === "number";
    const placeholder = k === "tree_size" ? "3% of rows × columns" : "";
    const hint = PARAM_HINTS[k] ? ` <span class="optional">${PARAM_HINTS[k]}</span>` : "";
    return `<label>${paramLabel(k)}${hint}<input data-param="${k}" type="${numeric ? "number" : "text"}" value="${value}" step="any" placeholder="${placeholder}" /></label>`;
  }).join("");
  const note = $("paramNote");
  if (note) {
    const text = DETECTOR_NOTES[det.id] || "";
    note.textContent = text;
    note.classList.toggle("hidden", !text);
  }
}

function collectParams() {
  const params = {};
  $("paramBox").querySelectorAll("[data-param]").forEach((el) => {
    const raw = el.value;
    params[el.dataset.param] = raw === "" || Number.isNaN(Number(raw)) ? raw : Number(raw);
  });
  return params;
}

function renderDatasetCatalog() {
  const query = ($("datasetSearch").value || "").trim().toLowerCase();
  const items = (state.catalog.datasets || []).filter((d) => `${d.name} ${d.collection} ${d.description}`.toLowerCase().includes(query));
  $("datasetCatalog").innerHTML = items.map((d) => `
    <article class="catalog-card">
      <div class="catalog-top"><span class="collection">${esc(d.collection || "Research data")}</span><span class="status-pill">${d.synthetic ? "Synthetic" : "Provider"}</span></div>
      <h3>${esc(d.name)}</h3>
      <p>${esc(d.description || "Hyperspectral research scene from the original provider.")}</p>
      ${d.notes ? `<p>${esc(d.notes)}</p>` : ""}
      <div class="card-actions">
        ${d.source ? `<a class="text-btn" href="${esc(d.source)}" target="_blank" rel="noreferrer">Provider repository</a>` : "<span></span>"}
        ${isSceneLoaded(d.id)
          ? `<button class="btn small" type="button" data-open-scene="${esc(d.id)}">Open scene</button>`
          : d.downloadable
            ? `<button class="btn small" type="button" data-fetch-scene="${esc(d.id)}">Import scene</button>`
            : `<button class="btn small" type="button" data-open-upload="1">Import locally</button>`}
      </div>
    </article>
  `).join("") || `<div class="section-empty"><p>No datasets match your search.</p></div>`;
}

function renderModels() {
  const detectors = state.catalog.detectors;
  const ready = detectors.filter((d) => d.ready).length;
  const families = new Set(detectors.map((d) => d.family_name)).size;
  $("registrySummary").innerHTML = [
    [detectors.length, "Catalogue detectors"],
    [ready, "Ready in this environment"],
    [families, "Method families"],
    [5, "Reduction techniques"],
  ].map(([num, label]) => `<div class="summary-card"><strong>${num}</strong><span>${label}</span></div>`).join("");
  const query = ($("modelSearch").value || "").toLowerCase();
  const family = $("modelFamilyFilter").value;
  const readyOnly = $("readyOnly").checked;
  const filtered = detectors.filter((d) =>
    (!query || `${d.name} ${d.id} ${d.description} ${d.family_name}`.toLowerCase().includes(query))
    && (family === "all" || d.family_name === family)
    && (!readyOnly || d.ready)
  );
  $("modelFilterCount").textContent = `${filtered.length} of ${detectors.length} detectors`;
  $("navModelCount").textContent = detectors.length;
  $("modelRegistry").innerHTML = filtered.map((d) => `
    <article class="model-card">
      <div class="model-card-head">
        <span class="model-family">${esc(d.family_name)}</span>
        <span class="status-pill ${d.ready ? "" : "setup"}">${d.ready ? "Ready" : "Setup needed"}</span>
      </div>
      <h3>${esc(d.name)}</h3>
      <p>${esc(d.description)}</p>
      <div class="model-impl">${esc(d.implementation)} · ${esc(d.source)}</div>
      <div class="card-actions">
        ${d.url ? `<a class="text-btn" href="${esc(d.url)}" target="_blank" rel="noreferrer">Method details</a>` : `<button class="text-btn" type="button" data-model-details="${esc(d.id)}">Method details</button>`}
        <button class="btn small" type="button" data-use-model="${esc(d.id)}">${d.ready ? "Use detector" : "Configure"}</button>
      </div>
    </article>
  `).join("") || `<div class="section-empty"><h3>No matching detectors</h3><p>Change your search or readiness filter.</p></div>`;
}

function openModel(id) {
  const det = state.catalog.detectors.find((d) => d.id === id);
  if (!det) return;
  state.detailId = id;
  $("modelDetailFamily").textContent = det.family_name;
  $("modelDetailTitle").textContent = det.name;
  const params = Object.entries(det.defaults || {});
  $("modelDetailBody").innerHTML = `
    <p>${esc(det.description)}</p>
    <p class="hint">${esc(det.reference)}</p>
    ${params.length ? `<table class="param-table"><tbody>${params.map(([k, v]) => `<tr><td>${esc(paramLabel(k))}</td><td>${esc(paramDefaultText(k, v))}</td></tr>`).join("")}</tbody></table>` : "<p class='hint'>No tunable parameters.</p>"}
    ${DETECTOR_NOTES[det.id] ? `<p class="hint">${esc(DETECTOR_NOTES[det.id])}</p>` : ""}
  `;
  $("modelDetailSource").innerHTML = det.url
    ? `<a class="text-btn" href="${esc(det.url)}" target="_blank" rel="noreferrer">${esc(det.source || "Open source")}</a>`
    : esc(det.source || "");
  $("modelDialog").showModal();
}

function useModel(id) {
  const det = state.catalog.detectors.find((d) => d.id === id);
  if (!det) return;
  $("familySelect").value = det.family;
  fillDetectors(det.id);
  if ($("modelDialog").open) $("modelDialog").close();
  showView("workspace");
}

async function openScene(id) {
  if (state.session && state.session.id === id) {
    showView("workspace");
    return;
  }
  setBusy(true, "Opening scene…");
  try {
    await loadFreshSession(await api(`/api/session/${encodeURIComponent(id)}`));
    showView("workspace");
  } catch (err) { alert(err.message); }
  finally { setBusy(false); }
}

async function loadDemo() {
  setBusy(true, "Generating synthetic scene…");
  try {
    await loadFreshSession(await api("/api/demo", { method: "POST" }));
    showView("workspace");
  } catch (err) { alert(err.message); }
  finally { setBusy(false); }
}

async function fetchScene(id) {
  setBusy(true, "Importing catalogue scene…");
  try {
    await loadFreshSession(await api("/api/datasets/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ catalog_id: id }),
    }));
    showView("workspace");
  } catch (err) { alert(err.message); }
  finally { setBusy(false); }
}

async function uploadScene(event) {
  event.preventDefault();
  const cube = $("cubeFile").files[0];
  if (!cube) return alert("Choose a hyperspectral cube file first.");
  const form = new FormData();
  form.append("cube", cube);
  if ($("gtFile").files[0]) form.append("gt", $("gtFile").files[0]);
  setBusy(true, "Reading cube…");
  try {
    await loadFreshSession(await api("/api/upload", { method: "POST", body: form }));
    $("uploadDialog").close();
    showView("workspace");
  } catch (err) { alert(err.message); }
  finally { setBusy(false); }
}

function unloadWorkspace() {
  state.session = null;
  resetAnalysisUI();
  $("sessionName").textContent = "No scene loaded";
  $("sessionMeta").textContent = "Start with a scene to begin";
  $("sceneFoot").textContent = "No scene loaded";
  $("runHelp").textContent = "Load a dataset to start an experiment.";
  setClearVisible(false);
  fillParams();
  renderLoaded();
  updateViews();
  showInspectorTab("spectrum");
  showView("workspace");
}

async function clearAll() {
  const sessionId = state.session && state.session.id;
  setBusy(true, "Clearing workspace…");
  try {
    if (sessionId) {
      await api(`/api/session/${encodeURIComponent(sessionId)}/reset`, { method: "POST" });
    }
    unloadWorkspace();
  } catch (err) { alert(err.message); }
  finally { setBusy(false); }
}

async function runPipeline() {
  if (!state.session) return alert("Load a scene first.");
  setBusy(true, `Running ${$("detectorSelect").selectedOptions[0].text}…`);
  try {
    applySession(await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.session.id,
        preprocess: $("prepMethod").value,
        prep_params: {
          median_size: Number($("medianSize").value) || 0,
          remove_bands: ($("removeBands").value || "").trim().replace(/[–—]/g, "-"),
          crop: ($("cropWindow").value || "").trim().replace(/[–—]/g, "-"),
        },
        dimred: $("drMethod").value,
        n_components: Number($("nComp").value),
        detector: $("detectorSelect").value,
        params: collectParams(),
      }),
    }));
    state.resultView = "heatmap";
    document.querySelectorAll("#resultModes button").forEach((b) => b.classList.toggle("active", b.dataset.view === "heatmap"));
    updateViews();
  } catch (err) { alert(err.message); }
  finally { setBusy(false); }
}

async function inspectPixel(event, imageId) {
  if (!state.session) return;
  const img = $(imageId);
  if (!img || !img.getBoundingClientRect().width) return;
  const rect = img.getBoundingClientRect();
  const col = Math.floor(((event.clientX - rect.left) / rect.width) * state.session.shape.width);
  const row = Math.floor(((event.clientY - rect.top) / rect.height) * state.session.shape.height);
  try {
    const data = await api("/api/spectrum", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.session.id, row, col }),
    });
    $("pixelReadout").textContent = `(${data.row}, ${data.col})`;
    showInspectorTab("spectrum");
    drawSpectrum(data.spectrum);
  } catch (err) {
    $("pixelReadout").textContent = err.message;
  }
}

function updatePrepExtras() {
  $("prepExtras").classList.toggle("hidden", $("prepMethod").value === "none");
}

function updateDrExtras() {
  const method = $("drMethod").value;
  $("drExtras").classList.toggle("hidden", method === "none");
  $("drHint").textContent = DR_HINTS[method] || "";
}

document.querySelectorAll(".nav-item").forEach((btn) => { btn.onclick = () => showView(btn.dataset.view); });
document.querySelectorAll("#sceneModes button").forEach((btn) => {
  btn.onclick = () => {
    state.sceneView = btn.dataset.view;
    document.querySelectorAll("#sceneModes button").forEach((b) => b.classList.toggle("active", b === btn));
    updateViews();
  };
});
document.querySelectorAll("#resultModes button").forEach((btn) => {
  btn.onclick = () => {
    state.resultView = btn.dataset.view;
    document.querySelectorAll("#resultModes button").forEach((b) => b.classList.toggle("active", b === btn));
    updateViews();
    if (state.resultView === "overlay" || state.resultView === "binary") applyThreshold();
  };
});
document.querySelectorAll(".inspector .tabs button").forEach((btn) => {
  btn.onclick = () => showInspectorTab(btn.dataset.tab);
});

document.addEventListener("click", (event) => {
  const open = event.target.closest("[data-open-scene]");
  if (open) { openScene(open.dataset.openScene); return; }
  const fetchBtn = event.target.closest("[data-fetch-scene]");
  if (fetchBtn) { fetchScene(fetchBtn.dataset.fetchScene); return; }
  const details = event.target.closest("[data-model-details]");
  if (details) { openModel(details.dataset.modelDetails); return; }
  const use = event.target.closest("[data-use-model]");
  if (use) { useModel(use.dataset.useModel); return; }
  if (event.target.closest("[data-open-upload]")) $("uploadDialog").showModal();
});

$("demoBtn").onclick = loadDemo;
$("demoBtn2").onclick = loadDemo;
$("emptyDemo").onclick = loadDemo;
$("clearBtn").onclick = clearAll;
$("clearBtn2").onclick = clearAll;
$("importBtn").onclick = () => $("uploadDialog").showModal();
$("importBtn2").onclick = () => $("uploadDialog").showModal();
$("closeUpload").onclick = () => $("uploadDialog").close();
$("closeModel").onclick = () => $("modelDialog").close();
$("useModelBtn").onclick = () => useModel(state.detailId);
$("uploadForm").onsubmit = uploadScene;
$("runBtn2").onclick = runPipeline;
$("familySelect").onchange = () => fillDetectors();
$("detectorSelect").onchange = fillParams;
$("prepMethod").onchange = updatePrepExtras;
$("drMethod").onchange = updateDrExtras;
$("datasetSearch").oninput = renderDatasetCatalog;
$("modelSearch").oninput = renderModels;
$("modelFamilyFilter").onchange = renderModels;
$("readyOnly").onchange = renderModels;
$("datasetSelect").onchange = (event) => { if (event.target.value) openScene(event.target.value); };
$("threshold").oninput = applyThreshold;
bindPairedViewers();

api("/api/catalog").then((catalog) => {
  state.catalog = catalog;
  state.sessions = catalog.library || [];
  fillFamilies();
  renderDatasetCatalog();
  renderModels();
  renderLoaded();
  renderCompare();
  updatePrepExtras();
  updateDrExtras();
  drawROC(null);
  drawSpectrum([]);
}).catch((err) => {
  $("emptyState").innerHTML = `<p>Could not reach the HAD Studio API: ${err.message}</p>`;
});
