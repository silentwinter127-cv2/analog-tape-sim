/*
 * 故障速查 — the fault catalogue, rendered from /api/faults.
 *
 * Deliberately a *table*, not a slider panel: the point is to let a symptom be
 * traced back to a mechanism, and to show plainly which mechanisms this
 * simulator does not implement.  Clicking a row jumps to the parameter layer
 * that drives it, so the reference and the controls describe one apparatus.
 */
(function () {
  "use strict";

  let cached = null;

  function statusPill(status) {
    const map = {
      modelled: ["已建模", "ok"],
      approx: ["近似", "warn"],
      "not-modelled": ["未建模", "bad"],
    };
    const [label, cls] = map[status] || [status, ""];
    return `<span class="pill ${cls}">${label}</span>`;
  }

  async function loadFaults() {
    if (cached) return cached;
    const r = await fetch("/api/faults");
    cached = await r.json();
    return cached;
  }

  function renderFaults(data) {
    const body = document.getElementById("faultBody");
    const cov = document.getElementById("faultCoverage");
    if (!body) return;
    const c = data.coverage || {};
    if (cov) {
      cov.textContent =
        `${c.total} 条：已建模 ${c.modelled} · 近似 ${c.approx} · 未建模 ${c.not_modelled}`;
    }
    const labels = data.stage_labels || {};
    let html = `<table class="ftable">
      <thead><tr>
        <th>环节</th><th>典型画面异常</th><th>主要原因</th>
        <th>物理因素 / 机理</th><th>本工具如何表达</th><th></th>
      </tr></thead><tbody>`;
    let lastStage = null;
    for (const f of data.faults || []) {
      const stageCell = f.stage === lastStage ? "" :
        `<td class="stage">${labels[f.stage] || f.stage}</td>`;
      lastStage = f.stage;
      // a row with parameters is clickable: it jumps to the layer that owns them
      const target = (f.params && f.params.length) ? f.params[0][0] : "";
      const cls = f.status === "not-modelled" ? "row-missing" : "";
      html += `<tr class="${cls}" ${target ? `data-layer="${target}"` : ""}>
        ${stageCell}
        <td class="sym">${f.symptom}</td>
        <td class="cause">${f.causes}</td>
        <td class="phys">${f.physics}</td>
        <td class="model">${f.model}</td>
        <td class="st">${statusPill(f.status)}</td>
      </tr>`;
    }
    html += "</tbody></table>";
    body.innerHTML = html;

    body.querySelectorAll("tr[data-layer]").forEach((tr) => {
      tr.onclick = () => {
        const layer = tr.dataset.layer;
        if (typeof S !== "undefined" && layer) {
          S.layer = layer;
          if (typeof buildLayerTabs === "function") buildLayerTabs();
          if (typeof renderParams === "function") renderParams();
        }
        closeFaults();
      };
    });
  }

  function openFaults() {
    const m = document.getElementById("faultModal");
    if (m) m.classList.remove("hidden");
    loadFaults().then(renderFaults).catch((e) => {
      const body = document.getElementById("faultBody");
      if (body) body.textContent = "加载失败：" + e.message;
    });
  }

  function closeFaults() {
    const m = document.getElementById("faultModal");
    if (m) m.classList.add("hidden");
  }

  function initFaults() {
    const btn = document.getElementById("btnFaults");
    if (btn) btn.onclick = openFaults;
    const close = document.getElementById("faultClose");
    if (close) close.onclick = closeFaults;
    const modal = document.getElementById("faultModal");
    if (modal) {
      modal.onclick = (ev) => { if (ev.target === modal) closeFaults(); };
    }
    window.openFaults = openFaults;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initFaults);
  } else {
    initFaults();
  }
})();
