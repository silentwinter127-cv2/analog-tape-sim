/*
 * 效果矩阵 — see every effect of one control at once, on your own material.
 *
 * The complaint this answers: validating the model meant running scripts or
 * dragging a slider back and forth and remembering what the previous position
 * looked like.  Neither is review.  Here a single parameter is rendered at a set
 * of values and laid out as one grid, so the *relationship* between the control
 * and the picture is visible in one glance, and a second parameter can be swept
 * at the same time where two mechanisms interact.
 *
 * Everything is driven from the schema, so a parameter that is added to the engine
 * appears here without any change to this file.
 */
(function () {
  "use strict";

  let schemaCache = null;
  let current = { layer: "tape", key: "", values: [] };

  async function getSchema() {
    if (schemaCache) return schemaCache;
    const r = await fetch("/api/schema");
    schemaCache = await r.json();
    return schemaCache;
  }

  /* Build a sensible set of values for a parameter: inside its physical range
   * where one exists, because those are the settings the model claims are real. */
  function valuesFor(spec) {
    const lo = spec.phys_min != null ? spec.phys_min : spec.min;
    const hi = spec.phys_max != null ? spec.phys_max : spec.max;
    const n = 5;
    const out = [];
    for (let i = 0; i < n; i++) {
      const v = lo + (hi - lo) * (i / (n - 1));
      out.push(Number(v.toFixed(4)));
    }
    return out;
  }

  /* The schema nests layers -> groups -> params, and the parameters carry
   * min/max (not minimum/maximum).  Flattening it here keeps the rest of this
   * file free of that shape. */
  function paramList(schema) {
    const out = [];
    for (const layer of (schema.layers || [])) {
      const groups = layer.groups || {};
      for (const gname of Object.keys(groups)) {
        for (const spec of (groups[gname] || [])) {
          out.push({ layer: layer.layer, key: spec.key, spec, group: gname });
        }
      }
    }
    return out;
  }

  function label(param) {
    const s = param.spec || {};
    const name = s.label ? s.label : param.key;
    return `${param.layer} · ${name}`;
  }

  function fillParamSelect(schema, keep) {
    const sel = document.getElementById("mxParam");
    if (!sel) return [];
    const list = paramList(schema).filter((p) =>
      p.spec && p.spec.max > p.spec.min);
    sel.innerHTML = "";
    for (const p of list) {
      const o = document.createElement("option");
      o.value = `${p.layer}|${p.key}`;
      o.textContent = label(p);
      o.dataset.spec = JSON.stringify(p.spec || {});
      sel.appendChild(o);
    }
    if (keep) sel.value = keep;
    if (!sel.value && list.length) sel.value = `${list[0].layer}|${list[0].key}`;
    return list;
  }

  function selectedParam(schema) {
    const sel = document.getElementById("mxParam");
    if (!sel || !sel.value) return null;
    const [layer, key] = sel.value.split("|");
    const opt = sel.options[sel.selectedIndex];
    let spec = {};
    try { spec = JSON.parse(opt.dataset.spec || "{}"); } catch (e) { /* keep {} */ }
    return { layer, key, spec };
  }

  function renderGrid(data, param, param2) {
    const body = document.getElementById("mxBody");
    const info = document.getElementById("mxInfo");
    if (!body) return;
    const cols = Math.max(1, data.cols || data.tiles.length);
    if (info) {
      info.textContent =
        `${data.tiles.length} 格 · ${cols} 列 · ${data.ms} ms · t=${data.t.toFixed(2)}s`;
    }
    body.innerHTML = "";
    const grid = document.createElement("div");
    grid.className = "mxgrid";
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    for (const tile of data.tiles) {
      const cell = document.createElement("figure");
      cell.className = "mxcell";
      const img = document.createElement("img");
      img.src = "data:image/jpeg;base64," + tile.jpeg;
      img.alt = String(tile.value);
      const cap = document.createElement("figcaption");
      let text = `${param.key} = ${tile.value}`;
      if (tile.value2 !== undefined && param2) {
        text += ` · ${param2.key} = ${tile.value2}`;
      }
      cap.textContent = text;
      const sub = document.createElement("span");
      sub.className = "mxmean";
      sub.textContent = `均值 ${tile.mean}`;
      cap.appendChild(sub);
      cell.appendChild(img);
      cell.appendChild(cap);
      // clicking a cell adopts that value, so browsing the grid can set the project
      cell.onclick = () => {
        const over = { [param.layer]: { [param.key]: tile.value } };
        if (tile.value2 !== undefined && param2) {
          over[param2.layer] = over[param2.layer] || {};
          over[param2.layer][param2.key] = tile.value2;
        }
        fetch("/api/project", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ params: over }),
        }).then(() => {
          if (typeof S !== "undefined") S.layer = param.layer;
          if (typeof buildLayerTabs === "function") buildLayerTabs();
          if (typeof renderParams === "function") renderParams();
          if (typeof refreshPreview === "function") refreshPreview();
        }).catch(() => {});
      };
      grid.appendChild(cell);
    }
    body.appendChild(grid);
  }

  async function runSweep() {
    const btn = document.getElementById("mxRun");
    const body = document.getElementById("mxBody");
    const schema = await getSchema();
    const param = selectedParam(schema);
    if (!param) return;
    const values = valuesFor(param.spec);
    current = { layer: param.layer, key: param.key, values };

    let second = null;
    const sel2 = document.getElementById("mxParam2");
    if (sel2 && sel2.value) {
      const [l2, k2] = sel2.value.split("|");
      const opt2 = sel2.options[sel2.selectedIndex];
      let spec2 = {};
      try { spec2 = JSON.parse(opt2.dataset.spec || "{}"); } catch (e) { /* keep */ }
      second = { layer: l2, key: k2, values: valuesFor(spec2).slice(0, 3) };
    }

    if (btn) { btn.disabled = true; btn.textContent = "渲染中…"; }
    if (body) body.innerHTML = '<p class="note">正在逐格渲染…</p>';
    try {
      const req = { layer: param.layer, key: param.key, values,
                    t: (typeof S !== "undefined" ? S.t : 0), scale: 0.35 };
      if (second) req.second = second;
      const r = await fetch("/api/sweep", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      renderGrid(j, param, second);
    } catch (e) {
      if (body) body.innerHTML = `<p class="note">失败：${e.message}</p>`;
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = "渲染矩阵"; }
    }
  }

  async function openMatrix() {
    const m = document.getElementById("mxModal");
    if (m) m.classList.remove("hidden");
    try {
      const schema = await getSchema();
      const sel = document.getElementById("mxParam");
      const keep = sel && sel.value;
      fillParamSelect(schema, keep || "tape|drum_speed");
      const sel2 = document.getElementById("mxParam2");
      if (sel2) {
        sel2.innerHTML = '<option value="">（不叠加第二个参数）</option>';
        for (const o of document.getElementById("mxParam").options) {
          const c = document.createElement("option");
          c.value = o.value; c.textContent = o.textContent;
          c.dataset.spec = o.dataset.spec;
          sel2.appendChild(c);
        }
      }
      if (!document.querySelector("#mxBody .mxgrid")) await runSweep();
    } catch (e) {
      const body = document.getElementById("mxBody");
      if (body) body.textContent = "加载失败：" + e.message;
    }
  }

  function closeMatrix() {
    const m = document.getElementById("mxModal");
    if (m) m.classList.add("hidden");
  }

  function initMatrix() {
    const b = document.getElementById("btnMatrix");
    if (b) b.onclick = openMatrix;
    const c = document.getElementById("mxClose");
    if (c) c.onclick = closeMatrix;
    const run = document.getElementById("mxRun");
    if (run) run.onclick = runSweep;
    const sel = document.getElementById("mxParam");
    if (sel) sel.onchange = () => { const g = document.querySelector("#mxBody .mxgrid"); if (g) g.remove(); };
    const m = document.getElementById("mxModal");
    if (m) m.onclick = (ev) => { if (ev.target === m) closeMatrix(); };
    // deep link, matching the canvas pane's behaviour
    if (location.hash === "#matrix") setTimeout(openMatrix, 400);
    window.openMatrix = openMatrix;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMatrix);
  } else {
    initMatrix();
  }
})();
