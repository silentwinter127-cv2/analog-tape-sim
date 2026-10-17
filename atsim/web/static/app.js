/* analog-tape-sim WebUI
 *
 * The page is deliberately schema-driven: every control is generated from
 * /api/schema, so adding a parameter to the Python schema makes it appear here
 * with its unit, its physical range and its provenance.  That is what keeps the
 * UI and the signal model from drifting apart.
 */
"use strict";

const S = {
  schema: null,
  catalog: null,
  project: null,
  params: {},
  layer: "format",
  presetTab: "factory",
  t: 0.4,
  previewScale: "0.5",
  previewQuality: 88,
  exportScale: "1.0",
  exportQuality: "high",
  ab: true,
  auto: true,
  busy: false,
  pending: false,
  token: 0,
  frameUrl: null,
  origUrl: null,
  dirtyStages: {},
};

const $ = (id) => document.getElementById(id);
const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(0)
  : Math.abs(v) >= 10 ? v.toFixed(1)
  : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.error || j.trace || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.blob();
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  S.schema = await api("/api/schema");
  S.catalog = (await api("/api/catalog")).catalog;
  applyProject(await api("/api/project"));
  buildLayerTabs();
  buildPresetTabs();
  renderPresetList();
  renderParams();
  wireChrome();
  refreshPreview();
  loadTimelineMeta();
  // the canvas lives in canvas.js; it needs the schema before it can render a
  // node inspector, so it is initialised after the schema arrives
  if (typeof initCanvas === "function") {
    initCanvas();
    await loadGraph();
    // open the canvas only once the graph is loaded, otherwise it would draw
    // from an empty state
    if (location.hash === "#canvas" && typeof window.showCanvas === "function")
      window.showCanvas();
  }
}

function applyProject(d) {
  S.project = d.project;
  S.params = d.params;
  $("seed").value = d.project.seed;
  const info = d.source_info || {};
  $("badge").textContent = d.project.source
    ? `${(info.width||"?")}×${(info.height||"?")} · ${(info.fps||0).toFixed(3)} fps · ${info.frame_count||"?"} 帧`
    : "内置测试图（未导入素材）";
  // 维度面板：环境 / 设备个体 / 关键帧时间线。放在最后，
  // 所以面板反映的是服务端刚刚返回的状态——只有一份真相。
  if (window.SessionUI) SessionUI.syncFromProject(d);
}

/* --------------------------------------------------------------- chrome */
function wireChrome() {
  $("btnImport").onclick = () => $("clipFile").click();
  $("clipFile").onchange = uploadClip;
  const still = $("btnStill");
  if (still) still.onclick = exportStill;
  $("seed").onchange = () => setProject({ seed: parseInt($("seed").value, 10) || 0 });
  $("timeSlider").oninput = () => {
    S.t = parseFloat($("timeSlider").value);
    $("timeLabel").textContent = `t = ${S.t.toFixed(2)} s`;
    renderParams();
    if (window.SessionUI) {
      if (typeof SessionUI.renderKeyframes === "function") SessionUI.renderKeyframes();
      if (typeof SessionUI.renderEnv === "function") SessionUI.renderEnv();
    }
    refreshPreview();
  };
  const pScale = $("previewScale");
  if (pScale) {
    pScale.onchange = () => {
      S.previewScale = pScale.value || "0.5";
      refreshPreview();
    };
  }
  const pQual = $("previewQuality");
  if (pQual) {
    pQual.onchange = () => {
      S.previewQuality = parseInt(pQual.value, 10) || 88;
      refreshPreview();
    };
  }
  const eScale = $("exportScale");
  if (eScale) {
    eScale.onchange = () => { S.exportScale = eScale.value || "1.0"; };
  }
  const eQual = $("exportQuality");
  if (eQual) {
    eQual.onchange = () => { S.exportQuality = eQual.value || "high"; };
  }
  $("abToggle").onchange = () => { S.ab = $("abToggle").checked; layoutAB(); };
  $("autoRefresh").onchange = () => { S.auto = $("autoRefresh").checked; };
  $("btnRender").onclick = startRender;
  $("btnSave").onclick = () => api("/api/save", jsonPost({ name: S.project.name }));

  // **Configuration export and import.**  Export is a navigation rather than a fetch: the route sends the file with
  // a Content-Disposition header and a timestamped name, so the browser saves it without any code here.  Import has
  // to go through a file input, since a page cannot read a path the user did not choose.
  //
  // The whole point of the pair is that a reproduction can be handed over, so the import reports what it applied
  // rather than failing silently -- an import that half-applies is worse than one that refuses.
  const btnExport = $("btnExport");
  if (btnExport) {
    btnExport.onclick = () => {
      const a = document.createElement("a");
      a.href = "/api/export";
      a.download = "";
      document.body.appendChild(a);
      a.click();
      a.remove();
      toast("已导出设置文件");
    };
  }
  const btnImportCfg = $("btnImportCfg");
  const cfgFile = $("cfgFile");
  if (btnImportCfg && cfgFile) {
    btnImportCfg.onclick = () => cfgFile.click();
    cfgFile.onchange = async () => {
      const f = cfgFile.files && cfgFile.files[0];
      if (!f) return;
      try {
        const text = await f.text();
        // parsed here only to fail early with a clear message; the server parses it again
        const parsed = JSON.parse(text);
        if (parsed.kind && parsed.kind !== "atsim.config") {
          throw new Error(`这不是本程序的配置文件（kind=${parsed.kind}）`);
        }
        const r = await api("/api/import", jsonPost({ config: parsed }));
        if (r && r.error) throw new Error(r.error);
        const n = (r && r.imported && r.imported.params_applied) || 0;
        toast(`已导入 ${n} 项设置${parsed.exported ? "（导出于 " + parsed.exported + "）" : ""}`);
      } catch (err) {
        toast("导入失败：" + (err && err.message ? err.message : err));
      } finally {
        cfgFile.value = "";
      }
    };
  }
  $("btnTimeline").onclick = showTimeline;
  $("tlClose").onclick = () => $("modal").classList.add("hidden");
  $("jobClose").onclick = () => $("jobbar").classList.add("hidden");
  initDivider();
  layoutAB();
}

function jsonPost(obj) {
  return { method: "POST", headers: { "Content-Type": "application/json" },
           body: JSON.stringify(obj) };
}

function initDivider() {
  const v = $("viewer"), d = $("divider");
  let drag = false;
  const move = (e) => {
    if (!drag) return;
    const r = v.getBoundingClientRect();
    const x = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
    d.style.left = (x * 100) + "%";
    $("originalImg").style.clipPath = `inset(0 0 0 ${x * 100}%)`;
  };
  d.onmousedown = () => drag = true;
  window.addEventListener("mouseup", () => drag = false);
  window.addEventListener("mousemove", move);
  d.ontouchmove = (e) => move(e.touches[0]);
}

function layoutAB() {
  $("originalImg").style.display = S.ab ? "block" : "none";
  $("divider").style.display = S.ab ? "block" : "none";
  $("hint").style.display = S.ab ? "block" : "none";
  syncAB();
}

/**
 * Force the two images onto one identical rectangle.
 *
 * `object-fit: contain` sizes each image by *its own* aspect ratio, so two images
 * with different ratios are centred and scaled differently -- the comparison then
 * lines up nowhere, and content at the frame edges (the head-switching strip, most
 * obviously) never appears in the same place on both sides.  Here the rectangle is
 * computed once and applied to both, so a given screen pixel is the same part of
 * the picture on either side of the divider.
 */
function syncAB() {
  const v = $("viewer"), a = $("previewImg"), b = $("originalImg");
  if (!v || !a || !b) return;
  const vw = v.clientWidth, vh = v.clientHeight;
  const rw = S.frameW || a.naturalWidth || 4;
  const rh = S.frameH || a.naturalHeight || 3;
  if (!vw || !vh || !rw || !rh) return;
  const scale = Math.min(vw / rw, vh / rh);
  const w = rw * scale, h = rh * scale;
  const left = (vw - w) / 2, top = (vh - h) / 2;
  for (const el of [a, b]) {
    el.style.width = w + "px";
    el.style.height = h + "px";
    el.style.left = left + "px";
    el.style.top = top + "px";
    el.style.right = "auto";
    el.style.bottom = "auto";
    el.style.objectFit = "fill";   // the box now has the right ratio already
  }
}

/* --------------------------------------------------------------- tabs */
function buildLayerTabs() {
  // Drive the tabs from the *schema*, not a hardcoded list: the schema is the
  // single source of truth for which device layers exist, so adding a layer
  // (as happened with `codec`) cannot leave the UI out of step with the engine.
  const order = (S.schema.layers || []).map((l) => l.layer);
  const labels = S.catalog.layers || {};
  $("layerTabs").innerHTML = "";
  order.forEach((k) => {
    const b = document.createElement("button");
    // never trust a label to exist: a missing one must degrade, not crash
    b.textContent = (labels[k] || k).replace(/^(\d+)\s*/, "$1 ");
    b.className = k === S.layer ? "on" : "";
    b.onclick = () => { S.layer = k; buildLayerTabs(); renderParams(); };
    $("layerTabs").appendChild(b);
  });
  if (!order.includes(S.layer) && order.length) S.layer = order[0];
  renderChainBar();
}

function buildPresetTabs() {
  const tabs = [["factory", "整套配置"], ["media", "记录媒介"],
                ["player", "播放设备"], ["display", "显示设备"], ["camera", "摄像机"]];
  $("presetTabs").innerHTML = "";
  tabs.forEach(([k, label]) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.className = k === S.presetTab ? "on" : "";
    b.onclick = () => { S.presetTab = k; buildPresetTabs(); renderPresetList(); };
    $("presetTabs").appendChild(b);
  });
}

function presetEntries() {
  if (S.presetTab === "factory") {
    return Object.entries(S.catalog.factory).map(([k, v]) => [k, v.label, "整套：摄像机+格式+介质+播放+显示"]);
  }
  const g = S.catalog.profiles[S.presetTab] || {};
  return Object.entries(g).map(([k, v]) => [k, v.label, v.note || ""]);
}

function renderPresetList() {
  const box = $("presetList");
  box.innerHTML = "";
  presetEntries().forEach(([key, label, note]) => {
    const d = document.createElement("div");
    d.className = "card" + (S.project.preset === key ? " on" : "");
    d.innerHTML = `<div class="t"></div><div class="n"></div>`;
    d.querySelector(".t").textContent = label;
    d.querySelector(".n").textContent = note;
    d.onclick = () => applyPreset(key);
    box.appendChild(d);
  });
}

/* --------------------------------------------------------------- params */
function renderParams() {
  const layer = (S.schema.layers || []).find((l) => l.layer === S.layer);
  const box = $("paramList");
  box.innerHTML = "";
  if (!layer) return;
  Object.entries(layer.groups).forEach(([gname, specs]) => {
    const g = document.createElement("div");
    g.className = "pgroup";
    const h = document.createElement("h3");
    h.textContent = gname;
    g.appendChild(h);
    specs.forEach((sp) => g.appendChild(paramRow(sp)));
    box.appendChild(g);
  });
}

// **Parameters that are selectors rather than quantities.**  A night-vision mode is one of four named functions of the
// camera, so a slider reading "1.7" tells the user nothing -- and the schema has no `choices` field, so the names live
// here.  The control still calls `setParam(layer, key, value)` with a number, so the API, the settings file, the
// keyframes and the render path are all unchanged: only the widget differs.
const SELECTORS = {
  "camera.nv_mode": [
    [0, "关"],
    [1, "NightShot（红外 + 绿单色）"],
    [2, "Super NightShot（帧积累）"],
    [3, "Color Slow Shutter（慢快门保留彩色）"],
  ],
  "camera.zoom_kind": [
    [0, "0 · 线性 (linear)"],
    [1, "1 · 家用电动变焦马达 (motor)"],
    [2, "2 · 广播级伺服变焦 (cinema)"],
  ],
  "format.filter_kind": [
    [0, "0 · 高斯无相移 (gaussian)"],
    [1, "1 · 定 K 型多节级联 (constant_k)"],
    [2, "2 · 巴特沃斯有源滤波 (butterworth)"],
  ],
  "format.separation_kind": [
    [0, "0 · 频带分割 (band)"],
    [1, "1 · 窄孔径盒式 (box)"],
    [2, "2 · 1H 延迟线梳状 (line1)"],
    [3, "3 · 2H 双延迟线梳状 (line2)"],
    [4, "4 · 2D 对角 FIR 梳状 (fir2d)"],
    [5, "5 · 递归垂直色度混合 (iir)"],
  ],
  "codec.codec_kind": [
    [0, "0 · 关闭数字编解码（模拟直通）"],
    [1, "1 · DV25 / MiniDV (4:1:1 / 4:2:0 DCT)"],
    [2, "2 · MPEG-1 / Video CD (352×240 运动补偿)"],
  ],
  "tape.tape_speed": [
    [0, "0 · SP 标准速度"],
    [1, "1 · LP 二倍长时"],
    [2, "2 · EP / SLP 三倍超长时"],
  ],
  "tape.shuttle_mode": [
    [0, "0 · 自动（随介质格式）"],
    [1, "1 · 强制模拟整行条带 (VHS/Hi8)"],
    [2, "2 · 强制数字 16×16 宏块 (MiniDV)"],
  ],
  "player.separation_kind": [
    [0, "0 · 频带分割 (band)"],
    [1, "1 · 窄孔径盒式 (box)"],
    [2, "2 · 1H 延迟线梳状 (line1)"],
    [3, "3 · 2H 双延迟线梳状 (line2)"],
    [4, "4 · 2D 对角 FIR 梳状 (fir2d)"],
    [5, "5 · 递归垂直色度混合 (iir)"],
  ],
  "display.mask_type": [
    [0, "0 · 荫栅 (Aperture Grille / 特丽珑)"],
    [1, "1 · 槽孔荫罩 (Slot Mask)"],
    [2, "2 · 圆点荫罩 (Dot Triad Shadow Mask)"],
  ],
  "overlay.at": [
    [0, "0 · 画面成形之后（摄像机自身 OSD）"],
    [1, "1 · 介质之前（字幕机录到磁带上）"],
  ],
  "overlay.style": [
    [0, "0 · 自定义（使用下方颜色参数）"],
    [1, "1 · 绿字·深描边（OSD 经典）"],
    [2, "2 · 红字·深描边（REC 经典）"],
    [3, "3 · 白字·深描边"],
    [4, "4 · 黄字·深描边"],
    [5, "5 · 青字·深描边"],
  ],
  "overlay.halign": [
    [0, "0 · 左对齐"],
    [1, "1 · 水平居中"],
    [2, "2 · 右对齐"],
  ],
  "overlay.valign": [
    [0, "0 · 顶部对齐"],
    [1, "1 · 垂直居中"],
    [2, "2 · 底部对齐"],
  ],
  "overlay.pos": [
    [0, "0 · 左上"],
    [1, "1 · 上中"],
    [2, "2 · 右上"],
    [3, "3 · 左中"],
    [4, "4 · 正中"],
    [5, "5 · 右中"],
    [6, "6 · 左下"],
    [7, "7 · 下中（字幕常用）"],
    [8, "8 · 右下（录像机常用）"],
  ],
};

function selectorOptionsFor(layer, sp) {
  const fullKey = `${layer}.${sp.key}`;
  if (fullKey === "overlay.font") {
    if (window.SessionUI && typeof SessionUI.getOsdFonts === "function") {
      return SessionUI.getOsdFonts().map((f, i) => [i, `${i} · ${f[1]}`]);
    }
    if (S.catalog && Array.isArray(S.catalog.fonts)) {
      return S.catalog.fonts.filter((f) => f.available !== false).map((f, i) => [i, `${i} · ${f.label}`]);
    }
  }
  return SELECTORS[fullKey] || null;
}

function formatReadout(sp, v) {
  if (sp.kind === "text") return String(v ?? "");
  const isBool = sp.unit === "bool" || (sp.step === 1 && sp.min === 0 && sp.max === 1 && sp.unit === "0/1");
  if (isBool) return v > 0.5 ? "开" : "关";
  if (sp.step === 1) return Math.round(v).toFixed(0);
  return fmt(v);
}

function paramRow(sp) {
  const layerName = S.layer;
  const cur = (S.params[layerName] || {})[sp.key];
  const baseVal = cur === undefined ? sp.default : cur;
  const kfInfo = (sp.kind !== "text" && window.SessionUI && typeof SessionUI.trackInfo === "function")
    ? SessionUI.trackInfo(layerName, sp.key, S.t)
    : { hasTrack: false, atPoint: false, pointCount: 0, value: null };
  const val = (kfInfo.hasTrack && kfInfo.value !== null) ? kfInfo.value : baseVal;

  const row = document.createElement("div");
  row.className = "prow" + (kfInfo.hasTrack ? " prow-kf" : "");

  const head = document.createElement("div");
  head.className = "plabel";
  const name = document.createElement("span");
  name.textContent = sp.label + (sp.unit && sp.unit !== "0-1" && sp.unit !== "bool" && sp.unit !== "text" ? ` (${sp.unit})` : "");

  const rightBox = document.createElement("span");
  rightBox.className = "pright";

  const vv = document.createElement("span");
  vv.className = "v";
  const inPhys = sp.phys_min === null || sp.phys_max === null || sp.kind === "text"
    || (val >= sp.phys_min - 1e-9 && val <= sp.phys_max + 1e-9);
  if (!inPhys) vv.classList.add("artistic");
  vv.textContent = formatReadout(sp, val);
  rightBox.appendChild(vv);

  // 为所有可调节数值参数添加关键帧控制按钮（◆ 打关键帧 / 跳上一帧 / 跳下一帧 / 打开曲线）
  if (sp.kind !== "text" && window.SessionUI && typeof SessionUI.upsertKeyframe === "function") {
    if (kfInfo.hasTrack) {
      const prevBtn = document.createElement("button");
      prevBtn.className = "kfNav";
      prevBtn.textContent = "‹";
      prevBtn.title = "跳到上一个关键帧";
      prevBtn.onclick = (ev) => {
        ev.stopPropagation();
        const tTarget = SessionUI.stepKeyframe(layerName, sp.key, -1);
        if (tTarget != null) {
          S.t = tTarget;
          if ($("timeSlider")) $("timeSlider").value = String(tTarget);
          if ($("timeLabel")) $("timeLabel").textContent = `t = ${tTarget.toFixed(2)} s`;
          renderParams();
          refreshPreview();
        }
      };
      const nextBtn = document.createElement("button");
      nextBtn.className = "kfNav";
      nextBtn.textContent = "›";
      nextBtn.title = "跳到下一个关键帧";
      nextBtn.onclick = (ev) => {
        ev.stopPropagation();
        const tTarget = SessionUI.stepKeyframe(layerName, sp.key, 1);
        if (tTarget != null) {
          S.t = tTarget;
          if ($("timeSlider")) $("timeSlider").value = String(tTarget);
          if ($("timeLabel")) $("timeLabel").textContent = `t = ${tTarget.toFixed(2)} s`;
          renderParams();
          refreshPreview();
        }
      };
      rightBox.append(prevBtn, nextBtn);
    }

    const kfBtn = document.createElement("button");
    kfBtn.className = "kfBtn" + (kfInfo.hasTrack ? (kfInfo.atPoint ? " on at" : " on") : "");
    kfBtn.textContent = "◆";
    kfBtn.title = kfInfo.hasTrack
      ? (kfInfo.atPoint
          ? `当前时刻 (t=${S.t.toFixed(2)}s) 已打关键帧（点击删除该点，Shift+点击删除整条轨道，双击打开曲线面板）`
          : `已有 ${kfInfo.pointCount} 个关键帧（点击在 t=${S.t.toFixed(2)}s 插入关键帧，Shift+点击删除整条轨道，双击打开曲线面板）`)
      : `在当前时刻 t=${S.t.toFixed(2)}s 为「${sp.label}」打关键帧`;
    kfBtn.onclick = (ev) => {
      ev.stopPropagation();
      if (ev.shiftKey && kfInfo.hasTrack) {
        SessionUI.removeKeyframe(layerName, sp.key, null, true);
        renderParams();
        return;
      }
      const curInputVal = r ? (sp.kind === "text" ? r.value : parseFloat(r.value)) : val;
      if (kfInfo.atPoint) {
        SessionUI.removeKeyframe(layerName, sp.key, S.t, false);
      } else {
        const defaultCurve = (sp.step === 1) ? "hold" : "linear";
        SessionUI.upsertKeyframe(layerName, sp.key, S.t, curInputVal, defaultCurve);
      }
      renderParams();
    };
    kfBtn.ondblclick = (ev) => {
      ev.stopPropagation();
      if (typeof SessionUI.focusTrack === "function") SessionUI.focusTrack(layerName, sp.key);
    };
    rightBox.appendChild(kfBtn);
  }

  head.append(name, rightBox);

  const opts = selectorOptionsFor(layerName, sp);
  let r;
  if (sp.kind === "text") {
    r = document.createElement("input");
    r.type = "text";
    r.className = "ssText";
    r.style.width = "100%";
    r.value = val == null ? "" : String(val);
    r.onchange = () => {
      S.params[layerName] = S.params[layerName] || {};
      S.params[layerName][sp.key] = r.value;
      setProject({ params: { [layerName]: { [sp.key]: r.value } } });
    };
  } else if (opts) {
    r = document.createElement("select");
    r.className = "psel";
    for (const [v, label] of opts) {
      const o = document.createElement("option");
      o.value = String(v);
      o.textContent = label;
      if (Math.abs(v - Math.round(val)) < 1e-6) o.selected = true;
      r.appendChild(o);
    }
    vv.textContent = formatReadout(sp, val);
    r.onchange = () => {
      const nv = parseFloat(r.value);
      vv.textContent = formatReadout(sp, nv);
      if (kfInfo.hasTrack && window.SessionUI) {
        SessionUI.upsertKeyframe(layerName, sp.key, S.t, nv, "hold");
      }
      S.params[layerName] = S.params[layerName] || {};
      S.params[layerName][sp.key] = nv;
      setProject({ params: { [layerName]: { [sp.key]: nv } } });
    };
  } else {
    r = document.createElement("input");
    r.type = "range";
    r.min = sp.min; r.max = sp.max;
    r.step = sp.step === 1 ? 1 : (sp.max - sp.min) / 200;
    r.value = val;
    r.oninput = () => {
      const nv = parseFloat(r.value);
      vv.textContent = formatReadout(sp, nv);
      const stillPhys = sp.phys_min === null || sp.phys_max === null
        || (nv >= sp.phys_min - 1e-9 && nv <= sp.phys_max + 1e-9);
      vv.classList.toggle("artistic", !stillPhys);
      setParam(layerName, sp.key, nv, true);
    };
    r.onchange = () => setParam(layerName, sp.key, parseFloat(r.value), false);
  }

  row.append(head, r);
  if (sp.help) {
    const p = document.createElement("div");
    p.className = "phelp";
    p.textContent = sp.help;
    row.appendChild(p);
  }
  if (sp.phys_min !== null && sp.phys_max !== null) {
    const s = document.createElement("div");
    s.className = "src";
    s.textContent = `物理区间 ${fmt(sp.phys_min)}–${fmt(sp.phys_max)} · 来源：${sp.source}`;
    row.appendChild(s);
  }
  return row;
}

/* --------------------------------------------------------------- actions */
let saveTimer = null;
function setParam(layer, key, value, debounce) {
  S.params[layer] = S.params[layer] || {};
  S.params[layer][key] = value;
  const hasKf = window.SessionUI && typeof SessionUI.trackInfo === "function"
    && SessionUI.trackInfo(layer, key, S.t).hasTrack;
  if (!S.auto) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (hasKf) {
      SessionUI.upsertKeyframe(layer, key, S.t, value);
    } else {
      setProject({ params: { [layer]: { [key]: value } } });
    }
  }, debounce ? 130 : 0);
}

async function setProject(payload) {
  try {
    applyProject(await api("/api/project", jsonPost(payload)));
    renderParams();
    refreshPreview();
    loadTimelineMeta();
  } catch (e) { toast(e.message); }
}

async function applyPreset(key) {
  try {
    applyProject(await api("/api/preset", jsonPost({ key })));
    renderParams();
    renderPresetList();
    refreshPreview();
    loadTimelineMeta();
  } catch (e) { toast(e.message); }
}

async function uploadClip(ev) {
  const f = ev.target.files[0];
  if (!f) return;
  $("badge").textContent = `导入中… ${(f.size / 1048576).toFixed(1)} MB`;
  try {
    const r = await fetch("/api/upload", {
      method: "POST", headers: { "X-Filename": encodeURIComponent(f.name) }, body: f });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    applyProject(j.project);
    refreshPreview();
  } catch (e) { toast("导入失败：" + e.message); }
}

/**
 * Export the frame at the current time cursor as a full-resolution still.
 *
 * Rendered at scale 1.0, not the preview scale: the point of exporting a still is
 * to have the real output, and the preview's 0.5 proxy would quietly halve the
 * resolution of anything the user files away.
 */
const EXPORT_PROFILES = {
  lossless: { format: "png", q: 100, crf: 12, preset: "slow", label: "极高画质" },
  high:     { format: "png", q: 95,  crf: 16, preset: "medium", label: "标准高画质" },
  medium:   { format: "jpg", q: 92,  crf: 20, preset: "medium", label: "均衡画质" },
  draft:    { format: "jpg", q: 82,  crf: 24, preset: "fast", label: "快速导出" },
};

async function exportStill() {
  const btn = $("btnStill");
  const was = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "导出中…"; }
  try {
    const prof = EXPORT_PROFILES[S.exportQuality] || EXPORT_PROFILES.high;
    const sc = encodeURIComponent(S.exportScale || "1.0");
    const url = `/api/still?t=${S.t}&scale=${sc}&format=${prof.format}&q=${prof.q}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const ext = prof.format === "jpg" ? "jpg" : "png";
    a.download = `atsim_${S.t.toFixed(2).replace(".", "_")}s.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    toast("已导出静帧 " + a.download);
  } catch (e) {
    toast("导出失败：" + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = was || "导出静帧"; }
  }
}

/* --------------------------------------------------------------- preview */
async function refreshPreview() {
  if (S.busy) { S.pending = true; return; }
  S.busy = true;
  const my = ++S.token;
  const started = performance.now();
  const engine = (typeof G !== "undefined" && G.perNode) ? "&engine=graph" : "";
  const pScale = encodeURIComponent(S.previewScale || "0.5");
  const pQual = S.previewQuality || 88;
  try {
    const blob = await api(`/api/frame?t=${S.t}&scale=${pScale}&q=${pQual}&_=${my}${engine}`);
    if (my !== S.token) return;
    if (S.frameUrl) URL.revokeObjectURL(S.frameUrl);
    S.frameUrl = URL.createObjectURL(blob);
    const pimg = $("previewImg");
    pimg.onload = () => {
      S.frameW = pimg.naturalWidth; S.frameH = pimg.naturalHeight;
      syncAB();
    };
    pimg.src = S.frameUrl;
    // The A/B reference must follow the time cursor.  It used to be fetched once
    // only (guarded by `!S.origUrl`), so after the first load the right-hand side
    // sat frozen on an old frame while the simulation advanced -- the two sides
    // then showed *different content*, which reads as "the comparison is broken"
    // rather than as a stale overlay.
    if (S.ab) await loadOriginal();
    const ms = performance.now() - started;
    const via = (typeof G !== "undefined" && G.perNode) ? "节点图" : "固定链路";
    $("badge").textContent =
      `${via} · 预览 ${S.frameW || "?"}×${S.frameH || "?"} · ` +
      `${ms.toFixed(0)} ms/帧 · t=${S.t.toFixed(2)}s`;
    const tl = await api("/api/timeline");
    paintTimeline(tl);
    if (typeof refreshGraphPerf === "function") await refreshGraphPerf();
  } catch (e) {
    // After the boot handler has replaced the body, `badge` no longer exists --
    // reporting the failure must not itself throw, or the original error is lost.
    const b = $("badge");
    if (b) b.textContent = "错误：" + e.message.slice(0, 120);
    else console.error("refreshPreview failed:", e);
  } finally {
    S.busy = false;
    if (S.pending) { S.pending = false; refreshPreview(); }
  }
}

async function loadOriginal() {
  // the original is the simulation with camera/format/tape/player neutralised,
  // so the comparison is honest about what the chain did
  const pScale = encodeURIComponent(S.previewScale || "0.5");
  const pQual = S.previewQuality || 88;
  try {
    const blob = await api(`/api/frame?t=${S.t}&scale=${pScale}&q=${pQual}&mode=source`);
    // release the previous object URL, or every time-cursor move leaks one
    if (S.origUrl) URL.revokeObjectURL(S.origUrl);
    S.origUrl = URL.createObjectURL(blob);
    const oimg = $("originalImg");
    oimg.onload = () => syncAB();
    oimg.src = S.origUrl;
  } catch (e) { /* the A/B overlay is optional */ }
}

function paintTimeline(tl) {
  const track = $("tlTrack");
  track.innerHTML = "";
  const stages = tl.stages || [];
  stages.forEach((s, i) => {
    const seg = document.createElement("div");
    seg.className = "tl-seg";
    seg.style.left = (i / stages.length * 100) + "%";
    seg.style.width = (100 / stages.length) + "%";
    seg.style.background = s.dirty ? "rgba(232,163,61,.35)" : "rgba(127,191,106,.22)";
    seg.style.borderRight = "1px solid var(--line)";
    seg.title = `${s.label} · 缓存 ${s.cached} 帧${s.dirty ? " · 需重算" : ""}`;
    seg.textContent = s.label;
    seg.style.fontSize = "10px";
    seg.style.paddingLeft = "6px";
    seg.style.lineHeight = "16px";
    seg.style.color = "#cbbfae";
    track.appendChild(seg);
  });
  renderChainBar(stages);

  const def = $("tlDefects");
  def.innerHTML = "";
  const dens = ((tl.defects || []).find((d) => d.kind === "tape_plan") || {}).dropout_density || [];
  const max = Math.max(1, ...dens.map((d) => d.count));
  const total = Math.max(1, (tl.defects || []).find((d) => d.kind === "tape_plan")?.prepared_frames || 1);
  dens.forEach((d) => {
    const seg = document.createElement("div");
    seg.className = "tl-seg";
    const x = (d.second / (total / (tl.fps || 30))) * 100;
    seg.style.left = Math.min(99, x) + "%";
    seg.style.width = "2px";
    seg.style.background = "#d9614c";
    seg.style.opacity = 0.35 + 0.65 * (d.count / max);
    seg.title = `第 ${d.second} 秒：${d.count} 次失落`;
    def.appendChild(seg);
  });
  const head = document.createElement("div");
  head.className = "tl-tick";
  head.style.left = (S.t / (total / (tl.fps || 30)) * 100) + "%";
  def.appendChild(head);
}

function renderChainBar(stages) {
  const bar = $("chainbar");
  bar.innerHTML = "";
  (stages || []).forEach((s) => {
    const d = document.createElement("div");
    d.className = "chainnode" + (s.dirty ? " dirty" : " cached");
    d.innerHTML = `<span class="nm"></span><span class="cnt"></span>`;
    d.querySelector(".nm").textContent = s.label;
    d.querySelector(".cnt").textContent = s.dirty ? "待重算" : `${s.cached} 帧`;
    d.title = `参数哈希 ${s.hash} · ${s.dirty ? "参数已变，需要重算" : "已缓存"}`;
    bar.appendChild(d);
  });
}

/* --------------------------------------------------------------- render */
let jobTimer = null;
async function startRender() {
  const prof = EXPORT_PROFILES[S.exportQuality] || EXPORT_PROFILES.high;
  const sc = encodeURIComponent(S.exportScale || "1.0");
  $("jobbar").classList.remove("hidden");
  $("jobDownload").classList.add("hidden");
  $("jobFill").style.width = "0%";
  $("jobText").textContent = `启动渲染 (${S.exportScale || "1.0"} · CRF ${prof.crf})…`;
  try {
    const { job } = await api(`/api/render?scale=${sc}&crf=${prof.crf}&preset=${prof.preset}`);
    clearInterval(jobTimer);
    jobTimer = setInterval(() => pollJob(job), 700);
    pollJob(job);
  } catch (e) { $("jobText").textContent = "失败：" + e.message; }
}

async function pollJob(id) {
  try {
    const j = await api("/api/job?id=" + id);
    $("jobFill").style.width = ((j.progress || 0) * 100).toFixed(1) + "%";
    $("jobText").textContent = j.message || j.state;
    if (j.state === "done") {
      clearInterval(jobTimer);
      const a = $("jobDownload");
      a.href = "/api/download?id=" + id;
      a.classList.remove("hidden");
    } else if (j.state === "error") {
      clearInterval(jobTimer);
      $("jobText").innerHTML = `<span class="err">${(j.message || "").slice(0, 200)}</span>`;
    }
  } catch (e) { /* keep polling */ }
}

/* --------------------------------------------------------------- misc */
async function loadTimelineMeta() { /* defect strip refreshes with the frame */ }

async function showTimeline() {
  const tl = await api("/api/timeline");
  const plan = (tl.defects || []).find((d) => d.kind === "tape_plan") || {};
  $("tlSeed").textContent = `seed = ${tl.seed}`;
  const rows = (plan.dropout_events || []).slice(0, 400);
  $("tlBody").innerHTML = `
    <p class="note">共 ${plan.dropout_count || 0} 次失落事件，${(plan.creases || []).length} 条折痕；
    下方列出前 ${rows.length} 条。</p>
    <table><thead><tr><th>开始 (s)</th><th>持续 (s)</th><th>行</th><th>高度</th>
    <th>列</th><th>宽度</th><th>场</th><th>严重度</th></tr></thead><tbody>
    ${rows.map((e) => `<tr><td>${e.start_s}</td><td>${e.duration_s}</td><td>${e.row}</td>
      <td>${e.rows}</td><td>${e.col}</td><td>${e.cols}</td><td>${e.field}</td>
      <td>${e.severity}</td></tr>`).join("")}
    </tbody></table>`;
  $("modal").classList.remove("hidden");
}

function toast(msg) {
  const b = $("badge");
  b.textContent = msg;
  b.style.color = "#d9614c";
  setTimeout(() => { b.style.color = ""; }, 2500);
}

boot().catch((e) => {
  // Log the real stack before replacing the DOM: a bare message here hides
  // which line actually failed, which cost real debugging time once already.
  console.error("BOOT FAILED:", e, e && e.stack);
  document.body.innerHTML =
    `<pre class="err" style="padding:24px">启动失败：${e.message}\n\n${(e.stack || "").split("\n").slice(0, 8).join("\n")}</pre>`;
});
