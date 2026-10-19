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
  syncWTBar();
  if (window.SessionUI) SessionUI.syncFromProject(d);
}

function updateWTReadout(zVal) {
  const cam = (S.params && S.params.camera) || {};
  const z = Math.min(1, Math.max(0, zVal !== undefined ? zVal : (cam.wt_zoom || 0)));
  const maxZ = cam.wt_max_zoom || 4.0;
  const mag = Math.pow(maxZ, z);
  const fnum = 1.4 + (2.9 - 1.4) * Math.pow(z, 1.15);
  const state = z < 0.15 ? "W 端光晕态" : (z < 0.55 ? "中焦过渡态" : "T 端具体光源细节");
  const rd = $("wtReadout");
  if (rd) rd.textContent = `${mag.toFixed(2)}× · f/${fnum.toFixed(1)} · ${state}`;
}

function syncWTBar() {
  const cam = (S.params && S.params.camera) || {};
  const sl = $("wtZoomSlider");
  if (sl && document.activeElement !== sl) sl.value = cam.wt_zoom !== undefined ? cam.wt_zoom : 0;
  const dm = $("wtDynMode");
  if (dm) dm.value = String(Math.round(cam.zoom_dynamic_mode || 0));
  const osd = $("wtOsdToggle");
  if (osd) osd.checked = (cam.zoom_osd || 0) > 0.5;
  updateWTReadout(cam.wt_zoom || 0);
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
  wireWTBar();
  initDivider();
  layoutAB();
  initAudioMonitor();
}

function wireWTBar() {
  const sl = $("wtZoomSlider");
  const btnW = $("btnZoomW");
  const btnT = $("btnZoomT");
  const dm = $("wtDynMode");
  const osd = $("wtOsdToggle");
  if (!sl) return;

  const applyZ = (nv, debounce) => {
    const z = Math.min(1, Math.max(0, parseFloat(nv) || 0));
    sl.value = z;
    S.params.camera = S.params.camera || {};
    S.params.camera.wt_zoom = z;
    updateWTReadout(z);
    if (S.layer === "camera") renderParams();
    setParam("camera", "wt_zoom", z, debounce);
  };

  sl.oninput = () => applyZ(sl.value, true);
  sl.onchange = () => applyZ(sl.value, false);

  let rockerTimer = null;
  const startRocker = (dir) => {
    clearInterval(rockerTimer);
    const stepOnce = () => {
      const cur = parseFloat(sl.value) || 0;
      const next = Math.min(1, Math.max(0, cur + dir * 0.04));
      applyZ(next, true);
      if (next <= 0 || next >= 1) stopRocker();
    };
    stepOnce();
    rockerTimer = setInterval(stepOnce, 75);
  };
  const stopRocker = () => {
    if (rockerTimer) {
      clearInterval(rockerTimer);
      rockerTimer = null;
      applyZ(sl.value, false);
    }
  };
  if (btnW) {
    btnW.onmousedown = () => startRocker(-1);
    btnW.ontouchstart = (e) => { e.preventDefault(); startRocker(-1); };
  }
  if (btnT) {
    btnT.onmousedown = () => startRocker(1);
    btnT.ontouchstart = (e) => { e.preventDefault(); startRocker(1); };
  }
  window.addEventListener("mouseup", stopRocker);
  window.addEventListener("touchend", stopRocker);

  if (dm) {
    dm.onchange = () => {
      const v = parseFloat(dm.value) || 0;
      S.params.camera = S.params.camera || {};
      S.params.camera.zoom_dynamic_mode = v;
      setProject({ params: { camera: { zoom_dynamic_mode: v } } });
    };
  }
  if (osd) {
    osd.onchange = () => {
      const v = osd.checked ? 1.0 : 0.0;
      S.params.camera = S.params.camera || {};
      S.params.camera.zoom_osd = v;
      setProject({ params: { camera: { zoom_osd: v } } });
    };
  }
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
  const tabs = [["factory", "整套配置"], ["custom", "自定义预设"], ["media", "记录媒介"],
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
    return Object.entries(S.catalog.factory || {}).map(([k, v]) => [k, v.label, v.note || "整套：摄像机+格式+介质+播放（默认无CRT）"]);
  }
  if (S.presetTab === "custom") {
    return Object.entries(S.catalog.custom || {}).map(([k, v]) => [k, v.label, v.note || "用户自定义保存预设"]);
  }
  const g = (S.catalog.profiles || {})[S.presetTab] || {};
  return Object.entries(g).map(([k, v]) => [k, v.label, v.note || ""]);
}

async function saveCustomPreset(labelInput, noteInput) {
  const label = (labelInput ? labelInput.value : "").trim() || prompt("请输入自定义预设名称：", S.project?.name || "我的 VHS 预设");
  if (!label) return;
  const note = (noteInput ? noteInput.value : "").trim();
  try {
    const res = await api("/api/preset/save", jsonPost({ label, note }));
    if (res.catalog) S.catalog = res.catalog;
    if (res.project) applyProject(res.project);
    S.presetTab = "custom";
    buildPresetTabs();
    renderPresetList();
    toast(`已保存自定义预设「${label}」`);
  } catch (e) {
    toast("保存预设失败：" + e.message);
  }
}

async function deleteCustomPreset(key, label, ev) {
  if (ev) ev.stopPropagation();
  try {
    const res = await api("/api/preset/delete", jsonPost({ key }));
    if (res.catalog) S.catalog = res.catalog;
    if (res.project) applyProject(res.project);
    renderPresetList();
    toast(`已删除预设「${label}」`);
  } catch (e) {
    toast("删除预设失败：" + e.message);
  }
}

function renderPresetList() {
  const box = $("presetList");
  box.innerHTML = "";

  // Inline bar to save current parameters + node graph as a user custom preset
  const saveBar = document.createElement("div");
  saveBar.className = "custom-preset-bar";
  saveBar.innerHTML = `
    <input type="text" id="customPresetName" class="cp-input" placeholder="输入名称保存当前参数为预设…" />
    <button type="button" id="btnSaveCustomPreset" class="cp-btn">+ 保存预设</button>
  `;
  const nameIn = saveBar.querySelector("#customPresetName");
  const saveBtn = saveBar.querySelector("#btnSaveCustomPreset");
  saveBtn.onclick = () => saveCustomPreset(nameIn, null);
  nameIn.onkeydown = (e) => { if (e.key === "Enter") saveCustomPreset(nameIn, null); };
  box.appendChild(saveBar);

  const entries = presetEntries();
  if (!entries.length && S.presetTab === "custom") {
    const empty = document.createElement("div");
    empty.className = "phelp";
    empty.style.padding = "8px 4px";
    empty.textContent = "暂无自定义预设。调节任意参数或节点连线后，在上方输入名称点击「+ 保存预设」即可永久保存。";
    box.appendChild(empty);
    return;
  }

  entries.forEach(([key, label, note]) => {
    const d = document.createElement("div");
    d.className = "card" + (S.project.preset === key ? " on" : "");
    d.innerHTML = `<div class="card-head"><div class="t"></div></div><div class="n"></div>`;
    d.querySelector(".t").textContent = label;
    d.querySelector(".n").textContent = note;
    if (key.startsWith("custom_") || S.presetTab === "custom") {
      const del = document.createElement("button");
      del.className = "cp-del";
      del.title = "删除此自定义预设";
      del.textContent = "×";
      del.onclick = (ev) => deleteCustomPreset(key, label, ev);
      d.querySelector(".card-head").appendChild(del);
    }
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
  "camera.zoom_dynamic_mode": [
    [0, "手动固定 W–T 位置"],
    [1, "W→T 电动推进放大（光晕→具体光源）"],
    [2, "T→W 电动拉远缩小（具体光源→光晕）"],
    [3, "W↔T 往复变焦测试"],
  ],
  "camera.nv_mode": [
    [0, "关"],
    [1, "NightShot（红外 + 绿单色）"],
    [2, "Super NightShot（帧积累）"],
    [3, "Color Slow Shutter（慢快门保留彩色）"],
  ],
  "display.mask_type": [
    [0, "荫栅 Aperture Grille（特丽珑连续竖条 + 阻尼丝）"],
    [1, "槽孔荫罩 Slot Mask（家用彩电交错砖形槽）"],
    [2, "点状荫罩 Dot Triad（球面管三角圆点阵）"],
  ],
  "rf.tv_standard": [
    [0, "PAL-D/K（中国大陆 8MHz / +6.5MHz 伴音）"],
    [1, "PAL-I（港澳珠三角 8MHz / +6.0MHz 伴音）"],
    [2, "NTSC-M（6MHz / +4.5MHz 伴音）"],
  ],
  "rf.propagation_mode": [
    [0, "本地视距传播 (LOS)"],
    [1, "常规超视距弱信号 (Troposcatter)"],
    [2, "TV-DX 大气波导现象 (Atmospheric Ducting 异常清晰)"],
    [3, "电离层 Es 突发层反射 (Sporadic-E Skywave)"],
  ],
  "rf.tuner_type": [
    [0, "传统铁壳高频头 (Can Tuner)"],
    [1, "淘宝廉价电视卡硅调谐器 (SAA713x / TDA9887)"],
    [2, "高选择性 DX 接收机 (High-Q DX Tuner)"],
  ],
  "audio.mode": [
    [0, "自动（跟随磁带制式）"],
    [1, "VHS 线性磁迹"],
    [2, "VHS Hi-Fi 调频深层记录"],
    [3, "8mm / Hi8 AFM 调频伴音"],
    [4, "MiniDV PCM 数字音频"],
    [5, "机内麦克风直录"],
    [6, "RF 射频潮湿受潮回路伴音"],
    [7, "TV-DX 电视卡 FM 伴音解调（含静噪门限与波导呼吸）"],
  ],
  "overlay.style": [
    [0, "0 · 自定义（使用上方RGB参数）"],
    [1, "1 · 绿字黑边（OSD 经典）"],
    [2, "2 · 红字黑边（REC 经典）"],
    [3, "3 · 白字黑边"],
    [4, "4 · 黄字黑边"],
    [5, "5 · 青字黑边"],
    [6, "6 · 80年代纪录片·金黄标宋立体字"],
    [7, "7 · 80年代纪录片·象牙白光学叠印字"],
    [8, "8 · 广播级字幕机·琥珀黄滚屏字"],
    [9, "9 · 80年代专题片·深红描金标题字"],
  ],
  "overlay.anim_mode": [
    [0, "0 · 静态常显"],
    [1, "1 · 逐字打字机显字"],
    [2, "2 · 逐行顺序显字"],
    [3, "3 · 水平光栅扫描擦除"],
    [4, "4 · 百叶窗隔行扫描显字"],
    [5, "5 · 纵向滚屏（片尾字幕）"],
    [6, "6 · 光学叠化淡入淡出"],
    [7, "7 · 横向滑入定格"],
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
    refreshAudioTimeline();
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


/* --------------------------------------------------------------- audio monitor & A/V sync scope */
let _audioTl = null;
let _audioRaf = null;
let _audioWinStart = 0.0;
let _audioWinDur = 3.0;

function initAudioMonitor() {
  const btn = $("btnAudioPlay");
  const player = $("audioPlayer");
  const scope = $("audioScope");
  if (!btn || !player || !scope) return;

  const btnWin = $("btnAudioExportWin");
  if (btnWin) {
    btnWin.onclick = async () => {
      const clean = $("audioClean") && $("audioClean").checked ? "original" : "sim";
      const start = Math.max(0, Math.floor(S.t / 1.5) * 1.5);
      const orig = btnWin.textContent;
      btnWin.disabled = true;
      btnWin.textContent = "导出中…";
      try {
        const r = await fetch(`/api/audio?t=${start.toFixed(3)}&dur=${_audioWinDur}&mode=${clean}&_=${Date.now()}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const blob = await r.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `atsim_audio_${start.toFixed(1).replace(".", "_")}s_${_audioWinDur}s.wav`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        toast("已导出试听段伴音 " + a.download);
      } catch (e) {
        toast("伴音导出失败：" + e.message);
      } finally {
        btnWin.disabled = false;
        btnWin.textContent = orig;
      }
    };
  }
  const btnFull = $("btnAudioExportFull");
  if (btnFull) {
    btnFull.onclick = async () => {
      const clean = $("audioClean") && $("audioClean").checked ? "original" : "sim";
      const orig = btnFull.textContent;
      btnFull.disabled = true;
      btnFull.textContent = "合成中…";
      try {
        const info = (S.project && S.project.source_info) || {};
        const dur = Math.min(30.0, Math.max(3.0, parseFloat(info.duration) || 6.0));
        const r = await fetch(`/api/audio?t=0&dur=${dur.toFixed(3)}&mode=${clean}&_=${Date.now()}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const blob = await r.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `atsim_audio_full_${dur.toFixed(1).replace(".", "_")}s.wav`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        toast("已导出全片伴音 " + a.download);
      } catch (e) {
        toast("全片伴音导出失败：" + e.message);
      } finally {
        btnFull.disabled = false;
        btnFull.textContent = orig;
      }
    };
  }

  btn.onclick = () => {
    if (!player.paused) {
      player.pause();
      return;
    }
    const clean = $("audioClean") && $("audioClean").checked ? "original" : "sim";
    _audioWinDur = 3.0;
    // Align preview start time with refreshAudioTimeline & exported window
    _audioWinStart = Math.max(0, Math.floor(S.t / 1.5) * 1.5);
    player.src = `/api/audio?t=${_audioWinStart.toFixed(3)}&dur=${_audioWinDur}&mode=${clean}&_=${Date.now()}`;
    btn.textContent = "加载伴音…";
    player.play().then(() => {
      btn.textContent = "⏹ 停止试听";
      pumpAudioSyncCursor();
    }).catch((err) => {
      btn.textContent = "▶ 试听伴音 (3s)";
      toast("音频播放失败：" + err.message);
    });
  };

  player.onended = () => {
    btn.textContent = "▶ 试听伴音 (3s)";
    if (_audioRaf) cancelAnimationFrame(_audioRaf);
    drawAudioScope();
  };
  player.onpause = () => {
    btn.textContent = "▶ 试听伴音 (3s)";
    if (_audioRaf) cancelAnimationFrame(_audioRaf);
    drawAudioScope();
  };

  scope.onclick = (ev) => {
    const r = scope.getBoundingClientRect();
    const frac = Math.min(Math.max((ev.clientX - r.left) / Math.max(1, r.width), 0), 1);
    const tNew = _audioWinStart + frac * _audioWinDur;
    S.t = Math.max(0, Math.min(10, tNew));
    const slider = $("timeSlider");
    if (slider) slider.value = S.t;
    const lbl = $("timeLabel");
    if (lbl) lbl.textContent = `t = ${S.t.toFixed(2)} s`;
    refreshPreview();
  };
}

function pumpAudioSyncCursor() {
  const player = $("audioPlayer");
  if (!player || player.paused) return;
  const curT = _audioWinStart + (player.currentTime || 0);
  drawAudioScope(curT);
  if ($("audioSyncPlay") && $("audioSyncPlay").checked && !S.busy) {
    if (Math.abs(curT - S.t) >= (1.0 / ((_audioTl && _audioTl.fps) || 29.97))) {
      S.t = Math.min(10.0, curT);
      const slider = $("timeSlider");
      if (slider) slider.value = S.t;
      const lbl = $("timeLabel");
      if (lbl) lbl.textContent = `t = ${S.t.toFixed(2)} s`;
      refreshPreview();
    }
  }
  _audioRaf = requestAnimationFrame(pumpAudioSyncCursor);
}

async function refreshAudioTimeline() {
  const scope = $("audioScope");
  if (!scope) return;
  try {
    _audioWinDur = 3.0;
    const start = Math.max(0, Math.floor(S.t / 1.5) * 1.5);
    _audioWinStart = start;
    _audioTl = await api(`/api/audio_timeline?t=${start.toFixed(2)}&dur=${_audioWinDur}`);
    drawAudioScope(S.t);
  } catch (e) {
    console.warn("audio_timeline error:", e);
  }
}

function drawAudioScope(cursorT) {
  const scope = $("audioScope");
  if (!scope || !_audioTl) return;
  const ctx = scope.getContext("2d");
  const w = scope.width, h = scope.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "#100e0c";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#26211c";
  ctx.beginPath();
  ctx.moveTo(0, 11); ctx.lineTo(w, 11);
  ctx.moveTo(0, 22); ctx.lineTo(w, 22);
  ctx.stroke();

  const wf = _audioTl.waveform || [];
  if (wf.length > 0) {
    ctx.strokeStyle = "#7fbf6a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < wf.length; i++) {
      const x = (i / wf.length) * w;
      const pk = Math.min(1.0, wf[i].peak || 0);
      const y0 = 11 - pk * 10;
      const y1 = 11 + pk * 10;
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
    }
    ctx.stroke();
  }

  const frames = _audioTl.frames || [];
  let curFrameFault = null;
  const tRef = cursorT !== undefined ? cursorT : S.t;
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const x0 = (i / Math.max(1, frames.length)) * w;
    const bw = Math.max(2, w / Math.max(1, frames.length));
    if (Math.abs((f.time_s || 0) - tRef) < (0.65 / (_audioTl.fps || 30))) {
      curFrameFault = f;
    }
    if ((f.dropout || 0) > 0.02 || f.afm_unlock) {
      ctx.fillStyle = f.afm_unlock ? "rgba(217, 97, 76, 0.92)" : "rgba(217, 97, 76, 0.55)";
      ctx.fillRect(x0, 23, bw, 5);
    }
    if ((f.head_switch_buzz || 0) > 0.05) {
      ctx.fillStyle = "rgba(232, 163, 61, 0.75)";
      ctx.fillRect(x0, 28, bw, 3);
    }
    if (Math.abs(f.pitch_cents || 0) > 4.0) {
      ctx.fillStyle = "rgba(104, 168, 214, 0.78)";
      ctx.fillRect(x0, 31, bw, 3);
    }
    if ((f.hum_buzz || 0) > 0.02 || f.hdmi_drop) {
      ctx.fillStyle = "rgba(186, 120, 214, 0.80)";
      ctx.fillRect(x0, 23, bw, 11);
    }
  }

  const frac = ((tRef - _audioWinStart) / Math.max(0.001, _audioWinDur));
  if (frac >= 0 && frac <= 1) {
    const cx = frac * w;
    ctx.strokeStyle = "#f2c14e";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();
  }

  const badge = $("audioFaultBadge");
  if (badge) {
    const modeLabel = (_audioTl.summary && _audioTl.summary.mode) || "auto";
    if (curFrameFault) {
      const tags = [];
      if (curFrameFault.hdmi_drop) tags.push("HDMI断流静音");
      if (curFrameFault.afm_unlock) tags.push("AFM解锁→线性轨");
      else if ((curFrameFault.dropout || 0) > 0.03) tags.push(`失落 ${(curFrameFault.dropout * 100).toFixed(0)}%`);
      if (Math.abs(curFrameFault.pitch_cents || 0) >= 2.0) {
        const c = curFrameFault.pitch_cents;
        tags.push(`音高 ${c >= 0 ? "+" : ""}${c.toFixed(1)}c`);
      }
      if ((curFrameFault.head_switch_buzz || 0) > 0.1) tags.push("切头脉冲");
      if ((curFrameFault.hum_buzz || 0) > 0.02) tags.push("哼声/RF蜂音");
      badge.textContent = `${modeLabel} · ` + (tags.length ? tags.join(" | ") : "音画同步正常");
      badge.style.color = (curFrameFault.afm_unlock || curFrameFault.hdmi_drop) ? "#d9614c" : "";
    } else {
      badge.textContent = `${modeLabel} · t=${tRef.toFixed(2)}s`;
      badge.style.color = "";
    }
  }
}

boot().catch((e) => {
  // Log the real stack before replacing the DOM: a bare message here hides
  // which line actually failed, which cost real debugging time once already.
  console.error("BOOT FAILED:", e, e && e.stack);
  document.body.innerHTML =
    `<pre class="err" style="padding:24px">启动失败：${e.message}\n\n${(e.stack || "").split("\n").slice(0, 8).join("\n")}</pre>`;
});
