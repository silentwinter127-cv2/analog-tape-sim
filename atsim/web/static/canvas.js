/* analog-tape-sim -- node graph canvas
 *
 * The canvas is deliberately a *device* editor: nodes are pieces of equipment
 * and edges are cables.  Two things it must get right, because they are the
 * whole reason a graph beats a fixed chain:
 *
 *   1. ports are coloured by signal type, and a drop that would connect
 *      mismatched types is refused with the reason shown -- the backend does the
 *      authoritative check, the client just previews it;
 *   2. the dirty subtree is visible.  After any edit the backend reports which
 *      nodes must re-run (a node's hash covers all its ancestors), and those
 *      nodes are outlined, so it is obvious what an edit costs.
 */
"use strict";

const G = {
  state: null,          // last /api/graph response
  dragging: null,       // {name, port, type, x, y} while drawing a cable
  moving: null,         // node name while dragging it
  offset: [0, 0],
  selected: null,
  scale: 1,
  pan: [0, 0],
  panning: null,
  perNode: false,       // preview through the graph instead of the fixed chain
  dirty: new Set(),
  ms: {},
};

const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

/**
 * Convert a node's stored position into canvas pixels.
 *
 * Graph presets are written with *grid* coordinates (0, 1, 2 ... one step per
 * device) because that is readable in source, but the canvas needs pixels.
 * Without this conversion the whole chain spans 5 units, so fitting it to the
 * viewport produced a scale so small that the graph looked like one node.
 */
const NODE_W = 168, NODE_H = 54;
const COL_STEP = 224, ROW_STEP = 108;

function px(node) {
  /* Positions are pixels, and this is now a pass-through.
   *
   * It used to convert a grid column to pixels by multiplying, which meant asking "is this
   * number a column or a pixel" on every render -- and the answer for a node dragged near the
   * origin was wrong, so it was multiplied again and thrown across the canvas. The server
   * converts a preset's columns as it lays the graph out, so nothing arrives in grid units
   * and there is nothing to guess. A genuine pixel position of (5, 0) is left alone, which
   * the heuristic could not do: it would have read column 5 and moved the node to x=1120. */
  return [node.position ? node.position[0] : 0, node.position ? node.position[1] : 0];
}

const TYPE_COLOR = {
  rgb: "#e8a33d", composite: "#69b0c9", rf: "#c96fb0", dv: "#7fbf6a",
  yiq: "#c9b569", hdmi: "#9a8fd0", audio: "#8a8a8a", "?": "#666",
};

async function graphApi(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json();
  if (j.error) {
    flash(j.error);
    if (j.describe) applyGraphState(j);
    return null;
  }
  return j;
}

function flash(msg) {
  const b = document.getElementById("canvasMsg");
  b.textContent = msg;
  b.style.color = "#d9614c";
  clearTimeout(flash._t);
  flash._t = setTimeout(() => { b.style.color = ""; b.textContent = ""; }, 4000);
}

async function loadGraph() {
  const j = await graphApi("/api/graph");
  if (j) applyGraphState(j);
}

/* Convert every node to pixel coordinates, once, when a payload is adopted.
 *
 * This is the fix for nodes "running around" while being dragged.  `px()` converts a grid
 * position to pixels by multiplying, and it runs on every render -- so a node sitting at a
 * pixel position near the origin, x below 12 or y below 8, was read as a grid position on
 * the next render and multiplied by COL_STEP, which threw it across the canvas. Preset nodes
 * start at (0,0)..(5,0), so they were all in that range.
 *
 * No flag or threshold can fix that, because each new payload from the server re-creates the
 * ambiguity: the server stores grid units, so the question "is this number a column or a
 * pixel" is asked again every time. Normalising here means the canvas has exactly one unit
 * from then on, and the question is never asked again.
 *
 * Both lists are converted. `describe.nodes` is what the canvas draws and
 * `graph.nodes` is what `/api/graph/move` is called with, and they are separate copies --
 * converting one and not the other puts them back out of step on the next drag.
 */
function normalisePositions(j) {
  /* Kept as the single place a payload's positions could be converted, but it no longer
   * converts anything: the server does it.  Left in place rather than deleted so that the
   * call site in applyGraphState still marks where such a conversion would belong, and so
   * that a payload which somehow carries grid units can be fixed here rather than in a
   * scatter of heuristics. */
  if (!j) return j;
  for (const list of [j.describe && j.describe.nodes, j.graph && j.graph.nodes]) {
    if (!Array.isArray(list)) continue;
    for (const n of list) {
      if (Array.isArray(n.position)) n.px = true;
    }
  }
  return j;
}

function applyGraphState(j) {
  const first = !G.state;
  normalisePositions(j);
  G.state = j;
  renderPalette();
  renderGraphPresets();
  if (first) fitView();
  renderCanvas();
  renderIssues();
}

/* ------------------------------------------------------------- palette */
function renderPalette() {
  const box = document.getElementById("palette");
  if (!box) return;
  const groups = {};
  for (const n of G.state.catalog) (groups[n.group] ||= []).push(n);
  box.innerHTML = "";
  for (const [g, items] of Object.entries(groups)) {
    const h = document.createElement("div");
    h.className = "palGroup";
    h.textContent = g;
    box.appendChild(h);
    for (const n of items) {
      const d = document.createElement("div");
      d.className = "palItem";
      d.innerHTML = `<span class="palDot"></span><span class="palLabel"></span>`;
      d.querySelector(".palDot").style.background = typeColor(n.outputs[0]?.type);
      d.querySelector(".palLabel").textContent = n.label;
      d.title = `${n.device}\n${n.note}`;
      d.onclick = () => addNode(n.kind);
      box.appendChild(d);
    }
  }
}

function renderGraphPresets() {
  const box = document.getElementById("graphPresets");
  if (!box) return;
  box.innerHTML = "";
  for (const [k, v] of Object.entries(G.state.presets)) {
    const b = document.createElement("button");
    b.textContent = v.label;
    b.title = v.note;
    b.onclick = async () => {
      const j = await graphApi("/api/graph/preset", jsonPost({ key: k }));
      if (j) {
        applyGraphState(j);
        fitView();
        refreshPreview();
      }
    };
    box.appendChild(b);
  }
}

function typeColor(t) { return TYPE_COLOR[t] || TYPE_COLOR["?"]; }

/* -------------------------------------------------------------- canvas */
/**
 * Fit every node into the visible area.
 *
 * Without this, a graph preset whose nodes are laid out in a wide row starts
 * mostly off-screen and the canvas looks empty -- which is exactly how it looked
 * the first time it was verified.
 */
function fitView() {
  if (!G.state || !G.state.describe) return;
  const nodes = G.state.describe.nodes;
  if (!nodes.length) return;
  const NW = 168, NH = 54, PAD = 40;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of nodes) {
    const [nx, ny] = px(n);
    x0 = Math.min(x0, nx);
    y0 = Math.min(y0, ny);
    x1 = Math.max(x1, nx + NW);
    y1 = Math.max(y1, ny + NH);
  }
  const svg = document.getElementById("canvas");
  const w = svg.clientWidth || 900, h = svg.clientHeight || 500;
  const gw = Math.max(1, x1 - x0), gh = Math.max(1, y1 - y0);
  G.scale = Math.min(1.15, Math.max(0.3,
    Math.min((w - PAD * 2) / gw, (h - PAD * 2) / gh)));
  G.pan[0] = PAD - x0 * G.scale + Math.max(0, (w - PAD * 2 - gw * G.scale) / 2);
  G.pan[1] = PAD - y0 * G.scale + Math.max(0, (h - PAD * 2 - gh * G.scale) / 2);
}

function renderCanvas() {
  const svg = document.getElementById("canvas");
  if (!svg) return;
  // Guard: the canvas can be asked to draw before the graph has arrived (a
  // `#canvas` deep link, or a resize during boot).  Drawing nothing is correct
  // then; throwing a null dereference is not.
  if (!G.state || !G.state.describe) return;
  svg.innerHTML = "";
  const nodes = G.state.describe.nodes;
  const edges = G.state.describe.edges;
  const NW = 168, NH = 54;

  const defs = svgEl("defs");
  const marker = svgEl("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9",
    refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" });
  marker.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#8a7f72" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  const g = svgEl("g", { transform:
    `translate(${G.pan[0]},${G.pan[1]}) scale(${G.scale})` });

  // Layering: cables live in their own group *under* the nodes, so a cable is
  // never painted over a node box.  Previously both were appended to one group
  // in document order, which meant every edge covered the nodes it connected.
  const edgeLayer = svgEl("g", { class: "edges" });
  const nodeLayer = svgEl("g", { class: "nodes" });
  g.appendChild(edgeLayer);
  g.appendChild(nodeLayer);

  // --- edges, with a cubic that leaves the output to the right ----------
  for (const e of edges) {
    const src = nodes.find((n) => n.name === e.src);
    const dst = nodes.find((n) => n.name === e.dst);
    if (!src || !dst) continue;
    const ps = px(src), pd = px(dst);
    const y0 = portY(ps, src.outputs, e.src_port);
    const y1 = portY(pd, dst.inputs, e.dst_port);
    const x0 = ps[0] + NW, x1 = pd[0];
    const mid = Math.max(24, Math.abs(x1 - x0) * 0.45);
    const path = svgEl("path", {
      d: `M ${x0} ${y0} C ${x0 + mid} ${y0}, ${x1 - mid} ${y1}, ${x1} ${y1}`,
      fill: "none", stroke: typeColor(e.type), "stroke-width": 2.2,
      "stroke-opacity": 0.85, "marker-end": "url(#arrow)",
    });
    path.appendChild(svgEl("title")).textContent =
      `${e.src}.${e.src_port} → ${e.dst}.${e.dst_port}  (${e.type})`;
    path.style.cursor = "pointer";
    path.onclick = (ev) => { ev.stopPropagation(); disconnect(e.dst, e.dst_port); };
    // clicking a cable must not also start a pan or a node drag
    path.onmousedown = (ev) => ev.stopPropagation();
    edgeLayer.appendChild(path);
  }

  // --- live cable being dragged ----------------------------------------
  if (G.dragging) {
    const c = svgEl("path", {
      d: `M ${G.dragging.x0} ${G.dragging.y0} L ${G.dragging.x} ${G.dragging.y}`,
      fill: "none", stroke: typeColor(G.dragging.type), "stroke-width": 2,
      "stroke-dasharray": "5 4",
    });
    edgeLayer.appendChild(c);
  }

  // --- nodes ------------------------------------------------------------
  for (const n of nodes) {
    const [x, y] = px(n);
    const dirty = G.dirty.has(n.name);
    const sel = G.selected === n.name;
    const ng = svgEl("g", { transform: `translate(${x},${y})`, class: "node" });
    ng.appendChild(svgEl("rect", {
      x: 0, y: 0, width: NW, height: NH, rx: 9,
      fill: n.enabled ? "#241f1a" : "#1a1715",
      stroke: sel ? "#e8a33d" : (dirty ? "#c9822e" : "#3a322b"),
      "stroke-width": sel ? 2.4 : (dirty ? 2 : 1.2),
      "stroke-dasharray": n.enabled ? "" : "4 3",
    }));
    const t1 = svgEl("text", { x: 10, y: 21, fill: "#eae1d6",
      "font-size": "12.5", "font-weight": "600" });
    t1.textContent = n.label;
    ng.appendChild(t1);
    const t2 = svgEl("text", { x: 10, y: 37, fill: "#8d8175", "font-size": "10",
      "font-family": "ui-monospace,monospace" });
    const ms = G.ms[n.name];
    t2.textContent = `${n.name} · ${n.hash}` + (ms ? ` · ${ms.toFixed(0)}ms` : "");
    ng.appendChild(t2);
    if (ms) {
      const t3 = svgEl("text", { x: 10, y: 48, fill: dirty ? "#c9822e" : "#5f7f52",
        "font-size": "9.5", "font-family": "ui-monospace,monospace" });
      t3.textContent = dirty ? "需重算" : "已缓存";
      ng.appendChild(t3);
    }

    // ports
    n.inputs.forEach((p, i) => ng.appendChild(makePort(n, p, i, false, NW, NH)));
    n.outputs.forEach((p, i) => ng.appendChild(makePort(n, p, i, true, NW, NH)));

    ng.onmousedown = (ev) => {
      if (ev.target.dataset.port) return;
      G.selected = n.name;
      G.moving = { name: n.name,
                   dx: ev.clientX / G.scale - x - G.pan[0] / G.scale,
                   dy: ev.clientY / G.scale - y - G.pan[1] / G.scale };
      // Every node is already in pixel coordinates by the time it reaches a drag, because
      // `normalisePositions` converts them as the payload is adopted. The flag is asserted
      // here only so that a node added by some future path cannot slip through unconverted.
      n.px = true;
      renderCanvas();
      renderNodeInspector();
    };
    nodeLayer.appendChild(ng);
  }
  svg.appendChild(g);
}

function portY(pos, ports, portName) {
  // `ports` is the node's own port list, passed in rather than looked up: an
  // earlier version searched by name and compared a string against a node
  // object, so the lookup always failed and every edge threw.
  const idx = Math.max(0, (ports || []).findIndex((p) => p.name === portName));
  return pos[1] + 16 + idx * 18;
}

function makePort(node, port, idx, isOut, NW, NH) {
  const cx = isOut ? NW : 0;
  const cy = 16 + idx * 18;
  const g = svgEl("g", { "data-port": "1" });
  g.appendChild(svgEl("circle", {
    cx, cy, r: 5.5, fill: "#141110", stroke: typeColor(port.type), "stroke-width": 2.4,
    "data-port": "1", style: "cursor:crosshair",
  }));
  const label = svgEl("text", {
    x: isOut ? cx - 10 : cx + 10, y: cy + 3.5, fill: "#7d7268", "font-size": "9.5",
    "text-anchor": isOut ? "end" : "start", "font-family": "ui-monospace,monospace",
  });
  label.textContent = port.type;
  g.appendChild(label);
  g.appendChild(svgEl("title")).textContent =
    `${port.label || port.name} · ${G.state.types[port.type]?.label || port.type}`;

  g.onmousedown = (ev) => {
    ev.stopPropagation();
    const [nx, ny] = px(node);
    if (isOut) {
      G.dragging = { src: node.name, src_port: port.name, type: port.type,
                     x0: nx + NW, y0: ny + cy, x: nx + NW, y: ny + cy };
    } else {
      // dragging from an input disconnects it, which is the usual gesture
      disconnect(node.name, port.name);
    }
    renderCanvas();
  };
  g.onmouseup = (ev) => {
    ev.stopPropagation();
    if (!G.dragging || isOut) return;
    connect(G.dragging.src, G.dragging.src_port, node.name, port.name);
  };
  return g;
}

/* ------------------------------------------------------------ mutating */
async function addNode(kind) {
  const j = await graphApi("/api/graph/add", jsonPost({
    kind, x: 40 + Math.random() * 260, y: 40 + Math.random() * 200 }));
  if (j) applyGraphState(j);
}

async function connect(src, src_port, dst, dst_port) {
  G.dragging = null;
  const j = await graphApi("/api/graph/connect",
    jsonPost({ src, src_port, dst, dst_port }));
  if (j) { applyGraphState(j); refreshPreview(); }
}

async function disconnect(dst, dst_port) {
  const j = await graphApi("/api/graph/disconnect", jsonPost({ dst, dst_port }));
  if (j) { applyGraphState(j); refreshPreview(); }
}

async function removeSelected() {
  if (!G.selected) return;
  const j = await graphApi("/api/graph/remove", jsonPost({ name: G.selected }));
  G.selected = null;
  if (j) { applyGraphState(j); renderNodeInspector(); refreshPreview(); }
}

/* --------------------------------------------------- node inspector */
function renderNodeInspector() {
  const box = document.getElementById("nodeInspector");
  if (!box) return;
  if (!G.selected || !G.state.graph.nodes.find((n) => n.name === G.selected)) {
    box.innerHTML = `<div class="hint2">点选一个节点可单独调它的参数，或拖动输出端口连线。</div>`;
    return;
  }
  const name = G.selected;
  const spec = G.state.describe.nodes.find((n) => n.name === name);
  const stored = G.state.graph.nodes.find((n) => n.name === name);
  box.innerHTML = `
    <div class="niHead">
      <b>${spec.label}</b>
      <span class="mono">${name}</span>
      <button id="niDel" class="ghost" title="删除节点">删除</button>
    </div>
    <div class="niNote">${spec.device} — ${spec.note}</div>
    <label class="chk"><input id="niEnabled" type="checkbox"
      ${stored.enabled ? "checked" : ""}> 启用该节点</label>
    <div id="niParams"></div>
    <div class="niLayer">参数层：${spec.params_layer || "（无）"}</div>`;
  document.getElementById("niDel").onclick = removeSelected;
  document.getElementById("niEnabled").onchange = (e) =>
    setNode(name, { enabled: e.target.checked });

  const layer = spec.params_layer;
  const pbox = document.getElementById("niParams");
  if (!layer) {
    pbox.innerHTML = `<div class="hint2">该节点没有可调参数。</div>`;
    return;
  }
  const layerSpec = (S.schema.layers || []).find((l) => l.layer === layer);
  if (!layerSpec) return;
  for (const [gname, specs] of Object.entries(layerSpec.groups)) {
    const h = document.createElement("div");
    h.className = "palGroup";
    h.textContent = gname;
    pbox.appendChild(h);
    for (const sp of specs) {
      const cur = (S.params[layer] || {})[sp.key];
      const val = cur === undefined ? sp.default : cur;
      const row = document.createElement("div");
      row.className = "prow";
      row.innerHTML = `<div class="plabel"><span>${sp.label}</span>
        <span class="v mono">${fmt(val)}</span></div>`;
      const r = document.createElement("input");
      r.type = "range"; r.min = sp.min; r.max = sp.max;
      r.step = sp.step === 1 ? 1 : (sp.max - sp.min) / 200;
      r.value = val;
      r.oninput = () => {
        row.querySelector(".v").textContent =
          sp.step === 1 ? (parseFloat(r.value) > 0.5 ? "开" : "关") : fmt(parseFloat(r.value));
      };
      r.onchange = () => setNode(name, { params: { [sp.key]: parseFloat(r.value) } });
      row.appendChild(r);
      pbox.appendChild(row);
    }
  }
}

async function setNode(name, payload) {
  const j = await graphApi("/api/graph/node", jsonPost({ name, ...payload }));
  if (j) { applyGraphState(j); renderNodeInspector(); refreshPreview(); }
}

/* -------------------------------------------------------------- issues */
function renderIssues() {
  const box = document.getElementById("canvasIssues");
  if (!box) return;
  const issues = G.state.issues || [];
  if (!issues.length) {
    box.innerHTML = `<span class="okDot"></span> 接线有效，类型全部匹配`;
    return;
  }
  box.innerHTML = issues.map((i) =>
    `<div class="${i.severity === "error" ? "issErr" : "issWarn"}">
       ${i.severity === "error" ? "✕" : "!"} ${i.message}
       ${i.node ? ` <span class="mono">[${i.node}]</span>` : ""}</div>`).join("");
}

/* ------------------------------------------------------------- wiring */
function initCanvas() {
  const svg = document.getElementById("canvas");
  if (!svg) return;
  const toWorld = (ev) => {
    const r = svg.getBoundingClientRect();
    return [(ev.clientX - r.left - G.pan[0]) / G.scale,
            (ev.clientY - r.top - G.pan[1]) / G.scale];
  };
  svg.onmousemove = (ev) => {
    const [x, y] = toWorld(ev);
    if (G.dragging) { G.dragging.x = x; G.dragging.y = y; renderCanvas(); }
    else if (G.moving) {
      const nd = G.state.describe.nodes.find((m) => m.name === G.moving.name);
      if (nd) { nd.position = [x - G.moving.dx, y - G.moving.dy]; renderCanvas(); }
    } else if (G.panning) {
      G.pan[0] += ev.movementX; G.pan[1] += ev.movementY; renderCanvas();
    }
  };
  svg.onmouseup = () => {
    if (G.moving) {
      const nd = G.state.describe.nodes.find((m) => m.name === G.moving.name);
      if (nd) {
        nd.px = true;                       // pixel coordinates from now on
        const gnode = G.state.graph.nodes.find((m) => m.name === nd.name);
        if (gnode) { gnode.px = true; }
        fetch("/api/graph/move", jsonPost({ name: nd.name, x: nd.position[0],
                                            y: nd.position[1] }))
          .then((r) => r.json())
          .then((j) => { if (j && j.describe) { G.state = j; renderCanvas(); } })
          .catch(() => {});
      }
    }
    G.dragging = null; G.moving = null; G.panning = null; renderCanvas();
  };
  svg.onmouseleave = () => {
    G.dragging = null; G.moving = null; G.panning = null; renderCanvas();
  };
  svg.onmousedown = (ev) => {
    if (ev.target === svg || ev.target.tagName === "svg") {
      G.panning = true; G.selected = null; renderNodeInspector();
    }
  };
  svg.onwheel = (ev) => {
    ev.preventDefault();
    G.scale = Math.min(2, Math.max(0.35, G.scale * (ev.deltaY < 0 ? 1.1 : 0.9)));
    renderCanvas();
  };
  const eng = document.getElementById("engineGraph");
  if (eng) eng.onchange = () => { G.perNode = eng.checked; refreshPreview(); };
  const rm = document.getElementById("btnRemove");
  if (rm) rm.onclick = removeSelected;
  const open = document.getElementById("btnCanvas");
  const showCanvas = () => {
    document.getElementById("canvasPane").classList.remove("hidden");
    fitView();
    renderCanvas();
    renderNodeInspector();
  };
  if (open) open.onclick = showCanvas;
  // `#canvas` opens the canvas directly, which is handy for linking someone
  // straight to the apparatus view (and for automated verification)
  if (location.hash === "#canvas") showCanvas();
  const close = document.getElementById("btnCanvasClose");
  if (close) close.onclick = () =>
    document.getElementById("canvasPane").classList.add("hidden");
  window.addEventListener("resize", () => renderCanvas());
  if (typeof showCanvas === "function") window.showCanvas = showCanvas;
  // expose the canvas state for console inspection and automated checks: the
  // drag verification needs it, and so does anyone debugging a graph
  window.G = G;
  const fit = document.getElementById("btnFitView");
  if (fit) fit.onclick = () => { fitView(); renderCanvas(); };
  const ref = document.getElementById("btnGraphRefresh");
  if (ref) ref.onclick = async () => {
    const j = await graphApi("/api/graph");
    if (j) applyGraphState(j);
    refreshPreview();
  };
}

async function refreshGraphPerf() {
  const j = await graphApi("/api/graph/perf");
  if (!j) return;
  G.ms = {}; G.dirty = new Set();
  for (const st of Object.values(j)) {
    Object.assign(G.ms, st.node_ms || {});
  }
  renderCanvas();
}
