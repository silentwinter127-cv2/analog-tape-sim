/* Virtual Media Lab (虚拟媒介实验室) -- node graph canvas (with Blender-style Node Group support)
 *
 * Supports:
 *   1. Multi-selection (Shift/Ctrl+Click, Box Select drag / `B` key, or quick VHS 4-node selector)
 *   2. Packing selected nodes into a named Node Group (`Ctrl+G` or `📦 打包节点组`)
 *   3. Duplicating / Series-Chaining a Node Group (`Shift+D`, `📋 复制`, `⛓ 串联复制`)
 *   4. Opening a Node Group (`Tab` / Double-Click / `📂 进入组内视图`) to inspect and edit internal
 *      nodes on a dedicated sub-canvas with breadcrumb navigation, PLUS inline tabbed parameter
 *      editing directly inside `#nodeInspector`
 *   5. Unpacking a Node Group (`Alt+G` or `🔓 解包`)
 *   6. Subtitle Generator (`overlay` / `字幕机`) as a first-class graph node with text & selector controls
 */
"use strict";

const G = {
  state: null,          // last /api/graph response
  dragging: null,       // {src, src_port, type, x0, y0, x, y} while drawing a cable
  moving: null,         // {name, dx, dy, scale, gx, gy, isSub} while dragging a node
  selected: null,       // primary selected node name
  selectedSet: new Set(), // multi-selected node names (for Ctrl+G group packing)
  activeGroup: null,    // when non-null, name of the `node_group` currently opened in sub-canvas
  selectedSubNode: null,// selected internal node name inside `activeGroup` (or inspector tab)
  boxSelectMode: false, // toggled by `btnBoxSelect` or `B` key
  boxRect: null,        // {x0, y0, x1, y1} in world coords while box-dragging
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

const NODE_W = 176, NODE_H = 56;
const GROUP_W = 198, GROUP_H = 68;
const COL_STEP = 224, ROW_STEP = 108;

const ATSIM_CANVAS_BUILD = "canvas 2026-10-21 node-groups-blender-v2";
try {
  console.info("[atsim] " + ATSIM_CANVAS_BUILD);
} catch (e) {}

function toWorldFor(svgNode, ev) {
  const r = svgNode.getBoundingClientRect();
  return [(ev.clientX - r.left - G.pan[0]) / G.scale,
          (ev.clientY - r.top - G.pan[1]) / G.scale];
}

function px(node) {
  return [node.position ? node.position[0] : 0, node.position ? node.position[1] : 0];
}

const TYPE_COLOR = {
  rgb: "#00ff66", composite: "#60a5fa", rf: "#c084fc", dv: "#34d399",
  yiq: "#818cf8", hdmi: "#2dd4bf", audio: "#94a3b8", "?": "#64748b",
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

function flash(msg, ok = false) {
  const b = document.getElementById("canvasMsg");
  if (!b) return;
  b.textContent = msg;
  b.style.color = ok ? "#34d399" : "#f87171";
  clearTimeout(flash._t);
  flash._t = setTimeout(() => { b.style.color = ""; b.textContent = ""; }, 4200);
}

async function loadGraph() {
  const j = await graphApi("/api/graph");
  if (j) applyGraphState(j);
}

function normalisePositions(j) {
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
  if (!j) return;
  const first = !G.state;
  normalisePositions(j);
  G.state = j;
  if (j.project && typeof applyProject === "function") {
    applyProject(j.project);
    if (typeof renderParams === "function") renderParams();
  }
  if (typeof syncQuadBar === "function") syncQuadBar();
  // Validate activeGroup still exists
  if (G.activeGroup) {
    const grp = (G.state.graph?.nodes || []).find((n) => n.name === G.activeGroup && n.kind === "node_group");
    if (!grp) {
      G.activeGroup = null;
      G.selectedSubNode = null;
    }
  }
  // Prune removed nodes from selectedSet
  const validNames = new Set((G.state.graph?.nodes || []).map((n) => n.name));
  for (const s of Array.from(G.selectedSet)) {
    if (!validNames.has(s)) G.selectedSet.delete(s);
  }
  if (G.selected && !validNames.has(G.selected)) {
    G.selected = G.selectedSet.size ? Array.from(G.selectedSet)[0] : null;
  }
  renderPalette();
  renderGraphPresets();
  renderBreadcrumb();
  if (first) fitView();
  renderCanvas();
  renderIssues();
}

/* ------------------------------------------------------------- palette */
function renderPalette() {
  const boxes = document.querySelectorAll("#palette, #paletteSidebar, .palette-box");
  if (!boxes.length || !G.state) return;
  const groups = {};
  for (const n of (G.state.catalog || [])) (groups[n.group] ||= []).push(n);
  boxes.forEach((box) => {
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
  });
}

function renderGraphPresets() {
  const boxes = document.querySelectorAll("#graphPresets, #graphPresetsSidebar, .graph-presets-box");
  if (!boxes.length || !G.state) return;
  boxes.forEach((box) => {
    box.innerHTML = "";
    for (const [k, v] of Object.entries(G.state.presets || {})) {
      const b = document.createElement("button");
      b.textContent = v.label;
      b.title = v.note;
      b.onclick = async () => {
        G.activeGroup = null;
        G.selectedSubNode = null;
        const j = await graphApi("/api/graph/preset", jsonPost({ key: k }));
        if (j) {
          G.perNode = true;
          const pnt = document.getElementById("perNodeToggle");
          if (pnt) pnt.checked = true;
          applyGraphState(j);
          try {
            const proj = await api("/api/project");
            if (typeof applyProject === "function") applyProject(proj);
            if (typeof renderParams === "function") renderParams();
          } catch (_e) {}
          fitView();
          renderCanvas();
          renderNodeInspector();
          refreshPreview();
        }
      };
      box.appendChild(b);
    }
  });
}

function typeColor(t) { return TYPE_COLOR[t] || TYPE_COLOR["?"]; }

/* --------------------------------------------------- breadcrumb & group view */
function getActiveGroupNode() {
  if (!G.activeGroup || !G.state || !G.state.graph) return null;
  return (G.state.graph.nodes || []).find((n) => n.name === G.activeGroup && n.kind === "node_group") || null;
}

function enterNodeGroup(groupName) {
  const grp = (G.state?.graph?.nodes || []).find((n) => n.name === groupName && n.kind === "node_group");
  if (!grp || !grp.subgraph) return;
  G.activeGroup = groupName;
  const subNodes = grp.subgraph.nodes || [];
  G.selectedSubNode = subNodes.length ? subNodes[0].name : null;
  renderBreadcrumb();
  fitView();
  renderCanvas();
  renderNodeInspector();
  flash(`已进入节点组「${grp.label || groupName}」内部视图（按 Tab 或点击上方返回主图）`, true);
}

function exitNodeGroup() {
  if (!G.activeGroup) return;
  const prev = G.activeGroup;
  G.activeGroup = null;
  G.selected = prev;
  G.selectedSet = new Set([prev]);
  renderBreadcrumb();
  fitView();
  renderCanvas();
  renderNodeInspector();
}

function renderBreadcrumb() {
  const bar = document.getElementById("canvasBreadcrumb");
  if (!bar) return;
  const grp = getActiveGroupNode();
  if (!grp) {
    bar.classList.add("hidden");
    bar.innerHTML = "";
    return;
  }
  bar.classList.remove("hidden");
  const subCount = (grp.subgraph?.nodes || []).length;
  bar.innerHTML = `
    <button id="bcBackBtn" class="ghost bc-btn">⬅ 返回主节点图 (Tab)</button>
    <span class="bc-sep">›</span>
    <span class="bc-badge">📦 节点组内部视图：<b>${grp.label || grp.name}</b> <span class="mono">(${grp.name} · ${subCount} 个内部节点)</span></span>
    <span class="spacer"></span>
    <span class="hint2">点击组内任意设备节点可在右侧面板直接修改该节点的独立参数</span>
    <button id="bcDupBtn" class="ghost bc-btn" title="复制整个节点组">📋 复制该节点组</button>
    <button id="bcUngroupBtn" class="ghost bc-btn" title="将该节点组解包回主节点图">🔓 解包该节点组</button>
  `;
  document.getElementById("bcBackBtn").onclick = exitNodeGroup;
  document.getElementById("bcDupBtn").onclick = () => {
    const gname = grp.name;
    exitNodeGroup();
    duplicateNodeByName(gname, false);
  };
  document.getElementById("bcUngroupBtn").onclick = () => {
    const gname = grp.name;
    exitNodeGroup();
    ungroupByName(gname);
  };
}

/* -------------------------------------------------------------- canvas */
function currentViewGraph() {
  const grp = getActiveGroupNode();
  if (!grp || !grp.subgraph) {
    return {
      isSub: false,
      nodes: G.state?.describe?.nodes || [],
      edges: G.state?.describe?.edges || [],
    };
  }
  const sub = grp.subgraph;
  const subNodes = (sub.nodes || []).map((nd, idx) => {
    const catSpec = (G.state.catalog || []).find((c) => c.kind === nd.kind) || {
      kind: nd.kind, label: nd.label || nd.name, group: "设备",
      device: nd.kind, note: "", params_layer: nd.kind,
      inputs: [{ name: "in", type: "rgb", label: "in" }],
      outputs: [{ name: "out", type: "rgb", label: "out" }],
    };
    const pos = Array.isArray(nd.position) ? nd.position : [32 + (idx + 1) * COL_STEP, 96];
    return {
      ...catSpec,
      name: nd.name,
      label: nd.label || catSpec.label,
      enabled: nd.enabled !== false,
      iterations: nd.iterations || 1,
      position: pos,
      px: true,
      params: nd.params || {},
      hash: "sub",
    };
  });

  // Virtual boundary nodes: Group Input & Group Output (like Blender's Group Input / Group Output)
  const xs = subNodes.map((n) => n.position[0]);
  const ys = subNodes.map((n) => n.position[1]);
  const minX = xs.length ? Math.min(...xs) : 256;
  const maxX = xs.length ? Math.max(...xs) : 700;
  const avgY = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 96;

  const groupInNode = {
    name: "__group_in__",
    kind: "__group_in__",
    label: "组输入 (Group Input)",
    device: "节点组入口",
    note: "来自主节点图上游的输入信号",
    enabled: true,
    position: [minX - COL_STEP, avgY],
    px: true,
    inputs: [],
    outputs: (sub.inputs || []).map((p) => ({ name: p.name, type: p.type, label: p.label || p.name })),
    hash: "in",
  };
  const groupOutNode = {
    name: "__group_out__",
    kind: "__group_out__",
    label: "组输出 (Group Output)",
    device: "节点组出口",
    note: "送往主节点图下游的输出信号",
    enabled: true,
    position: [maxX + COL_STEP, avgY],
    px: true,
    inputs: (sub.outputs || []).map((p) => ({ name: p.name, type: p.type, label: p.label || p.name })),
    outputs: [],
    hash: "out",
  };

  const allNodes = [groupInNode, ...subNodes, groupOutNode];
  const allEdges = [];
  for (const pin of (sub.inputs || [])) {
    if (pin.target_node) {
      allEdges.push({
        src: "__group_in__", src_port: pin.name,
        dst: pin.target_node, dst_port: pin.target_port || "in",
        type: pin.type || "rgb", isBoundary: true,
      });
    }
  }
  for (const ed of (sub.edges || [])) {
    const srcNode = subNodes.find((n) => n.name === ed.src);
    const outPort = (srcNode?.outputs || []).find((p) => p.name === ed.src_port);
    allEdges.push({
      src: ed.src, src_port: ed.src_port,
      dst: ed.dst, dst_port: ed.dst_port,
      type: outPort?.type || "composite",
    });
  }
  for (const pout of (sub.outputs || [])) {
    if (pout.source_node) {
      allEdges.push({
        src: pout.source_node, src_port: pout.source_port || "out",
        dst: "__group_out__", dst_port: pout.name,
        type: pout.type || "yiq", isBoundary: true,
      });
    }
  }
  return { isSub: true, groupNode: grp, nodes: allNodes, edges: allEdges };
}

function fitView() {
  if (!G.state || !G.state.describe) return;
  const view = currentViewGraph();
  const nodes = view.nodes;
  if (!nodes.length) return;
  const PAD = 48;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of nodes) {
    const [nx, ny] = px(n);
    const nw = n.kind === "node_group" ? GROUP_W : NODE_W;
    const nh = n.kind === "node_group" ? GROUP_H : NODE_H;
    x0 = Math.min(x0, nx);
    y0 = Math.min(y0, ny);
    x1 = Math.max(x1, nx + nw);
    y1 = Math.max(y1, ny + nh);
  }
  const svg = document.getElementById("canvas");
  const w = svg.clientWidth || 900, h = svg.clientHeight || 500;
  const gw = Math.max(1, x1 - x0), gh = Math.max(1, y1 - y0);
  G.scale = Math.min(1.12, Math.max(0.3,
    Math.min((w - PAD * 2) / gw, (h - PAD * 2) / gh)));
  G.pan[0] = PAD - x0 * G.scale + Math.max(0, (w - PAD * 2 - gw * G.scale) / 2);
  G.pan[1] = PAD - y0 * G.scale + Math.max(0, (h - PAD * 2 - gh * G.scale) / 2);
}

function renderCanvas() {
  const svg = document.getElementById("canvas");
  if (!svg || !G.state || !G.state.describe) return;
  svg.innerHTML = "";
  const view = currentViewGraph();
  const nodes = view.nodes;
  const edges = view.edges;

  const defs = svgEl("defs");
  const marker = svgEl("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9",
    refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" });
  marker.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#64748b" }));
  const neoFilter = svgEl("filter", { id: "neoNodeShadow", x: "-20%", y: "-20%", width: "140%", height: "150%" });
  neoFilter.appendChild(svgEl("feDropShadow", { dx: "4", dy: "5", stdDeviation: "5", "flood-color": "#06090f", "flood-opacity": "0.75" }));
  defs.appendChild(neoFilter);
  defs.appendChild(marker);
  svg.appendChild(defs);

  const g = svgEl("g", { transform:
    `translate(${G.pan[0]},${G.pan[1]}) scale(${G.scale})` });

  const edgeLayer = svgEl("g", { class: "edges" });
  const nodeLayer = svgEl("g", { class: "nodes" });
  g.appendChild(edgeLayer);
  g.appendChild(nodeLayer);

  // --- edges ------------------------------------------------------------
  for (const e of edges) {
    const src = nodes.find((n) => n.name === e.src);
    const dst = nodes.find((n) => n.name === e.dst);
    if (!src || !dst) continue;
    const ps = px(src), pd = px(dst);
    const srcW = src.kind === "node_group" ? GROUP_W : NODE_W;
    const y0 = portY(ps, src.outputs, e.src_port);
    const y1 = portY(pd, dst.inputs, e.dst_port);
    const x0 = ps[0] + srcW, x1 = pd[0];
    const mid = Math.max(24, Math.abs(x1 - x0) * 0.45);
    const path = svgEl("path", {
      d: `M ${x0} ${y0} C ${x0 + mid} ${y0}, ${x1 - mid} ${y1}, ${x1} ${y1}`,
      fill: "none", stroke: typeColor(e.type), "stroke-width": 2.3,
      "stroke-opacity": e.isBoundary ? 0.65 : 0.88,
      "stroke-dasharray": e.isBoundary ? "5 3" : "",
      "marker-end": "url(#arrow)",
    });
    path.appendChild(svgEl("title")).textContent =
      `${e.src}.${e.src_port} → ${e.dst}.${e.dst_port}  (${e.type})`;
    if (!view.isSub) {
      path.style.cursor = "pointer";
      path.onclick = (ev) => { ev.stopPropagation(); disconnect(e.dst, e.dst_port); };
      path.onmousedown = (ev) => ev.stopPropagation();
    }
    edgeLayer.appendChild(path);
  }

  // --- live cable being dragged ----------------------------------------
  if (G.dragging && !view.isSub) {
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
    const isGroup = n.kind === "node_group";
    const isBoundary = n.kind === "__group_in__" || n.kind === "__group_out__";
    const nw = isGroup ? GROUP_W : NODE_W;
    const nh = isGroup ? GROUP_H : NODE_H;
    const dirty = G.dirty.has(n.name);
    const sel = view.isSub
      ? (G.selectedSubNode === n.name)
      : (G.selected === n.name || G.selectedSet.has(n.name));
    const multiSel = !view.isSub && G.selectedSet.size > 1 && G.selectedSet.has(n.name);

    const ng = svgEl("g", { transform: `translate(${x},${y})`, class: "node" });

    // Outer node card
    ng.appendChild(svgEl("rect", {
      x: 0, y: 0, width: nw, height: nh, rx: isGroup ? 11 : 9,
      fill: isBoundary ? "#141e2e" : (isGroup ? "#182438" : (n.enabled ? "#1b2436" : "#121722")),
      stroke: multiSel ? "#34d399" : (sel ? "#00ff66" : (isGroup ? "#60a5fa" : (dirty ? "#818cf8" : "#2c3b57"))),
      filter: "url(#neoNodeShadow)",
      "stroke-width": (sel || multiSel) ? 2.6 : (isGroup ? 1.8 : 1.2),
      "stroke-dasharray": n.enabled ? "" : "4 3",
    }));

    // Blender-style Node Group top header bar
    if (isGroup) {
      ng.appendChild(svgEl("rect", {
        x: 1, y: 1, width: nw - 2, height: 22, rx: 10,
        fill: "#1e324d", "fill-opacity": "0.95",
      }));
      const subNodes = n.subgraph?.nodes || [];
      const badge = svgEl("text", {
        x: nw - 8, y: 15, fill: "#7dd3fc", "font-size": "9.5",
        "text-anchor": "end", "font-family": "ui-monospace,monospace",
      });
      badge.textContent = `${subNodes.length} 节点 · 双击展开`;
      ng.appendChild(badge);
    }

    const t1 = svgEl("text", {
      x: 10, y: isGroup ? 16 : 20,
      fill: isGroup ? "#f0f7ff" : "#e2ecf8",
      "font-size": "12.5", "font-weight": "600",
    });
    const iterBadge = (n.iterations && n.iterations > 1) ? ` ×${n.iterations}` : "";
    t1.textContent = (isGroup ? `📦 ${n.label}` : n.label) + iterBadge;
    ng.appendChild(t1);

    const t2 = svgEl("text", {
      x: 10, y: isGroup ? 38 : 37,
      fill: isGroup ? "#93c5fd" : "#8b9eb7",
      "font-size": "10", "font-family": "ui-monospace,monospace",
    });
    if (isGroup) {
      const chainNames = (n.subgraph?.nodes || []).map((s) => s.label || s.name).join("→");
      t2.textContent = chainNames.length > 22 ? chainNames.slice(0, 21) + "…" : (chainNames || n.name);
    } else if (isBoundary) {
      t2.textContent = n.device;
    } else {
      const ms = G.ms[n.name];
      t2.textContent = `${n.name} · ${n.hash}` + (ms ? ` · ${ms.toFixed(0)}ms` : "");
    }
    ng.appendChild(t2);

    if (isGroup) {
      const t3 = svgEl("text", {
        x: 10, y: 55, fill: "#00ff66", "font-size": "9.5",
        "font-family": "ui-monospace,monospace",
      });
      const ms = G.ms[n.name];
      t3.textContent = `${n.name} · Tab/双击进入组内` + (ms ? ` · ${ms.toFixed(0)}ms` : "");
      ng.appendChild(t3);
    }

    // Ports
    (n.inputs || []).forEach((p, i) => ng.appendChild(makePort(n, p, i, false, nw, nh, view.isSub)));
    (n.outputs || []).forEach((p, i) => ng.appendChild(makePort(n, p, i, true, nw, nh, view.isSub)));

    ng.ondblclick = (ev) => {
      ev.stopPropagation();
      if (!view.isSub && isGroup) {
        enterNodeGroup(n.name);
      }
    };

    ng.onmousedown = (ev) => {
      if (ev.target.dataset.port) return;
      ev.stopPropagation();
      if (view.isSub) {
        if (!isBoundary) {
          G.selectedSubNode = n.name;
          const [wx, wy] = toWorldFor(svg, ev);
          G.moving = { name: n.name, dx: wx - x, dy: wy - y, scale: G.scale, gx: x, gy: y, isSub: true };
        }
        renderCanvas();
        renderNodeInspector();
        return;
      }

      if (ev.shiftKey || ev.ctrlKey || ev.metaKey) {
        if (G.selectedSet.has(n.name)) {
          G.selectedSet.delete(n.name);
          if (G.selected === n.name) {
            G.selected = G.selectedSet.size ? Array.from(G.selectedSet)[G.selectedSet.size - 1] : null;
          }
        } else {
          G.selectedSet.add(n.name);
          G.selected = n.name;
        }
      } else {
        if (!G.selectedSet.has(n.name) || G.selectedSet.size <= 1) {
          G.selectedSet = new Set([n.name]);
        }
        G.selected = n.name;
      }

      const [wx, wy] = toWorldFor(svg, ev);
      G.moving = { name: n.name, dx: wx - x, dy: wy - y, scale: G.scale, gx: x, gy: y, isSub: false };
      n.px = true;
      renderCanvas();
      renderNodeInspector();
    };
    nodeLayer.appendChild(ng);
  }

  // --- box selection rectangle -----------------------------------------
  if (G.boxRect) {
    const bx = Math.min(G.boxRect.x0, G.boxRect.x1);
    const by = Math.min(G.boxRect.y0, G.boxRect.y1);
    const bw = Math.abs(G.boxRect.x1 - G.boxRect.x0);
    const bh = Math.abs(G.boxRect.y1 - G.boxRect.y0);
    g.appendChild(svgEl("rect", {
      x: bx, y: by, width: bw, height: bh,
      fill: "rgba(56, 189, 248, 0.12)",
      stroke: "#00ff66", "stroke-width": 1.5, "stroke-dasharray": "4 3",
    }));
  }

  svg.appendChild(g);
}

function portY(pos, ports, portName) {
  const idx = Math.max(0, (ports || []).findIndex((p) => p.name === portName));
  return pos[1] + 22 + idx * 18;
}

function makePort(node, port, idx, isOut, NW, NH, isSub) {
  const cx = isOut ? NW : 0;
  const cy = 22 + idx * 18;
  const g = svgEl("g", { "data-port": "1" });
  g.appendChild(svgEl("circle", {
    cx, cy, r: 5.5, fill: "#0e131f", stroke: typeColor(port.type), "stroke-width": 2.4,
    "data-port": "1", style: isSub ? "cursor:default" : "cursor:crosshair",
  }));
  const label = svgEl("text", {
    x: isOut ? cx - 10 : cx + 10, y: cy + 3.5, fill: "#8b9eb7", "font-size": "9.5",
    "text-anchor": isOut ? "end" : "start", "font-family": "ui-monospace,monospace",
  });
  label.textContent = port.type;
  g.appendChild(label);
  g.appendChild(svgEl("title")).textContent =
    `${port.label || port.name} · ${G.state.types[port.type]?.label || port.type}`;

  if (!isSub) {
    g.onmousedown = (ev) => {
      ev.stopPropagation();
      const [nx, ny] = px(node);
      if (isOut) {
        G.dragging = { src: node.name, src_port: port.name, type: port.type,
                       x0: nx + NW, y0: ny + cy, x: nx + NW, y: ny + cy };
      } else {
        disconnect(node.name, port.name);
      }
      renderCanvas();
    };
    g.onmouseup = (ev) => {
      ev.stopPropagation();
      if (!G.dragging || isOut) return;
      connect(G.dragging.src, G.dragging.src_port, node.name, port.name);
    };
  }
  return g;
}

/* ------------------------------------------------------------ mutating */
async function addNode(kind) {
  if (G.activeGroup) exitNodeGroup();
  const j = await graphApi("/api/graph/add", jsonPost({
    kind, x: 60 + Math.random() * 240, y: 60 + Math.random() * 180 }));
  if (j) {
    applyGraphState(j);
    const added = (j.graph?.nodes || [])[j.graph.nodes.length - 1];
    if (added) {
      G.selected = added.name;
      G.selectedSet = new Set([added.name]);
    }
    renderCanvas();
    renderNodeInspector();
    refreshPreview();
  }
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
  if (G.activeGroup) {
    flash("请先返回主节点图或解包节点组后再删除节点");
    return;
  }
  const targets = G.selectedSet.size ? Array.from(G.selectedSet) : (G.selected ? [G.selected] : []);
  if (!targets.length) return;
  let lastJ = null;
  for (const nm of targets) {
    lastJ = await graphApi("/api/graph/remove", jsonPost({ name: nm }));
  }
  G.selected = null;
  G.selectedSet.clear();
  if (lastJ) { applyGraphState(lastJ); renderNodeInspector(); refreshPreview(); }
}

async function groupSelectedNodes(customLabel) {
  if (G.activeGroup) {
    flash("当前已在节点组内部视图中，请先返回主图");
    return;
  }
  let names = Array.from(G.selectedSet).filter((nm) => {
    const nd = (G.state?.graph?.nodes || []).find((n) => n.name === nm);
    return nd && nd.kind !== "source";
  });
  if (!names.length && G.selected) {
    const nd = (G.state?.graph?.nodes || []).find((n) => n.name === G.selected);
    if (nd && nd.kind !== "source") names = [G.selected];
  }
  // Convenient fallback: if user clicks "打包节点组" without multi-selecting first,
  // automatically look for the classic VHS 4-stage chain (camera, encoder, tape, deck)
  if (names.length < 2) {
    const vhsCandidates = ["camera", "encoder", "tape", "deck"].filter((nm) =>
      (G.state?.graph?.nodes || []).some((n) => n.name === nm)
    );
    if (vhsCandidates.length >= 2 && names.length === 0) {
      names = vhsCandidates;
    } else if (names.length === 0) {
      flash("请先按住 Shift 点选或框选要打包的节点（或点击「☑ 选VHS四件套」）");
      return;
    }
  }
  const nameIn = document.getElementById("groupNameInput");
  const label = (customLabel || (nameIn ? nameIn.value : "") || "VHS").trim() || "VHS";
  const j = await graphApi("/api/graph/group", jsonPost({ nodes: names, label }));
  if (j) {
    applyGraphState(j);
    if (j.created_group) {
      G.selected = j.created_group;
      G.selectedSet = new Set([j.created_group]);
    }
    renderCanvas();
    renderNodeInspector();
    refreshPreview();
    flash(`已将 ${names.length} 个节点打包为节点组「${label}」`, true);
  }
}

function selectVhsQuad() {
  if (G.activeGroup) exitNodeGroup();
  if (!G.state || !G.state.graph) return;
  const vhsKinds = new Set(["camera", "encoder", "dv_codec", "tape", "vhs_gen", "deck"]);
  const picked = (G.state.graph.nodes || [])
    .filter((n) => vhsKinds.has(n.kind))
    .map((n) => n.name);
  if (!picked.length) {
    flash("当前画布中没有找到成像/编码/介质/播放节点");
    return;
  }
  G.selectedSet = new Set(picked);
  G.selected = picked[0];
  renderCanvas();
  renderNodeInspector();
  flash(`已选中 ${picked.length} 个节点（${picked.join(" → ")}），点击「📦 打包节点组」即可打包为 VHS`, true);
}

async function ungroupByName(groupName) {
  const target = groupName || G.selected;
  if (!target) {
    flash("请先选中一个节点组");
    return;
  }
  const j = await graphApi("/api/graph/ungroup", jsonPost({ name: target }));
  if (j) {
    G.activeGroup = null;
    applyGraphState(j);
    if (Array.isArray(j.restored_nodes) && j.restored_nodes.length) {
      G.selectedSet = new Set(j.restored_nodes);
      G.selected = j.restored_nodes[0];
    }
    renderCanvas();
    renderNodeInspector();
    refreshPreview();
    flash(`已解包节点组「${target}」`, true);
  }
}

async function duplicateNodeByName(nodeName, chainInSeries = false) {
  const target = nodeName || G.selected;
  if (!target) {
    flash("请先选中一个节点或节点组");
    return;
  }
  const j = await graphApi("/api/graph/duplicate", jsonPost({ name: target, chain: !!chainInSeries }));
  if (j) {
    applyGraphState(j);
    if (j.duplicated_node) {
      G.selected = j.duplicated_node;
      G.selectedSet = new Set([j.duplicated_node]);
    }
    renderCanvas();
    renderNodeInspector();
    refreshPreview();
    flash(chainInSeries ? `已串联复制「${j.duplicated_node}」` : `已复制「${j.duplicated_node}」`, true);
  }
}

/* --------------------------------------------------- parameter control builder */
function appendLayerControls(container, layer, effectiveParams, onParamChange) {
  if (!layer) {
    container.innerHTML = `<div class="hint2">该节点没有可调参数。</div>`;
    return;
  }
  const layerSpec = (S.schema?.layers || []).find((l) => l.layer === layer);
  if (!layerSpec) return;

  for (const [gname, specs] of Object.entries(layerSpec.groups)) {
    const h = document.createElement("div");
    h.className = "palGroup";
    h.textContent = gname;
    container.appendChild(h);

    for (const sp of specs) {
      // Priority: node's own override (`effectiveParams[sp.key]`) -> project layer (`S.params[layer][sp.key]`) -> schema default
      const hasNodeVal = effectiveParams && effectiveParams[sp.key] !== undefined;
      const projVal = (S.params && S.params[layer]) ? S.params[layer][sp.key] : undefined;
      const val = hasNodeVal ? effectiveParams[sp.key] : (projVal !== undefined ? projVal : sp.default);

      const row = document.createElement("div");
      row.className = "prow";

      const opts = typeof selectorOptionsFor === "function" ? selectorOptionsFor(layer, sp) : null;
      const readStr = typeof formatReadout === "function"
        ? formatReadout(sp, val)
        : (sp.kind === "text" ? String(val ?? "") : (sp.step === 1 ? Math.round(val).toFixed(0) : fmt(val)));

      row.innerHTML = `<div class="plabel"><span>${sp.label}</span>
        <span class="v mono">${readStr}</span></div>`;
      const vSpan = row.querySelector(".v");

      if (sp.kind === "text") {
        const inp = document.createElement("textarea");
        inp.className = "ssTextArea";
        inp.rows = 2;
        inp.placeholder = "输入字幕文本（支持多行）…";
        inp.value = val == null ? "" : String(val);
        inp.onchange = () => {
          vSpan.textContent = inp.value;
          onParamChange(sp.key, inp.value);
        };
        row.appendChild(inp);
      } else if (opts) {
        const sel = document.createElement("select");
        sel.className = "psel";
        for (const [ov, olabel] of opts) {
          const o = document.createElement("option");
          o.value = String(ov);
          o.textContent = olabel;
          if (Math.abs(ov - Math.round(Number(val) || 0)) < 1e-6) o.selected = true;
          sel.appendChild(o);
        }
        sel.onchange = () => {
          const nv = parseFloat(sel.value);
          vSpan.textContent = typeof formatReadout === "function" ? formatReadout(sp, nv) : String(nv);
          onParamChange(sp.key, nv);
        };
        row.appendChild(sel);
      } else {
        const r = document.createElement("input");
        r.type = "range";
        r.min = sp.min;
        r.max = sp.max;
        r.step = sp.step === 1 ? 1 : (sp.max - sp.min) / 200;
        r.value = val;
        r.oninput = () => {
          const nv = parseFloat(r.value);
          vSpan.textContent = typeof formatReadout === "function"
            ? formatReadout(sp, nv)
            : (sp.step === 1 ? Math.round(nv).toFixed(0) : fmt(nv));
        };
        r.onchange = () => onParamChange(sp.key, parseFloat(r.value));
        row.appendChild(r);
      }
      container.appendChild(row);
    }
  }
}

/* --------------------------------------------------- node inspector */
function renderNodeInspector() {
  const box = document.getElementById("nodeInspector");
  if (!box || !G.state || !G.state.graph) return;

  // 1. Inside a Node Group's Sub-Canvas View
  const activeGrp = getActiveGroupNode();
  if (activeGrp && activeGrp.subgraph) {
    const subNodes = activeGrp.subgraph.nodes || [];
    if (!G.selectedSubNode || !subNodes.some((s) => s.name === G.selectedSubNode)) {
      G.selectedSubNode = subNodes.length ? subNodes[0].name : null;
    }
    const subNd = subNodes.find((s) => s.name === G.selectedSubNode);
    const catSpec = (G.state.catalog || []).find((c) => c.kind === subNd?.kind) || {};

    box.innerHTML = `
      <div class="ngBanner">
        <div><b>📦 节点组：${activeGrp.label || activeGrp.name}</b> <span class="mono">(${activeGrp.name})</span></div>
        <div class="ngSubTabs" id="ngSubTabs"></div>
      </div>
      <div id="niSubBody"></div>
    `;
    const tabsBox = document.getElementById("ngSubTabs");
    subNodes.forEach((s) => {
      const b = document.createElement("button");
      b.className = "ngTab" + (s.name === G.selectedSubNode ? " on" : "");
      b.textContent = s.label || s.name;
      b.onclick = () => {
        G.selectedSubNode = s.name;
        renderCanvas();
        renderNodeInspector();
      };
      tabsBox.appendChild(b);
    });

    const body = document.getElementById("niSubBody");
    if (!subNd) {
      body.innerHTML = `<div class="hint2">该节点组内暂无子节点。</div>`;
      return;
    }
    body.innerHTML = `
      <div class="niHead" style="margin-top:8px">
        <b>${subNd.label || catSpec.label || subNd.name}</b>
        <span class="mono">${subNd.name}</span>
      </div>
      <div class="niNote">${catSpec.device || subNd.kind} — ${catSpec.note || "节点组内部单元"}</div>
      <div class="ssRow" style="margin-bottom:8px">
        <label class="chk"><input id="niSubEnabled" type="checkbox" ${subNd.enabled !== false ? "checked" : ""}> 启用该内部节点</label>
        <span class="ssLabMini" style="margin-left:auto">迭代次数 <input id="niSubIter" type="number" min="1" max="25" value="${subNd.iterations || 1}" style="width:52px;padding:3px 5px"></span>
      </div>
      <div id="niSubParams"></div>
    `;
    document.getElementById("niSubEnabled").onchange = (e) =>
      setSubNode(activeGrp.name, subNd.name, { enabled: e.target.checked });
    document.getElementById("niSubIter").onchange = (e) =>
      setSubNode(activeGrp.name, subNd.name, { iterations: parseInt(e.target.value, 10) || 1 });

    const pbox = document.getElementById("niSubParams");
    appendLayerControls(pbox, catSpec.params_layer, subNd.params || {}, (key, val) => {
      setSubNode(activeGrp.name, subNd.name, { params: { [key]: val } });
    });
    return;
  }

  // 2. Multi-Selection Summary Bar (when 2+ nodes are selected on the main canvas)
  const multiNames = Array.from(G.selectedSet).filter((nm) =>
    (G.state.graph.nodes || []).some((n) => n.name === nm && n.kind !== "source")
  );
  if (multiNames.length >= 2) {
    box.innerHTML = `
      <div class="ngBanner">
        <b>已多选 ${multiNames.length} 个节点</b>
        <div class="niNote" style="margin:4px 0 8px">${multiNames.join(" → ")}</div>
        <div class="ssRow" style="gap:6px">
          <input id="niGroupLabel" type="text" value="${document.getElementById("groupNameInput")?.value || "VHS"}" placeholder="节点组名称 (如 VHS)" style="flex:1">
          <button id="niBtnPack" class="primary">📦 打包为节点组 (Ctrl+G)</button>
        </div>
      </div>
      <div id="niSingleBelow"></div>
    `;
    document.getElementById("niBtnPack").onclick = () => {
      const lbl = document.getElementById("niGroupLabel").value || "VHS";
      const topIn = document.getElementById("groupNameInput");
      if (topIn) topIn.value = lbl;
      groupSelectedNodes(lbl);
    };
  } else {
    box.innerHTML = `<div id="niSingleBelow"></div>`;
  }

  const targetBox = document.getElementById("niSingleBelow") || box;
  if (!G.selected || !G.state.graph.nodes.find((n) => n.name === G.selected)) {
    targetBox.innerHTML += `
      <div class="hint2" style="margin-bottom:8px">
        点选一个节点可单独调它的参数；按住 <b>Shift+点击</b> 或点上方 <b>「☑ 选VHS四件套」</b> 多选节点后点 <b>「📦 打包节点组」</b> 可将其打包为类似 Blender 的复合节点组。
      </div>
      <div class="ngQuickBox">
        <div class="palGroup">快速多选打包</div>
        <div id="niQuickChips" class="ngSubTabs"></div>
        <button id="niQuickVhsBtn" class="ghost wide" style="margin-top:6px">☑ 一键选中「成像+编码+介质+播放」并打包为 VHS</button>
      </div>
    `;
    const chipsBox = document.getElementById("niQuickChips");
    if (chipsBox) {
      for (const nd of (G.state.describe?.nodes || [])) {
        if (nd.kind === "source") continue;
        const b = document.createElement("button");
        b.className = "ngTab" + (G.selectedSet.has(nd.name) ? " on" : "");
        b.textContent = `${nd.label} (${nd.name})`;
        b.onclick = () => {
          if (G.selectedSet.has(nd.name)) G.selectedSet.delete(nd.name);
          else G.selectedSet.add(nd.name);
          G.selected = nd.name;
          renderCanvas();
          renderNodeInspector();
        };
        chipsBox.appendChild(b);
      }
    }
    const qBtn = document.getElementById("niQuickVhsBtn");
    if (qBtn) {
      qBtn.onclick = async () => {
        selectVhsQuad();
        await groupSelectedNodes("VHS");
      };
    }
    return;
  }

  const name = G.selected;
  const spec = G.state.describe.nodes.find((n) => n.name === name);
  const stored = G.state.graph.nodes.find((n) => n.name === name);
  if (!spec || !stored) return;

  // 3. Selected Node is a Blender-style `node_group` on the Main Canvas!
  if (stored.kind === "node_group" && stored.subgraph) {
    const subNodes = stored.subgraph.nodes || [];
    if (!G.selectedSubNode || !subNodes.some((s) => s.name === G.selectedSubNode)) {
      G.selectedSubNode = subNodes.length ? subNodes[0].name : null;
    }
    const activeSub = subNodes.find((s) => s.name === G.selectedSubNode);
    const activeSubSpec = (G.state.catalog || []).find((c) => c.kind === activeSub?.kind) || {};

    targetBox.innerHTML = `
      <div class="niHead">
        <b>📦 ${spec.label}</b>
        <span class="mono">${name}</span>
        <button id="niDel" class="ghost" title="删除该节点组">删除</button>
      </div>
      <div class="niNote">包含 ${subNodes.length} 个内部节点：${subNodes.map((s) => s.label || s.name).join(" → ")}</div>
      <div class="ssRow" style="gap:6px;margin-bottom:6px">
        <span class="ssLabMini">组名称</span>
        <input id="niGrpLabelIn" type="text" value="${stored.label || spec.label}" style="flex:1;padding:4px 7px;font-size:12px">
        <span class="ssLabMini">整组迭代 <input id="niGrpIter" type="number" min="1" max="25" value="${stored.iterations || 1}" style="width:48px;padding:3px 5px"></span>
      </div>
      <label class="chk" style="margin-bottom:8px"><input id="niEnabled" type="checkbox"
        ${stored.enabled ? "checked" : ""}> 启用该节点组</label>
      <div class="ngActionGrid">
        <button id="niEnterGrp" class="primary" title="进入节点组内部子画布 (快捷键 Tab 或双击节点)">📂 打开节点组视图 (Tab)</button>
        <button id="niDupGrp" class="ghost" title="复制一个独立的同配置节点组 (快捷键 Shift+D)">📋 复制节点组</button>
        <button id="niChainDupGrp" class="ghost" title="在下游串联复制一个节点组 (例如 VHS → VHS #2)">⛓ 串联复制</button>
        <button id="niUngroupGrp" class="ghost" title="解包还原为独立节点 (快捷键 Alt+G)">🔓 解包节点组</button>
      </div>
      <div class="palGroup" style="margin-top:12px">组内节点参数（点击切换内部设备直接修改）</div>
      <div id="niGrpSubTabs" class="ngSubTabs"></div>
      <div id="niGrpSubHeader" style="margin:8px 0 4px"></div>
      <div id="niParams"></div>
    `;
    document.getElementById("niDel").onclick = removeSelected;
    document.getElementById("niEnabled").onchange = (e) =>
      setNode(name, { enabled: e.target.checked });
    document.getElementById("niGrpLabelIn").onchange = (e) =>
      setNode(name, { label: e.target.value });
    document.getElementById("niGrpIter").onchange = (e) =>
      setNode(name, { iterations: parseInt(e.target.value, 10) || 1 });
    document.getElementById("niEnterGrp").onclick = () => enterNodeGroup(name);
    document.getElementById("niDupGrp").onclick = () => duplicateNodeByName(name, false);
    document.getElementById("niChainDupGrp").onclick = () => duplicateNodeByName(name, true);
    document.getElementById("niUngroupGrp").onclick = () => ungroupByName(name);

    const tabsEl = document.getElementById("niGrpSubTabs");
    subNodes.forEach((s) => {
      const b = document.createElement("button");
      b.className = "ngTab" + (s.name === G.selectedSubNode ? " on" : "");
      b.textContent = (s.enabled === false ? "🚫 " : "") + (s.label || s.name);
      b.onclick = () => {
        G.selectedSubNode = s.name;
        renderNodeInspector();
      };
      tabsEl.appendChild(b);
    });

    if (activeSub) {
      const subHdr = document.getElementById("niGrpSubHeader");
      subHdr.innerHTML = `
        <div class="ssRow" style="background:#171d19;padding:5px 8px;border-radius:6px;border:1px solid #2b3a30">
          <label class="chk"><input id="niInlineSubEn" type="checkbox" ${activeSub.enabled !== false ? "checked" : ""}> 启用「${activeSub.label || activeSub.name}」</label>
          <span class="ssLabMini" style="margin-left:auto">迭代 <input id="niInlineSubIt" type="number" min="1" max="25" value="${activeSub.iterations || 1}" style="width:46px;padding:2px 4px"></span>
        </div>
      `;
      document.getElementById("niInlineSubEn").onchange = (e) =>
        setSubNode(name, activeSub.name, { enabled: e.target.checked });
      document.getElementById("niInlineSubIt").onchange = (e) =>
        setSubNode(name, activeSub.name, { iterations: parseInt(e.target.value, 10) || 1 });

      const pbox = document.getElementById("niParams");
      appendLayerControls(pbox, activeSubSpec.params_layer, activeSub.params || {}, (key, val) => {
        setSubNode(name, activeSub.name, { params: { [key]: val } });
      });
    }
    return;
  }

  // 4. Regular Single Node Inspector
  targetBox.innerHTML = `
    <div class="niHead">
      <b>${spec.label}</b>
      <span class="mono">${name}</span>
      <button id="niDel" class="ghost" title="删除节点">删除</button>
    </div>
    <div class="niNote">${spec.device} — ${spec.note}</div>
    <div class="ssRow" style="margin-bottom:6px">
      <label class="chk"><input id="niEnabled" type="checkbox"
        ${stored.enabled ? "checked" : ""}> 启用该节点</label>
      ${stored.kind !== "source" ? `<span class="ssLabMini" style="margin-left:auto" title="单节点内循环执行该设备 1~25 次">迭代次数 <input id="niIter" type="number" min="1" max="25" value="${stored.iterations || 1}" style="width:52px;padding:3px 5px"></span>` : ""}
    </div>
    ${stored.kind !== "source" ? `
      <div class="ssRow" style="gap:6px;margin-bottom:8px">
        <button id="niDupBtn" class="ghost" style="flex:1;padding:4px 8px;font-size:11.5px">📋 复制节点</button>
        <button id="niChainDupBtn" class="ghost" style="flex:1;padding:4px 8px;font-size:11.5px">⛓ 串联复制</button>
      </div>
    ` : ""}
    <div id="niParams"></div>
    <div class="niLayer">参数层：${spec.params_layer || "（无）"}</div>`;

  document.getElementById("niDel").onclick = removeSelected;
  document.getElementById("niEnabled").onchange = (e) =>
    setNode(name, { enabled: e.target.checked });
  const iterIn = document.getElementById("niIter");
  if (iterIn) {
    iterIn.onchange = (e) => setNode(name, { iterations: parseInt(e.target.value, 10) || 1 });
  }
  const dupB = document.getElementById("niDupBtn");
  if (dupB) dupB.onclick = () => duplicateNodeByName(name, false);
  const cdupB = document.getElementById("niChainDupBtn");
  if (cdupB) cdupB.onclick = () => duplicateNodeByName(name, true);

  const pbox = document.getElementById("niParams");
  appendLayerControls(pbox, spec.params_layer, stored.params || {}, (key, val) => {
    setNode(name, { params: { [key]: val } });
  });
}

async function setNode(name, payload) {
  const j = await graphApi("/api/graph/node", jsonPost({ name, ...payload }));
  if (j) { applyGraphState(j); renderNodeInspector(); refreshPreview(); }
}

async function setSubNode(groupName, subNodeName, payload) {
  const j = await graphApi("/api/graph/group/node", jsonPost({
    group: groupName, name: subNodeName, ...payload,
  }));
  if (j) { applyGraphState(j); renderNodeInspector(); refreshPreview(); }
}

/* -------------------------------------------------------------- issues */
function renderIssues() {
  const box = document.getElementById("canvasIssues");
  if (!box || !G.state) return;
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

  svg.onmousemove = (ev) => {
    const [x, y] = toWorldFor(svg, ev);
    if (G.dragging) {
      G.dragging.x = x; G.dragging.y = y; renderCanvas();
    } else if (G.moving) {
      const k = G.moving.scale ? G.scale / G.moving.scale : 1;
      const nx = x - G.moving.dx * k;
      const ny = y - G.moving.dy * k;
      if (G.moving.isSub) {
        const grp = getActiveGroupNode();
        const subNd = (grp?.subgraph?.nodes || []).find((m) => m.name === G.moving.name);
        if (subNd) {
          subNd.position = [nx, ny];
          renderCanvas();
        }
      } else {
        const nd = G.state.describe.nodes.find((m) => m.name === G.moving.name);
        if (nd) {
          nd.position = [nx, ny];
          renderCanvas();
        }
      }
    } else if (G.boxRect) {
      G.boxRect.x1 = x;
      G.boxRect.y1 = y;
      renderCanvas();
    } else if (G.panning) {
      G.pan[0] += ev.movementX; G.pan[1] += ev.movementY; renderCanvas();
    }
  };

  svg.onmouseup = () => {
    if (G.moving) {
      if (G.moving.isSub) {
        const grp = getActiveGroupNode();
        const subNd = (grp?.subgraph?.nodes || []).find((m) => m.name === G.moving.name);
        if (grp && subNd && Array.isArray(subNd.position)) {
          fetch("/api/graph/group/node", jsonPost({
            group: grp.name, name: subNd.name,
            x: subNd.position[0], y: subNd.position[1],
          })).then((r) => r.json()).then((j) => {
            if (j && j.describe) { G.state = j; renderCanvas(); }
          }).catch(() => {});
        }
      } else {
        const nd = G.state.describe.nodes.find((m) => m.name === G.moving.name);
        if (nd) {
          nd.px = true;
          const gnode = G.state.graph.nodes.find((m) => m.name === nd.name);
          if (gnode) { gnode.px = true; gnode.position = [...nd.position]; }
          fetch("/api/graph/move", jsonPost({ name: nd.name, x: nd.position[0],
                                              y: nd.position[1] }))
            .then((r) => r.json())
            .then((j) => { if (j && j.describe) { G.state = j; renderCanvas(); } })
            .catch(() => {});
        }
      }
    } else if (G.boxRect) {
      const bx0 = Math.min(G.boxRect.x0, G.boxRect.x1);
      const by0 = Math.min(G.boxRect.y0, G.boxRect.y1);
      const bx1 = Math.max(G.boxRect.x0, G.boxRect.x1);
      const by1 = Math.max(G.boxRect.y0, G.boxRect.y1);
      if (bx1 - bx0 > 6 && by1 - by0 > 6 && !G.activeGroup) {
        const picked = [];
        for (const n of (G.state?.describe?.nodes || [])) {
          if (n.kind === "source") continue;
          const [nx, ny] = px(n);
          const nw = n.kind === "node_group" ? GROUP_W : NODE_W;
          const nh = n.kind === "node_group" ? GROUP_H : NODE_H;
          if (nx + nw >= bx0 && nx <= bx1 && ny + nh >= by0 && ny <= by1) {
            picked.push(n.name);
          }
        }
        G.selectedSet = new Set(picked);
        G.selected = picked.length ? picked[0] : null;
      }
      G.boxRect = null;
      G.boxSelectMode = false;
      const boxBtn = document.getElementById("btnBoxSelect");
      if (boxBtn) boxBtn.classList.remove("primary");
      renderCanvas();
      renderNodeInspector();
    }
    G.dragging = null; G.moving = null; G.panning = null; renderCanvas();
  };

  svg.onmouseleave = () => {
    G.dragging = null; G.moving = null; G.panning = null; G.boxRect = null; renderCanvas();
  };

  svg.onmousedown = (ev) => {
    if (ev.target === svg || ev.target.tagName === "svg") {
      if ((G.boxSelectMode || ev.shiftKey) && !G.activeGroup) {
        const [wx, wy] = toWorldFor(svg, ev);
        G.boxRect = { x0: wx, y0: wy, x1: wx, y1: wy };
        renderCanvas();
      } else {
        G.panning = true;
        if (!G.activeGroup) {
          G.selected = null;
          G.selectedSet.clear();
        }
        renderCanvas();
        renderNodeInspector();
      }
    }
  };

  svg.onwheel = (ev) => {
    ev.preventDefault();
    G.scale = Math.min(2, Math.max(0.35, G.scale * (ev.deltaY < 0 ? 1.1 : 0.9)));
    renderCanvas();
  };

  const eng = document.getElementById("engineGraph");
  if (eng) eng.onchange = () => { G.perNode = eng.checked; if (typeof syncQuadBar === "function") syncQuadBar(); refreshPreview(); };
  const rm = document.getElementById("btnRemove");
  if (rm) rm.onclick = removeSelected;

  const boxBtn = document.getElementById("btnBoxSelect");
  if (boxBtn) {
    boxBtn.onclick = () => {
      G.boxSelectMode = !G.boxSelectMode;
      boxBtn.classList.toggle("primary", G.boxSelectMode);
      if (G.boxSelectMode) flash("框选模式已开启：在画布空白处拖动鼠标框选多个节点", true);
    };
  }
  const selVhsBtn = document.getElementById("btnSelectVhs");
  if (selVhsBtn) selVhsBtn.onclick = selectVhsQuad;
  const grpBtn = document.getElementById("btnGroupNodes");
  if (grpBtn) grpBtn.onclick = () => groupSelectedNodes();
  const dupBtn = document.getElementById("btnDuplicateNode");
  if (dupBtn) dupBtn.onclick = () => duplicateNodeByName(G.selected, false);
  const cdupBtn = document.getElementById("btnChainDuplicate");
  if (cdupBtn) cdupBtn.onclick = () => duplicateNodeByName(G.selected, true);
  const ungrpBtn = document.getElementById("btnUngroupNode");
  if (ungrpBtn) ungrpBtn.onclick = () => ungroupByName(G.selected);

  // Blender-style keyboard shortcuts when canvas is open
  window.addEventListener("keydown", (ev) => {
    const pane = document.getElementById("canvasPane");
    if (!pane || pane.classList.contains("hidden")) return;
    const tag = (ev.target && ev.target.tagName) ? ev.target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return;

    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "g") {
      ev.preventDefault();
      groupSelectedNodes();
    } else if (ev.altKey && ev.key.toLowerCase() === "g") {
      ev.preventDefault();
      ungroupByName(G.activeGroup || G.selected);
    } else if (ev.shiftKey && ev.key.toLowerCase() === "d") {
      ev.preventDefault();
      duplicateNodeByName(G.selected, false);
    } else if (ev.key === "Tab") {
      if (G.activeGroup) {
        ev.preventDefault();
        exitNodeGroup();
      } else if (G.selected) {
        const nd = (G.state?.graph?.nodes || []).find((n) => n.name === G.selected);
        if (nd && nd.kind === "node_group") {
          ev.preventDefault();
          enterNodeGroup(nd.name);
        }
      }
    } else if (ev.key.toLowerCase() === "b" && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      G.boxSelectMode = !G.boxSelectMode;
      if (boxBtn) boxBtn.classList.toggle("primary", G.boxSelectMode);
      if (G.boxSelectMode) flash("框选模式已开启：在画布空白处拖动鼠标框选多个节点", true);
    }
  });

  const open = document.getElementById("btnCanvas");
  const showCanvas = () => {
    document.getElementById("canvasPane").classList.remove("hidden");
    fitView();
    renderCanvas();
    renderNodeInspector();
  };
  if (open) open.onclick = showCanvas;
  if (location.hash === "#canvas") showCanvas();
  const close = document.getElementById("btnCanvasClose");
  if (close) close.onclick = () =>
    document.getElementById("canvasPane").classList.add("hidden");
  window.addEventListener("resize", () => renderCanvas());
  window.showCanvas = showCanvas;
  window.G = G;
  window.groupSelectedNodes = groupSelectedNodes;
  window.enterNodeGroup = enterNodeGroup;
  window.exitNodeGroup = exitNodeGroup;
  window.duplicateNodeByName = duplicateNodeByName;
  window.ungroupByName = ungroupByName;

  const fit = document.getElementById("btnFitView");
  if (fit) fit.onclick = () => { fitView(); renderCanvas(); };
  for (const refId of ["btnGraphRefresh", "btnGraphRefreshSidebar"]) {
    const ref = document.getElementById(refId);
    if (ref) {
      ref.onclick = async () => {
        const j = await graphApi("/api/graph");
        if (j) applyGraphState(j);
        refreshPreview();
      };
    }
  }
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
