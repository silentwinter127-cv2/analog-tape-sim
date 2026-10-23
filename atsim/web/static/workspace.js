/* Virtual Media Lab (虚拟媒介实验室) WebUI — a Blender-like workspace
 *
 * Blender's workspaces are not extra features; they are **the same editors, arranged and shown differently**,
 * switched by a row of tabs, with each workspace remembering its own arrangement. That is what this does, and
 * the reason it is worth having here is that this interface has two populations of user at once: someone
 * building a chain wants the device catalogue and the graph, and someone grading an artefact wants the picture
 * as large as possible and the parameters beside it. Both use the same panels.
 *
 * Three pieces, and nothing else:
 *
 *   **splitters** — the two panel edges drag. The widths live in CSS variables on the grid so the layout stays
 *   declarative and no element needs an inline width. Clamped, and persisted.
 *
 *   **workspaces** — a named set of "which editors are visible". Switching one does not touch any project
 *   state: it shows and hides panels and the timeline, nothing more. That is the property that makes a
 *   workspace safe to click, and it is the thing to preserve if this grows.
 *
 *   **persistence** — widths per workspace and the active workspace, in localStorage, so the arrangement
 *   survives a reload. Kept under one key with a version, because a layout that cannot be reset is worse than
 *   one that cannot be saved.
 */
(function () {
  "use strict";

  var KEY = "atsim.workspace.v1";

  /* Each workspace is a set of visible editors.  `cols` is what the grid uses when a panel is hidden: a hidden
     panel's track collapses to zero rather than the panel being taken out of the flow, so the grid keeps its
     three tracks and the splitters keep working. */
  var WORKSPACES = [
    { id: "layout", name: "组合", hint: "全部编辑器", left: true, right: true, timeline: true },
    { id: "model", name: "建模", hint: "设备目录与节点图", left: true, right: false, timeline: false },
    { id: "grade", name: "调参", hint: "画面与参数", left: false, right: true, timeline: true },
    { id: "watch", name: "观看", hint: "只有画面", left: false, right: false, timeline: false },
    { id: "time", name: "时间线", hint: "关键帧与轨迹", left: true, right: true, timeline: true }
  ];

  var state = {
    active: "layout",
    widths: {},                 // workspace id -> { left, right }
    defaults: { left: 270, right: 340 }
  };

  var els = {};

  function load() {
    try {
      var raw = window.localStorage.getItem(KEY);
      if (!raw) return;
      var d = JSON.parse(raw);
      if (d && typeof d === "object") {
        if (typeof d.active === "string") state.active = d.active;
        if (d.widths && typeof d.widths === "object") state.widths = d.widths;
      }
    } catch (e) {
      /* a corrupt layout must not stop the page: fall back to the defaults */
    }
  }

  function save() {
    try {
      window.localStorage.setItem(KEY, JSON.stringify({ active: state.active, widths: state.widths }));
    } catch (e) { /* private mode, quota: the interface still works, it just forgets */ }
  }

  function ws() {
    for (var i = 0; i < WORKSPACES.length; i++) if (WORKSPACES[i].id === state.active) return WORKSPACES[i];
    return WORKSPACES[0];
  }

  function widthFor(side) {
    var w = state.widths[state.active];
    var v = w && w[side] != null ? w[side] : state.defaults[side];
    return Math.max(140, Math.min(720, Math.round(v)));
  }

  function setWidth(side, px) {
    if (!state.widths[state.active]) state.widths[state.active] = {};
    state.widths[state.active][side] = Math.max(140, Math.min(720, Math.round(px)));
    apply();
  }

  /* --- the layout ---------------------------------------------------------------------------------- */

  function apply() {
    var w = ws();
    var m = els.main;
    if (!m) return;
    var L = w.left ? widthFor("left") + "px" : "0px";
    var R = w.right ? widthFor("right") + "px" : "0px";
    m.style.setProperty("--w-left", L);
    m.style.setProperty("--w-right", R);
    if (els.left) els.left.classList.toggle("wsHidden", !w.left);
    if (els.right) els.right.classList.toggle("wsHidden", !w.right);
    if (els.splitL) els.splitL.classList.toggle("wsHidden", !w.left);
    if (els.splitR) els.splitR.classList.toggle("wsHidden", !w.right);
    if (els.strip) els.strip.classList.toggle("wsHidden", !w.timeline);
    if (typeof window.dispatchEvent === "function") {
      /* the canvas and the preview size themselves from their container, so they need telling */
      window.dispatchEvent(new Event("resize"));
    }
    renderBar();
  }

  function renderBar() {
    if (!els.bar) return;
    var cur = ws();
    els.bar.innerHTML = "";
    WORKSPACES.forEach(function (w) {
      var b = document.createElement("button");
      b.className = "wsTab" + (w.id === cur.id ? " on" : "");
      b.textContent = w.name;
      b.title = w.hint;
      b.dataset.ws = w.id;
      b.onclick = function () { setActive(w.id); };
      els.bar.appendChild(b);
    });
    var sp = document.createElement("span");
    sp.className = "wsSpacer";
    els.bar.appendChild(sp);

    var hint = document.createElement("span");
    hint.className = "wsHint";
    hint.textContent = cur.hint;
    els.bar.appendChild(hint);

    var reset = document.createElement("button");
    reset.className = "wsTab wsReset";
    reset.textContent = "重置布局";
    reset.title = "把当前工作区的宽度和可见性恢复默认";
    reset.onclick = function () {
      delete state.widths[state.active];
      save();
      apply();
    };
    els.bar.appendChild(reset);
  }

  function setActive(id) {
    state.active = id;
    save();
    apply();
  }

  /* --- the splitters ------------------------------------------------------------------------------ */

  function drag(side, ev) {
    ev.preventDefault();
    var start = ev.clientX;
    var startW = widthFor(side);
    var target = ev.currentTarget;
    target.classList.add("dragging");
    document.body.classList.add("wsDragging");

    function move(e) {
      var d = e.clientX - start;
      /* the right panel grows leftwards, so its delta is inverted.  The clamp is the same as setWidth's, which
         also guarantees the centre column keeps its own minimum because the grid's middle track is minmax(). */
      setWidth(side, side === "left" ? startW + d : startW - d);
    }
    function up() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      target.classList.remove("dragging");
      document.body.classList.remove("wsDragging");
      save();
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  function buildSplitter(side) {
    var s = document.createElement("div");
    s.className = "wsSplit wsSplit-" + side;
    s.title = "拖动调整宽度，双击恢复默认";
    s.addEventListener("pointerdown", function (ev) { drag(side, ev); });
    s.addEventListener("dblclick", function () {
      delete state.widths[state.active];
      save();
      apply();
    });
    return s;
  }

  /* --- the bar ------------------------------------------------------------------------------------ */

  function buildBar(main) {
    var bar = document.createElement("div");
    bar.className = "wsBar";
    bar.id = "wsBar";
    main.insertBefore(bar, main.firstChild);
    return bar;
  }

  function init() {
    els.main = document.querySelector("main");
    if (!els.main) return;
    els.left = document.getElementById("panelLeft");
    els.right = document.getElementById("panelRight");
    els.strip = document.getElementById("tlStrip");

    load();

    /* The bar occupies the first row and the three editors the second.  Written as an explicit
       grid-template-areas so the arrangement is one readable statement rather than a set of sizes, and so the
       splitters and the panels cannot be reordered by accident. */
    els.main.classList.add("wsGrid");
    els.bar = buildBar(els.main);
    els.splitL = buildSplitter("left");
    els.splitR = buildSplitter("right");
    /* The splitters are absolute overlays, not grid items, so their insertion order does not affect the
       layout -- they are appended and positioned from the CSS variables.  Appending rather than inserting keeps
       them out of the named areas entirely, which is the point. */
    els.main.appendChild(els.splitL);
    els.main.appendChild(els.splitR);

    apply();

    window.addEventListener("keydown", function (e) {
      /* Blender switches workspaces with the number row while a modifier is held; a plain digit would fight
         the text fields, so this uses Ctrl+Alt, which nothing else here uses. */
      if (!e.ctrlKey || !e.altKey) return;
      var n = parseInt(e.key, 10);
      if (n >= 1 && n <= WORKSPACES.length) {
        setActive(WORKSPACES[n - 1].id);
        e.preventDefault();
      }
    });

    window.Workspace = {
      state: state,
      list: WORKSPACES,
      set: setActive,
      setWidth: setWidth,
      apply: apply,
      reset: function () { state.widths = {}; save(); apply(); }
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
