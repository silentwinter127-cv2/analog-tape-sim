/* ============================================================================
   session_ui.js — 维度面板：环境状态 / 设备个体 / 关键帧时间线

   这三个维度在引擎里已经完整实现，但此前没有任何界面能碰到它们——只能在 Python 里手写参数。
   这个文件把它们接出来。

   设计上的两个约束：

   1. **不另造一套通信。** 用 app.js 已有的 `api` / `jsonPost` / `S` / `toast` / `applyProject`，
      所以状态只有一份，UI 与引擎不会各说各话。

   2. **每个面板都要在服务端没有这个字段时安静地不工作。** 这些能力是后加的，
      所以 applyProject 的返回值里它们可能是 undefined；面板必须容忍这一点，
      而不是抛异常把整个界面带崩。

   关键帧编辑器是画布而非表格：**曲线的形状是内容的一部分**。一个 exp 段和一个 linear 段
   在数值上可能端点相同，而形状不同——用表格表示会把这个差别藏起来，而它正是使用者要选的东西。
   ============================================================================ */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  /* 设备目录：和引擎里 STAGE_ORDER / 目录用的键对应。用户可自己加。 */
  const DEVICE_CLASSES = [
    ["camcorder", "摄像机"],
    ["deck", "录像机 / 播放器"],
    ["monitor", "监视器 / 电视机"],
    ["capture", "采集卡"],
  ];

  /* 环境量的可读说明。值是给滑块的，区间与引擎里的 Environment 一致。 */
  const ENV_FIELDS = [
    ["warmth", "暖度", 0, 1, 0.001,
      "1.0 = 已达到工作温度，0.0 = 冷机。冷机会拉偏彩色解码器、提高 CCD 暗电流、让伺服更吃力。"],
    ["age", "电容老化", 0, 1, 0.001,
      "0 = 全新，1 = 寿命终点。电解电容容量随老化下降，轨电压在负载下塌陷：失去增益与余量，钳位基准移动。"],
    ["humidity", "湿度", 0, 1, 0.001,
      "磁带与磁头之间的摩擦随湿度上升。这是「接口受潮」之外的另一个受潮部位。"],
    ["cold_sensitivity", "冷敏度", 0, 2, 0.01,
      "这台机器对冷的敏感程度。不同型号差别很大，1.0 表示完全按上表给出的幅度响应。"],
    ["age_sensitivity", "老化敏度", 0, 2, 0.01,
      "这台机器对电源老化的敏感程度。"],
  ];

  const CURVES = [
    ["linear", "linear · 恒定速率"],
    ["exp", "exp · 逼近平衡"],
    ["hold", "hold · 离散状态"],
    ["smooth", "smooth · 无物理依据时"],
  ];

  const SUI = {
    env: {},
    units: {},
    keyframes: [],
    selected: { track: 0, point: -1 },
    ready: false,
  };

  /* ------------------------------------------------------------ 工具 */
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function curveAt(track, i) {
    const pts = (track.points || []).slice().sort((a, b) => a.t - b.t);
    return pts[i] ? (pts[i].curve || "linear") : "linear";
  }

  function sortedPoints(track) {
    return (track.points || []).slice().sort((a, b) => a.t - b.t);
  }

  /* 与引擎 keyframes.Keyframe.valueAt 同一套语义，用来在编辑器里画曲线。
     故意重复这一小段算术而不是请求服务端：拖动时要逐帧重画，来回请求会让拖动发涩。
     两边的定义必须一致——所以曲线函数的形状在下面注释里写清楚，便于核对。 */
  function valueAt(track, t) {
    const pts = sortedPoints(track);
    if (!pts.length) return 0;
    if (t <= pts[0].t) return pts[0].value;
    if (t >= pts[pts.length - 1].t) return pts[pts.length - 1].value;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      if (t >= a.t && t <= b.t) {
        const u = (b.t - a.t) < 1e-9 ? 0 : (t - a.t) / (b.t - a.t);
        const c = a.curve || "linear";
        if (c === "hold") return a.value;
        if (c === "smooth") {
          const s = 0.5 - 0.5 * Math.cos(Math.PI * u);
          return a.value + (b.value - a.value) * s;
        }
        if (c === "exp") {
          // tau = 0.30：段末到达约 96%，余下留给下一段
          return b.value + (a.value - b.value) * Math.exp(-u / 0.30);
        }
        return a.value + (b.value - a.value) * u;
      }
    }
    return pts[pts.length - 1].value;
  }

  /* ------------------------------------------------------------ 服务端通信 */
  function push(patch, note) {
    if (typeof setProject === "function") {
      setProject(patch);
      if (note && typeof toast === "function") toast(note);
    } else if (typeof api === "function" && typeof jsonPost === "function") {
      api("/api/project", jsonPost(patch)).then((d) => {
        if (typeof applyProject === "function") applyProject(d);
        if (note && typeof toast === "function") toast(note);
      });
    }
  }

  /* ------------------------------------------------------------ 环境面板 */
  function envValue(key, dflt) {
    const v = SUI.env[key];
    return (v === undefined || v === null || v === "") ? dflt : Number(v);
  }

  function renderEnv() {
    const host = $("envRows");
    if (!host) return;
    host.textContent = "";
    ENV_FIELDS.forEach(([key, label, lo, hi, step, help]) => {
      const row = el("div", "ssRow");
      const lab = el("label", "ssLab", label);
      lab.title = help;
      const val = el("span", "ssVal mono");
      const inp = el("input");
      inp.type = "range";
      inp.min = lo; inp.max = hi; inp.step = step;
      // 未设置时显示名义值：暖度默认 1（已热），其余默认 0
      inp.value = envValue(key, key === "warmth" || key.endsWith("sensitivity") ? 1 : 0);
      val.textContent = Number(inp.value).toFixed(step < 0.01 ? 3 : 2);
      inp.oninput = () => { val.textContent = Number(inp.value).toFixed(step < 0.01 ? 3 : 2); };
      inp.onchange = () => {
        SUI.env[key] = Number(inp.value);
        push({ environment: Object.assign({}, SUI.env) }, null);
        renderEnvReadout();
      };
      row.append(lab, inp, val);
      host.append(row);

      const hint = el("div", "ssHelp", help);
      host.append(hint);
    });
    renderEnvReadout();
  }

  function renderEnvReadout() {
    const host = $("envReadout");
    if (!host) return;
    const warmth = envValue("warmth", 1);
    const age = envValue("age", 0);
    const hum = envValue("humidity", 0);
    const bits = [];
    bits.push(warmth >= 0.97 ? "已到工作温度"
      : warmth <= 0.03 ? "<b>冷机</b>：彩色解码器被拉偏、暗电流最高、伺服最吃力"
        : "<b>正在升温</b>（" + Math.round(warmth * 100) + "%）");
    if (age > 0.03) bits.push("电源已老化 " + Math.round(age * 100) + "%：轨电压塌陷，失去增益与余量");
    if (hum > 0.03) bits.push("湿度 " + Math.round(hum * 100) + "%");
    host.innerHTML = bits.join(" · ");
  }

  /* ------------------------------------------------------------ 设备面板 */
  function renderDevice() {
    const host = $("deviceRows");
    if (!host) return;
    host.textContent = "";
    DEVICE_CLASSES.forEach(([cls, label]) => {
      const row = el("div", "ssRow");
      const lab = el("label", "ssLab", label);
      const inp = el("input", "ssText");
      inp.type = "text";
      inp.placeholder = "留空 = 名义模型";
      inp.value = SUI.units[cls] || "";
      let timer = null;
      const commit = () => {
        const v = inp.value.trim();
        if (v) SUI.units[cls] = v; else delete SUI.units[cls];
        push({ unit_identities: Object.assign({}, SUI.units) }, null);
        renderDeviceReadout();
      };
      // 串号是连续输入的，但每次改动都重渲会打断打字——所以停顿后再提交
      inp.oninput = () => { clearTimeout(timer); timer = setTimeout(commit, 650); };
      inp.onchange = commit;
      row.append(lab, inp);
      host.append(row);
    });
    renderDeviceReadout();
  }

  function renderDeviceReadout() {
    const host = $("deviceReadout");
    if (!host) return;
    const rep = SUI.unitReport || [];
    if (!rep.length) {
      host.innerHTML = "<span class='ssDim'>没有设置序列号 —— 正在使用名义模型，"
        + "与加入设备个体差异之前逐位相同。</span>";
      return;
    }
    // 按偏差幅度排序：最大的几项才是这台机器的性格
    const rows = rep.slice().sort((a, b) => {
      const ra = Math.abs(a.shift || 0), rb = Math.abs(b.shift || 0);
      return rb - ra;
    }).slice(0, 8);
    const head = el("div", "ssDim",
      "这台机器偏离名义模型最大的 " + rows.length + " 项（共 " + rep.length + " 项）：");
    const ul = el("div", "ssDevList");
    rows.forEach((r) => {
      const line = el("div", "ssDev");
      const pct = (100 * (r.shift || 0)).toFixed(1);
      line.innerHTML = "<span class='mono'>" + r.layer + "." + r.key + "</span> "
        + "<span class='ssNm'>" + Number(r.nominal).toFixed(3) + "</span> → "
        + "<span class='mono ssNv'>" + Number(r.value).toFixed(3) + "</span> "
        + "<span class='ssPct'>" + (pct > 0 ? "+" : "") + pct + "%</span>";
      line.title = r.why || "";
      ul.append(line);
    });
    host.textContent = "";
    host.append(head, ul);
  }

  /* ------------------------------------------------------------ 关键帧编辑器 */

  /** 空轨道：默认驱动相机曝光，因为那是最直观能看到时间线在工作的地方。 */
  function newTrack() {
    return { layer: "camera", key: "exposure", absolute: true,
             points: [{ t: 0.0, value: 0.0, curve: "linear" },
                      { t: 1.0, value: 0.5, curve: "hold" }] };
  }

  /** 一次「录像场次」：冷机 → 预热 → 中途断电 → 重新升温。 */
  function sessionTracks() {
    return [{
      layer: "environment", key: "warmth", absolute: true,
      points: [{ t: 0.0, value: 0.0, curve: "exp" },
               { t: 4.0, value: 1.0, curve: "hold" },
               { t: 6.0, value: 1.0, curve: "hold" },
               { t: 6.2, value: 0.0, curve: "exp" },
               { t: 12.0, value: 1.0, curve: "hold" }],
    }];
  }

  /** 这个轨道在整段时间里的取值范围，用来定纵轴。 */
  function valueRange(track) {
    const pts = sortedPoints(track);
    if (!pts.length) return [-1, 1];
    let lo = Infinity, hi = -Infinity;
    // 采样而不是只看关键点：exp 段会越过两端之间的位置
    for (let i = 0; i <= 60; i++) {
      const t = (i / 60) * (pts[pts.length - 1].t || 1);
      const v = valueAt(track, t);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo) || !isFinite(hi)) return [-1, 1];
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.15;
    return [lo - pad, hi + pad];
  }

  function clipDuration() {
    const fps = (S.project && S.project.fps) || 25;
    const n = (S.project && S.project.frame_count) || 0;
    const total = (S.source_info && S.source_info.frame_count) || n || 0;
    const secs = total > 0 ? total / fps : 8;
    return Math.max(2, Math.min(120, secs));
  }

  function renderKeyframes() {
    const host = $("tlTracks");
    if (!host) return;
    host.textContent = "";
    if (!SUI.keyframes.length) {
      host.append(el("div", "ssDim", "时间线为空 —— 渲染与没有时间线时逐位相同。"));
      renderTlReadout();
      return;
    }
    SUI.keyframes.forEach((track, ti) => host.append(trackCard(track, ti)));
    renderTlReadout();
  }

  function paramOptions(selected) {
    // The schema is `{layers: [{layer, groups: {name: [spec, ...]}}]}`, which is how `renderParams` reads
    // it.  An earlier version indexed it as `schema[layer]`, which is undefined for every layer, so the
    // dropdown offered nothing usable -- the shape has to come from the payload the server sends, not from
    // what the reader expects.
    const sel = el("select", "ssSel");
    const add = (value, label) => {
      const o = el("option", null, label);
      o.value = value;
      if (value === selected) o.selected = true;
      sel.append(o);
    };
    // the environment is a virtual layer: it is not a device parameter but a condition that drives a set
    // of shifts, so it is offered first and separately
    add("environment.warmth", "environment · 暖度（驱动温度效应）");
    add("environment.age", "environment · 电容老化");
    add("environment.humidity", "environment · 湿度");
    const schema = (typeof S !== "undefined" && S.schema) || {};
    (schema.layers || []).forEach((layer) => {
      const groups = layer.groups || {};
      Object.keys(groups).forEach((gname) => {
        (groups[gname] || []).forEach((sp) => {
          if (!sp || !sp.key) return;
          add(layer.layer + "." + sp.key,
              layer.layer + " · " + (sp.label || sp.key));
        });
      });
    });
    return sel;
  }

  function trackCard(track, ti) {
    const card = el("div", "tlCard");
    const pts = sortedPoints(track);

    // --- 头部：参数选择 + 类型 + 删除
    const head = el("div", "tlHead");
    const combo = paramOptions(track.layer + "." + track.key);
    combo.onchange = () => {
      const parts = combo.value.split(".");
      track.layer = parts[0];
      track.key = parts.slice(1).join(".");
      schedule();
    };
    const absLab = el("label", "chk");
    const abs = el("input");
    abs.type = "checkbox";
    abs.checked = !!track.absolute;
    abs.title = "勾选 = 直接用该参数自己的单位；不勾 = 相对偏移（乘性，标称为 0 时按量程比例）";
    abs.onchange = () => { track.absolute = abs.checked; schedule(); };
    absLab.append(abs, document.createTextNode(" 绝对值"));
    const del = el("button", "ghost ssTiny", "删除轨道");
    del.onclick = () => {
      SUI.keyframes.splice(ti, 1);
      SUI.selected.track = Math.max(0, ti - 1);
      SUI.selected.point = -1;
      schedule();
    };
    head.append(combo, absLab, del);
    card.append(head);

    // --- 画布
    const cv = el("canvas", "tlCanvas");
    cv.width = 620; cv.height = 130;
    card.append(cv);

    // --- 点表
    const table = el("div", "tlPts");
    pts.forEach((p, pi) => {
      const row = el("div", "tlPt");
      const mk = (label, type, val, step, on) => {
        const cell = el("label", "ssLabMini", label);
        const inp = el("input");
        inp.type = type;
        inp.value = val;
        if (step != null) inp.step = step;
        inp.onchange = () => { on(inp.value); schedule(); };
        cell.append(inp);
        return cell;
      };
      row.append(mk("t (s)", "number", p.t.toFixed(3), "0.05", (v) => {
        p.t = Math.max(0, Number(v) || 0);
        pts.sort((a, b) => a.t - b.t);
      }));
      row.append(mk("值", "number", String(p.value), "0.01", (v) => { p.value = Number(v) || 0; }));
      const csel = el("select", "ssSelMini");
      CURVES.forEach(([cv2, lbl]) => {
        const o = el("option", null, lbl);
        o.value = cv2;
        if ((p.curve || "linear") === cv2) o.selected = true;
        csel.append(o);
      });
      csel.title = "这条曲线的形状属于「开始该段」的点：它描述的是怎么从这里走到下一个点";
      csel.onchange = () => { p.curve = csel.value; schedule(); };
      row.append(csel);
      const rm = el("button", "ghost ssTiny", "×");
      rm.title = "删除这个关键点";
      rm.onclick = () => {
        if (track.points.length <= 2) {
          if (typeof toast === "function") toast("一条轨道至少要留两个关键点");
          return;
        }
        track.points.splice(track.points.indexOf(p), 1);
        schedule();
      };
      row.append(rm);
      row.onmouseenter = () => { SUI.selected.track = ti; SUI.selected.point = pi; draw(); };
      table.append(row);
    });
    card.append(table);

    // --- 画布交互：点空白加关键点，拖点移动
    const geom = () => {
      const [lo, hi] = valueRange(track);
      const dur = clipDuration();
      return { lo, hi, dur };
    };
    const toPx = (p, g, r) => ({
      x: 34 + (p.t / g.dur) * (r.width - 46),
      y: r.height - 16 - ((p.value - g.lo) / (g.hi - g.lo)) * (r.height - 34),
    });
    const toVal = (x, y, g, r) => ({
      t: Math.max(0, ((x - 34) / (r.width - 46)) * g.dur),
      value: g.lo + (1 - (y - 16) / (r.height - 34)) * (g.hi - g.lo),
    });

    let drag = null;
    cv.onpointerdown = (e) => {
      const r = cv.getBoundingClientRect();
      const g = geom();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      let best = -1, bd = 12;
      pts.forEach((p, i) => {
        const q = toPx(p, g, r);
        const d = Math.hypot(q.x - px, q.y - py);
        if (d < bd) { bd = d; best = i; }
      });
      if (best >= 0) {
        drag = pts[best];
        SUI.selected.track = ti;
        SUI.selected.point = best;
        cv.setPointerCapture(e.pointerId);
      } else {
        // 空白处落一个新点，并把它拖到位置
        const v = toVal(px, py, g, r);
        const np = { t: v.t, value: v.value, curve: "linear" };
        track.points.push(np);
        pts.push(np);
        track.points.sort((a, b) => a.t - b.t);
        pts.sort((a, b) => a.t - b.t);
        drag = np;
        cv.setPointerCapture(e.pointerId);
        renderKeyframes();
      }
      draw();
    };
    cv.onpointermove = (e) => {
      if (!drag) return;
      const r = cv.getBoundingClientRect();
      const g = geom();
      const v = toVal(e.clientX - r.left, e.clientY - r.top, g, r);
      drag.t = v.t;
      drag.value = v.value;
      draw();
    };
    cv.onpointerup = (e) => {
      if (!drag) return;
      drag = null;
      try { cv.releasePointerCapture(e.pointerId); } catch (err) {}
      // 松手才提交：拖动过程中的每一像素都发请求会让界面发涩
      schedule(true);
      renderKeyframes();
    };

    function draw() {
      const ctx = cv.getContext("2d");
      const r = cv.getBoundingClientRect();
      const W = cv.width, H = cv.height;
      const g = geom();
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#12161c";
      ctx.fillRect(0, 0, W, H);

      // 纵轴刻度
      ctx.strokeStyle = "#242c38";
      ctx.fillStyle = "#5c6a7e";
      ctx.font = "10px ui-monospace, monospace";
      for (let i = 0; i <= 4; i++) {
        const y = 16 + (i / 4) * (H - 34);
        ctx.beginPath(); ctx.moveTo(34, y); ctx.lineTo(W - 12, y); ctx.stroke();
        const v = g.hi - (i / 4) * (g.hi - g.lo);
        ctx.fillText(v.toFixed(2), 2, y + 3);
      }
      // 横轴刻度
      const step = g.dur <= 5 ? 1 : g.dur <= 20 ? 5 : 10;
      for (let t = 0; t <= g.dur; t += step) {
        const x = 34 + (t / g.dur) * (W - 46);
        ctx.beginPath(); ctx.moveTo(x, 10); ctx.lineTo(x, H - 12); ctx.stroke();
        ctx.fillText(t + "s", x - 6, H - 2);
      }

      // 曲线：按引擎的曲线函数逐像素采样，所以看到的就是会生效的形状
      ctx.strokeStyle = "#5fd0a8";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i <= 240; i++) {
        const t = (i / 240) * g.dur;
        const p = toPx({ t: t, value: valueAt(track, t) }, g, r);
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();

      // 点与段的曲线标签
      pts.forEach((p, i) => {
        const q = toPx(p, g, r);
        const sel = SUI.selected.track === ti && SUI.selected.point === i;
        ctx.beginPath();
        ctx.arc(q.x, q.y, sel ? 5.5 : 4, 0, Math.PI * 2);
        ctx.fillStyle = sel ? "#ffd166" : "#8fe3c4";
        ctx.fill();
        if (i < pts.length - 1) {
          const mid = toPx({ t: (p.t + pts[i + 1].t) / 2, value: p.value }, g, r);
          ctx.fillStyle = "#7f8fa6";
          ctx.fillText(p.curve || "linear", mid.x - 12, mid.y - 8);
        }
      });
    }

    // 尺寸依赖布局，所以下一帧再画
    requestAnimationFrame(draw);
    return card;
  }

  function renderTlReadout() {
    const host = $("tlReadout");
    if (!host) return;
    if (!SUI.keyframes.length) {
      host.innerHTML = "<span class='ssDim'>没有时间线。所有参数保持常数，"
        + "渲染结果与没有这个功能时逐位相同。</span>";
      return;
    }
    const dur = clipDuration();
    const n = SUI.keyframes.reduce((a, t) => a + (t.points || []).length, 0);
    host.innerHTML = SUI.keyframes.length + " 条轨道、共 " + n + " 个关键点；"
      + "画布横轴按当前素材取 <b>" + dur.toFixed(1) + " s</b>。"
      + "曲线属于<b>开始该段</b>的点。";
  }

  /* 节流：编辑器会在拖动中频繁触发，但只有松手才需要真正推送。
     immediate=true 表示这次一定要发（拖拽结束、按钮操作）。 */
  let pending = null;
  function schedule(immediate) {
    renderTlReadout();
    if (pending) { clearTimeout(pending); pending = null; }
    const send = () => {
      pending = null;
      push({ keyframes: JSON.parse(JSON.stringify(SUI.keyframes)) }, null);
      renderTlReadout();
    };
    if (immediate) send();
    else pending = setTimeout(send, 400);
  }

  /* ------------------------------------------------------------ 装配 */
  /* ---------------------------------------------------------------- 字幕机 */

  /* The font list, with the coverage the user stated.  Recorded here as well as in the engine so the
     editor can warn before a render, and so the warning is specific: "this font has no Chinese" is
     actionable, "some glyphs are missing" is not. */
  const OSD_FONTS = [
    ["osd", "VCR OSD Mono", "仅英文/数字"],
    ["lyusung", "柳宋", "中英文数字"],
    ["apos", "A-POS 点阵", "中英文数字"],
  ];

  /* Which characters a font can draw, as far as the editor needs to know: the OSD font has no CJK at
     all, and the two Chinese fonts cover the common range.  Used only for a live warning beside the
     text; the engine reports the authoritative list after a render. */
  function charSupported(fontKey, ch) {
    const c = ch.codePointAt(0);
    if (c < 0x80) return true;
    if (fontKey === "osd") return false;
    return (c >= 0x3000 && c <= 0x30ff) || (c >= 0x4e00 && c <= 0x9fff)
        || (c >= 0xf900 && c <= 0xfaff) || (c >= 0xff00 && c <= 0xffef);
  }

  let osdEls = null;

  function osdPanels() {
    // the editor is a bespoke control, not a slider: the text is multi-line and the font is a named
    // choice, so the schema-driven parameter rows cannot express it.  The numeric parameters
    // (cell height, offsets, colour) stay in the parameter panel as usual, which is why this only
    // builds the parts that need it.
    if (osdEls) return osdEls;
    const host = $("osdPanel");
    if (!host) return null;
    host.textContent = "";

    const note = el("p", "ssNote");
    note.innerHTML = "**字符发生器不是排版引擎。** OSD 芯片有字模 ROM、每字一个固定字格、"
      + "一个位置寄存器：它把字形画进格子，<b>不测量、不调字距、不换行、不重排</b>。"
      + "汉字占两格、西文占一格——这就是当年的半角/全角约定。"
      + "一行超出右边缘的部分会直接丢失，因为真实发生器没有断行逻辑。";
    host.append(note);

    /* The enable switch comes first, and the text field enables it.  The *parameter* defaults to off --
       an unrequested overlay must not move any existing preset -- but an editor that requires a person to
       find a second control before their typed text appears is a defect in the editor, not a safety
       feature.  Typing enables; clearing disables; an explicit untick wins and keeps the text. */
    const erow = el("div", "ssRow");
    const echkLab = el("label", "chk");
    const echk = el("input");
    echk.type = "checkbox";
    const estate = el("span", "ssDim");
    echkLab.append(echk, document.createTextNode(" 启用字幕机"));
    erow.append(echkLab, estate);
    host.append(erow);

    // --- the text, multi-line, because a subtitle is
    const lab = el("div", "palGroup", "字幕文本");
    const ta = el("textarea", "ssTextArea");
    ta.rows = 4;
    ta.placeholder = "每行一条字幕。输入即启用。";
    ta.spellcheck = false;
    host.append(lab, ta);

    const warn = el("div", "ssWarn");
    host.append(warn);

    // --- the font
    const fl = el("div", "palGroup", "字体");
    const fsel = el("select", "ssSel");
    OSD_FONTS.forEach(([k, name, cover]) => {
      const o = el("option", null, name + "（" + cover + "）");
      o.value = k;
      fsel.append(o);
    });
    host.append(fl, fsel);

    // --- alignment, so a subtitle is placed rather than spaced
    const al = el("div", "palGroup", "对齐（不用再打空格挪位置）");
    const arow = el("div", "ssRow");
    const hsel = el("select", "ssSelMini");
    [["0", "左"], ["1", "居中"], ["2", "右"]].forEach(([v, t]) => {
      const o = el("option", null, "水平：" + t);
      o.value = v;
      hsel.append(o);
    });
    const vsel = el("select", "ssSelMini");
    [["0", "上"], ["1", "中"], ["2", "下"]].forEach(([v, t]) => {
      const o = el("option", null, "垂直：" + t);
      o.value = v;
      vsel.append(o);
    });
    arow.append(hsel, vsel);
    const arow2 = el("div", "ssRow");
    const quick = el("select", "ssSelMini");
    [["-1", "快速位置…"], ["8", "右下（录像机常用）"], ["7", "下中（字幕常用）"],
     ["4", "正中"], ["0", "左上"]].forEach(([v, t]) => {
      const o = el("option", null, t);
      o.value = v;
      quick.append(o);
    });
    arow2.append(quick);
    host.append(al, arow, arow2);

    // --- the position, which is the whole point of the feature
    const pl = el("div", "palGroup", "发生器位置");
    const psel = el("select", "ssSel");
    [["0", "画面成形之后（摄像机自身 OSD）"],
     ["1", "介质之前（字幕机录到磁带上）"]].forEach(([v, t]) => {
      const o = el("option", null, t);
      o.value = v;
      psel.append(o);
    });
    host.append(pl, psel);
    const pnote = el("div", "ssHelp");
    pnote.style.marginLeft = "0";
    pnote.innerHTML = "位置<b>不是</b>画面上的先后，而是信号链里的位置："
      + "摄像机 OSD 写在已成形的画面上，所以之后的磁带损伤落在画面、<b>不落在字上</b>；"
      + "字幕机录到磁带上，所以失落、磁头磨损、时基误差<b>连字一起毁</b>。"
      + "差别正好在磁带很差的时候看得出来。";
    host.append(pnote);

    // --- the enable switch and a live readout
    const row = el("div", "ssRow");
    const btn = el("button", "ghost", "应用到工程");
    row.append(btn);
    host.append(row);

    const readout = el("div", "ssReadout");
    host.append(readout);

    osdEls = { ta, fsel, psel, chk: echk, state: estate, warn, readout, btn,
               hsel, vsel, quick };

    let timer = null;
    const refreshWarn = () => {
      const font = fsel.value;
      const bad = [];
      Array.from(ta.value).forEach((ch) => {
        if (ch === "\n" || ch === "\r" || ch === "\t") return;
        if (!charSupported(font, ch) && bad.indexOf(ch) < 0) bad.push(ch);
      });
      if (!ta.value.trim()) {
        warn.textContent = "";
      } else if (bad.length) {
        warn.innerHTML = "字体 <b>" + font + "</b> 画不出：<b>" + bad.join(" ")
          + "</b> —— 这些字会被<b>明确报告为缺字</b>，而不是画成方块。换柳宋或 A-POS。";
      } else {
        warn.textContent = "";
      }
      const lines = ta.value ? ta.value.split("\n").length : 0;
      const chars = Array.from(ta.value).filter((c) => c !== "\n").length;
      const cells = Array.from(ta.value).reduce(
        (a, c) => a + (c === "\n" ? 0 : (c.codePointAt(0) >= 0x3000 ? 2 : 1)), 0);
      const on = echk.checked && chars > 0;
      state.innerHTML = on ? "<b>正在绘制</b>"
        : (chars > 0 ? "已关闭 —— 勾选「启用字幕机」后才会画" : "未输入文字");
      state.style.color = on ? "#7fe0bd" : "#5c6a7e";
      /* The frame's capacity, from the engine's own report when one has been rendered, and otherwise
         estimated from the image width -- a line wider than the frame is clipped at the edge, because a
         character generator does not wrap, and that is the commonest reason for "my text did not appear". */
      const rep = (SUI.osdReport) || null;
      const fit = rep && rep.cells_fit ? rep.cells_fit : null;
      // `cells_wanted` is the line's declared width; the drawn width is capped at the frame, so only the
      // wanted figure can show an overflow
      const used = Math.max(cells, rep && rep.cells_wanted ? rep.cells_wanted : 0);
      const over = rep ? !!rep.overflow : false;
      let width = "";
      if (fit) {
        width = (over || used > fit)
          ? " <b style='color:#ff6b6b'>最宽一行需要 " + used + " / " + fit
            + " 个半格，超出画面会被切掉（字符发生器不换行）</b>"
          : " 最宽一行占 " + used + " / " + fit + " 个半格，放得下。";
      }
      readout.innerHTML = lines + " 行、共 <b>" + chars + "</b> 字（占 <b>" + cells
        + "</b> 个半格）。" + width + " 字格高度和字色在右侧参数面板里调。";
    };

    /* The push is debounced: a subtitle is typed, not dragged, and one request per keystroke would
       re-render the preview on every character.  700 ms is longer than the font-commit delay because
       typing is continuous and a mid-word render is wasted. */
    /* `explicitOff` records that a person deliberately unticked the box.  Without it, typing after unticking
       would silently re-enable, and the control would appear not to work. */
    let explicitOff = false;
    const push = () => {
      const idx = OSD_FONTS.findIndex(([k]) => k === fsel.value);
      const hasText = ta.value.trim().length > 0;
      if (!explicitOff && hasText !== echk.checked) {
        // typing enables, clearing disables -- the text field is the real control
        echk.checked = hasText;
      }
      pushProject({ params: { overlay: {
        enabled: echk.checked ? 1 : 0,
        text: ta.value,
        font: idx < 0 ? 0 : idx,
        at: parseFloat(psel.value) || 0,
        halign: parseFloat(hsel.value) || 0,
        valign: parseFloat(vsel.value) || 0,
        // `pos` sets both alignments in the engine, so leaving a stale non-zero value here would silently
        // overrule the two dropdowns the user is looking at
        pos: 0,
      } } });
      refreshWarn();
    };
    const schedule = () => {
      refreshWarn();
      if (timer) clearTimeout(timer);
      timer = setTimeout(push, 700);
    };
    ta.oninput = schedule;
    fsel.onchange = () => { refreshWarn(); push(); };
    psel.onchange = push;
    hsel.onchange = push;
    vsel.onchange = push;
    quick.onchange = () => {
      const p = parseInt(quick.value, 10);
      quick.value = "-1";
      if (p < 0 || isNaN(p)) return;
      const HA = [0, 1, 2, 0, 1, 2, 0, 1, 2], VA = [0, 0, 0, 1, 1, 1, 2, 2, 2];
      hsel.value = String(HA[p]);
      vsel.value = String(VA[p]);
      push();
    };
    echk.onchange = () => {
      explicitOff = !echk.checked && ta.value.trim().length > 0;
      push();
    };
    btn.onclick = push;
    return osdEls;
  }

  /* Push a project patch using the host's own setProject, so the editor shares one state store with the
     rest of the interface rather than keeping a copy that can drift. */
  /* Layer values must go under `params`: `App.set_project` merges `payload["params"]` per layer, so a
     bare `{"overlay": {...}}` is ignored without complaint.  This interface has a second entry point,
     `set_levels`, which *does* take a bare layer mapping -- using the wrong one fails silently in both
     directions, so the shape is written down here rather than remembered. */
  function pushProject(patch) {
    /* Both paths send the same nested shape.  The fallback used to send the bare patch, which the API
       silently discards -- dead code while `app.js` is loaded, and a trap if it ever ran. */
    if (typeof setProject === "function") {
      setProject({ params: patch });
    } else if (typeof api === "function" && typeof jsonPost === "function") {
      api("/api/project", jsonPost({ params: patch })).then((d) => {
        if (typeof applyProject === "function") applyProject(d);
      });
    }
  }

  function syncOsd(d) {
    const e = osdPanels();
    if (!e) return;
    const cur = (d && d.params && d.params.overlay) || {};
    // do not clobber what the user is typing: only adopt the server's value when the field is not focused
    if (document.activeElement !== e.ta) {
      const t = cur.text == null ? "" : String(cur.text);
      if (e.ta.value !== t) e.ta.value = t;
    }
    const idx = Math.round(Number(cur.font == null ? 0 : cur.font));
    if (OSD_FONTS[idx]) e.fsel.value = OSD_FONTS[idx][0];
    e.psel.value = String(Math.round(Number(cur.at == null ? 0 : cur.at)));
    e.hsel.value = String(Math.round(Number(cur.halign == null ? 0 : cur.halign)));
    e.vsel.value = String(Math.round(Number(cur.valign == null ? 0 : cur.valign)));
    SUI.osdReport = (d && d.osd_report) || null;
    e.chk.checked = Number(cur.enabled == null ? 0 : cur.enabled) > 0.5;
    if (e.ta.value.trim().length && !e.chk.checked) {
      // the server says off while the field has text: that is a legitimate state (the user unticked), so
      // the state line explains it rather than the editor overriding it
    }
    // the engine's own report, if a frame has been rendered since
    const rep = (d && d.osd_report) || (S && S.osdReport) || null;
    if (e.state) { e.state.textContent = ""; }
    if (rep && rep.missing && rep.missing.length) {
      e.warn.innerHTML = "引擎报告缺字：<b>" + rep.missing.join(" ") + "</b>"
        + "（字体 " + (rep.font || "?") + "）";
    } else if (rep && rep.legible === false) {
      /* The engine measured the frame under the text and the text's own colour.  When they are too close the
         subtitle is drawn and invisible, which is exactly what happened: white text on a white corner of the
         picture, three glyphs reported as drawn and nothing to see.  The panel has to say so, because the
         reason is a parameter two groups further down and no one would guess it. */
      e.warn.innerHTML = "<b>字幕画出来了但看不见</b>：文字亮度 " + rep.text_luma
        + " 与它背后" + (rep.background_luma != null ? "的<b>底色块</b>" : "的<b>画面</b>")
        + "亮度 " + rep.behind_luma + " 太接近（对比度 " + rep.contrast
        + "）。打开右侧参数面板的<b>底色块</b>，或把字色改成深色。";
    }
  }

  function mountTabs() {
    // 在左侧面板已有内容之上插入三个按钮，用同一个 .tabs 容器
    let host = $("ssTabs");
    if (!host) {
      host = el("div", "tabs");
      host.id = "ssTabs";
      const left = $("panelLeft");
      if (!left) return;
      left.insertBefore(host, left.firstChild);
    }
    host.textContent = "";
    const panels = [["envPanel", "环境状态"], ["devicePanel", "设备个体"],
                    ["tlPanel", "关键帧时间线"], ["osdPanel", "字幕机"]];
    panels.forEach(([id, label]) => {
      const b = el("button", "tab", label);
      b.onclick = () => {
        const showing = !$(id).classList.contains("hidden");
        // 点已打开的按钮就收起，让出空间给参数列表
        panels.forEach(([other]) => $(other).classList.add("hidden"));
        if (!showing) $(id).classList.remove("hidden");
      };
      host.append(b);
    });
  }

  function mountPanels() {
    const left = $("panelLeft");
    if (!left || $("envPanel")) return;
    // 面板 HTML 由服务端作为一段模板下发，避免在 JS 里拼长字符串
    fetch("/static/session_panels.html")
      .then((r) => r.text())
      .then((html) => {
        const tmp = document.createElement("div");
        tmp.innerHTML = html;
        const anchor = $("presetList") || left.firstChild;
        Array.from(tmp.children).forEach((node) => left.insertBefore(node, anchor));
        mountTabs();
        renderEnv();
        renderDevice();
        renderKeyframes();
        try { osdPanels(); } catch (err) {}
        SUI.ready = true;
      })
      .catch(() => { /* 面板拿不到不该影响主界面 */ });
  }

  /* applyProject 之后同步面板状态。app.js 会在每次收到 /api/project 时调用这里。 */
  function syncFromProject(d) {
    if (!d) return;
    SUI.env = d.environment || {};
    SUI.units = d.unit_identities || {};
    SUI.keyframes = d.keyframes || [];
    SUI.unitReport = d.unit_report || [];
    if (SUI.ready) {
      renderEnv();
      renderDevice();
      renderKeyframes();
      // the editor is built from the project's own values, and reads them back from the same payload
      try { syncOsd(d); } catch (err) { /* the editor must not take the interface down */ }
    }
  }

  function init() {
    mountPanels();
    const addT = $("tlAddTrack");
    // 面板是异步插入的，所以按钮要等到位再绑
    const bind = setInterval(() => {
      if ($("tlAddTrack") && !$("tlAddTrack").dataset.bound) {
        $("tlAddTrack").dataset.bound = "1";
        $("tlAddTrack").onclick = () => {
          SUI.keyframes.push(newTrack());
          renderKeyframes();
          schedule(true);
        };
        $("tlAddSession").onclick = () => {
          SUI.keyframes = SUI.keyframes.concat(sessionTracks());
          renderKeyframes();
          schedule(true);
          if (typeof toast === "function") toast("已插入一次录像场次：冷机 → 预热 → 断电 → 重新升温");
        };
        $("tlClear").onclick = () => {
          SUI.keyframes = [];
          renderKeyframes();
          schedule(true);
        };
        clearInterval(bind);
      }
    }, 120);
    setTimeout(() => clearInterval(bind), 8000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.SessionUI = { syncFromProject: syncFromProject, state: SUI };
})();
