/* Broadcast Waveform Monitor (IRE Scope), Chroma Vectorscope & Node A/B Spectrum Analyzer
 *
 * Candidate D (Module C): Real-time Tektronix 1740A / Sony PVM style broadcast scopes drawer.
 * Provides:
 *   1. Broadcast IRE Waveform Monitor (-40 IRE .. +100 IRE) with HBI Sync Tip (-40 IRE),
 *      Back Porch Colour Burst (±20 IRE), Setup (7.5 IRE / 0 IRE), Luma/Composite & RGB Parade,
 *      plus interactive scanline trace selector.
 *   2. Broadcast Chroma Vectorscope with Polar Graticule, 75%/100% 6-colour target boxes
 *      (R, Mg, B, Cy, G, Yl), live Colour Burst vector (180° NTSC / ±135° PAL),
 *      and Hanover bar phase-split readout.
 *   3. Stage / Node-Level A/B Difference Heatmap (Total / Luma ΔY / Chroma ΔC) &
 *      0..6.5 MHz Horizontal Power Spectrum comparison (highlighting 3.58/4.43 MHz subcarrier
 *      and VHS luma bandwidth roll-off).
 */
"use strict";

(function () {
  const ScopeState = {
    open: true,
    wfmMode: "comp",       // "comp" | "luma" | "parade" | "line"
    abMode: "spectrum",    // "spectrum" | "heat_total" | "heat_luma" | "heat_chroma"
    lineYNorm: 0.5,        // Selected scanline (0..1)
    nodeA: "",             // "" = final output
    nodeB: "source",       // "source" or any stage/node name
    data: null,
    busy: false,
    pending: false,
  };

  const $id = (id) => document.getElementById(id);

  function initScopesPanel() {
    const drawer = $id("scopesDrawer");
    if (!drawer) return;

    const btnToggle = $id("btnToggleScopes");
    const btnHeaderScopes = $id("btnScopes");
    const setOpen = (open) => {
      ScopeState.open = !!open;
      drawer.classList.toggle("collapsed", !ScopeState.open);
      if (btnToggle) btnToggle.textContent = ScopeState.open ? "▼ 收起示波器" : "▲ 展开广播示波与校准面板";
      if (typeof window.syncAB === "function") {
        requestAnimationFrame(() => window.syncAB());
        setTimeout(() => window.syncAB(), 60);
      }
      if (ScopeState.open) refreshScopes();
    };

    if (btnToggle) {
      btnToggle.onclick = () => setOpen(!ScopeState.open);
    }
    if (btnHeaderScopes) {
      btnHeaderScopes.onclick = () => {
        setOpen(true);
        drawer.scrollIntoView({ behavior: "smooth", block: "nearest" });
      };
    }

    const wfmSel = $id("scopeWfmMode");
    if (wfmSel) {
      wfmSel.onchange = () => {
        ScopeState.wfmMode = wfmSel.value || "comp";
        drawWaveformScope();
      };
    }

    const lineSlider = $id("scopeLineSlider");
    if (lineSlider) {
      lineSlider.oninput = () => {
        ScopeState.lineYNorm = parseFloat(lineSlider.value) || 0.5;
        const lbl = $id("scopeLineLabel");
        if (lbl && ScopeState.data && ScopeState.data.waveform) {
          const tot = ScopeState.data.waveform.stats.total_lines || 480;
          lbl.textContent = `行 Y=${Math.round(ScopeState.lineYNorm * (tot - 1))}`;
        }
        refreshScopes();
      };
    }

    const abSel = $id("scopeAbMode");
    if (abSel) {
      abSel.onchange = () => {
        ScopeState.abMode = abSel.value || "spectrum";
        drawAbScope();
      };
    }

    const selA = $id("scopeNodeA");
    const selB = $id("scopeNodeB");
    if (selA) {
      selA.onchange = () => {
        ScopeState.nodeA = selA.value || "";
        refreshScopes();
      };
    }
    if (selB) {
      selB.onchange = () => {
        ScopeState.nodeB = selB.value || "source";
        refreshScopes();
      };
    }

    // Click on Waveform canvas to select horizontal scanline or inspect IRE
    const wfmCanvas = $id("wfmCanvas");
    if (wfmCanvas) {
      wfmCanvas.onclick = (ev) => {
        const r = wfmCanvas.getBoundingClientRect();
        const yn = Math.min(1, Math.max(0, (ev.clientY - r.top) / Math.max(1, r.height)));
        ScopeState.lineYNorm = yn;
        if (lineSlider) lineSlider.value = String(yn.toFixed(3));
        refreshScopes();
      };
    }

    // Also allow clicking on the main preview viewer with Alt+Click to pick scanline Y
    const viewer = $id("viewer");
    if (viewer) {
      viewer.addEventListener("click", (ev) => {
        if (!ev.altKey) return;
        const r = viewer.getBoundingClientRect();
        const yn = Math.min(1, Math.max(0, (ev.clientY - r.top) / Math.max(1, r.height)));
        ScopeState.lineYNorm = yn;
        if (lineSlider) lineSlider.value = String(yn.toFixed(3));
        refreshScopes();
      });
    }
  }

  async function refreshScopes() {
    if (!ScopeState.open) return;
    if (ScopeState.busy) {
      ScopeState.pending = true;
      return;
    }
    ScopeState.busy = true;
    try {
      const t = (typeof S !== "undefined" && S.t !== undefined) ? S.t : 0.4;
      const sc = (typeof S !== "undefined" && S.previewScale) ? S.previewScale : "0.5";
      const engine = (typeof G !== "undefined" && G.perNode) ? "graph" : "chain";
      const qs = new URLSearchParams({
        t: String(t),
        scale: String(sc),
        line: String(ScopeState.lineYNorm.toFixed(4)),
        engine: engine,
        node_a: ScopeState.nodeA || "",
        node_b: ScopeState.nodeB || "source",
      });
      const res = await fetch(`/api/scopes?${qs.toString()}`);
      if (!res.ok) return;
      const data = await res.json();
      ScopeState.data = data;
      syncNodeSelectors(data.available_nodes || []);
      drawWaveformScope();
      drawVectorscope();
      drawAbScope();
    } catch (e) {
      console.warn("refreshScopes error:", e);
    } finally {
      ScopeState.busy = false;
      if (ScopeState.pending) {
        ScopeState.pending = false;
        refreshScopes();
      }
    }
  }

  function syncNodeSelectors(nodes) {
    const selA = $id("scopeNodeA");
    const selB = $id("scopeNodeB");
    if (!selA || !selB || !Array.isArray(nodes) || !nodes.length) return;
    const sig = nodes.map((n) => `${n.id}:${n.label}`).join("|");
    if (selA.dataset.sig === sig) return;
    selA.dataset.sig = sig;
    selB.dataset.sig = sig;

    const prevA = ScopeState.nodeA || "";
    const prevB = ScopeState.nodeB || "source";

    selA.innerHTML = `<option value="">A: 最终输出 (Output)</option>` +
      nodes.map((n) => `<option value="${n.id}">A: ${n.label}</option>`).join("");
    selB.innerHTML = nodes.map((n) => `<option value="${n.id}">B: ${n.label}</option>`).join("");

    selA.value = prevA;
    selB.value = nodes.some((n) => n.id === prevB) ? prevB : "source";
  }

  /* ------------------------------------------------------------------
   * 1. Broadcast Waveform Monitor (IRE Scope: -45 IRE .. +115 IRE)
   * ------------------------------------------------------------------ */
  function drawWaveformScope() {
    const canvas = $id("wfmCanvas");
    if (!canvas || !ScopeState.data || !ScopeState.data.waveform) return;
    const wfm = ScopeState.data.waveform;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const ireMin = wfm.ire_min ?? -45;
    const ireMax = wfm.ire_max ?? 115;
    const ireToY = (ire) => {
      const norm = (ireMax - ire) / Math.max(1e-5, ireMax - ireMin);
      return Math.max(2, Math.min(H - 2, norm * (H - 12) + 6));
    };

    // CRT dark phosphor background
    ctx.fillStyle = "#070c09";
    ctx.fillRect(0, 0, W, H);

    const padL = 32;
    const plotW = W - padL - 6;
    const hbiFrac = wfm.stats.hbi_fraction || 0.16;
    const hbiX = padL + plotW * hbiFrac;

    // HBI shaded zone background
    ctx.fillStyle = "rgba(26, 48, 36, 0.26)";
    ctx.fillRect(padL, 0, plotW * hbiFrac, H);

    // Broadcast IRE Graticule Lines (-40, -20, 0, 7.5, 20, 40, 60, 80, 100 IRE)
    const graticules = [
      { ire: 100, label: "100", major: true, color: "rgba(218, 175, 72, 0.55)" },
      { ire: 80,  label: "80",  major: false },
      { ire: 60,  label: "60",  major: false },
      { ire: 40,  label: "40",  major: false },
      { ire: 20,  label: "20",  major: false },
      { ire: 7.5, label: "7.5", major: false, dashed: true, color: "rgba(120, 200, 150, 0.40)" },
      { ire: 0,   label: "0",   major: true, color: "rgba(218, 175, 72, 0.65)" },
      { ire: -20, label: "-20", major: false },
      { ire: -40, label: "-40", major: true, color: "rgba(217, 105, 82, 0.55)" },
    ];

    ctx.font = "9.5px ui-monospace, SFMono-Regular, Consolas, monospace";
    ctx.textBaseline = "middle";
    for (const g of graticules) {
      const y = ireToY(g.ire);
      ctx.strokeStyle = g.color || (g.major ? "rgba(190, 155, 75, 0.42)" : "rgba(95, 130, 105, 0.22)");
      ctx.lineWidth = g.major ? 1.0 : 0.75;
      ctx.setLineDash(g.dashed ? [3, 3] : []);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - 4, y);
      ctx.stroke();
      ctx.fillStyle = g.major ? "#d8b467" : "#6c8c74";
      ctx.fillText(g.label, 3, y);
    }
    ctx.setLineDash([]);

    // Vertical HBI / Active divider
    ctx.strokeStyle = "rgba(110, 165, 130, 0.35)";
    ctx.beginPath();
    ctx.moveTo(hbiX, 4);
    ctx.lineTo(hbiX, H - 4);
    ctx.stroke();
    ctx.fillStyle = "#7ba888";
    ctx.fillText("HBI(同步/色同步)", padL + 3, 11);
    ctx.fillText("有效行 (Active Video)", hbiX + 5, 11);

    const mode = ScopeState.wfmMode;

    if (mode === "parade" && wfm.parade) {
      // RGB Parade 3-column display
      const p = wfm.parade;
      const channels = [
        { p10: p.r_p10, p50: p.r_p50, p90: p.r_p90, fill: "rgba(255, 82, 82, 0.28)", stroke: "#ff5c5c", name: "R" },
        { p10: p.g_p10, p50: p.g_p50, p90: p.g_p90, fill: "rgba(82, 245, 125, 0.28)", stroke: "#52f57d", name: "G" },
        { p10: p.b_p10, p50: p.b_p50, p90: p.b_p90, fill: "rgba(88, 168, 255, 0.28)", stroke: "#58a8ff", name: "B" },
      ];
      const segW = plotW / 3;
      channels.forEach((ch, idx) => {
        const x0 = padL + idx * segW;
        const n = (ch.p50 || []).length;
        if (n < 2) return;
        ctx.fillStyle = ch.fill;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = x0 + (i / (n - 1)) * (segW - 4);
          const y = ireToY(ch.p90[i]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        for (let i = n - 1; i >= 0; i--) {
          const x = x0 + (i / (n - 1)) * (segW - 4);
          const y = ireToY(ch.p10[i]);
          ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = ch.stroke;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = x0 + (i / (n - 1)) * (segW - 4);
          const y = ireToY(ch.p50[i]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.fillStyle = ch.stroke;
        ctx.fillText(ch.name, x0 + 6, H - 8);
      });
    } else {
      // Draw 2D Phosphor Density Map for Composite/Luma IRE
      const grid = wfm.density_grid || [];
      const rows = grid.length;
      const cols = rows ? grid[0].length : 0;
      if (rows > 0 && cols > 0 && mode !== "line") {
        const cellW = plotW / cols;
        const cellH = (H - 12) / rows;
        for (let r = 0; r < rows; r++) {
          const rowArr = grid[r];
          const py = 6 + r * cellH;
          for (let c = 0; c < cols; c++) {
            const val = rowArr[c];
            if (val <= 0.02) continue;
            const alpha = Math.min(0.88, val * 0.92);
            const gBright = Math.round(165 + 90 * val);
            ctx.fillStyle = `rgba(55, ${gBright}, 105, ${alpha.toFixed(3)})`;
            ctx.fillRect(padL + c * cellW, py, Math.ceil(cellW), Math.ceil(cellH));
          }
        }
      }

      // Overlay selected scanline trace (bright yellow-green or cyan trace)
      const trace = (mode === "luma") ? (wfm.line_luma_ire || []) : (wfm.line_comp_ire || []);
      if (trace.length > 1) {
        ctx.strokeStyle = mode === "line" ? "#56f596" : "rgba(255, 224, 112, 0.90)";
        ctx.lineWidth = mode === "line" ? 1.5 : 1.15;
        ctx.beginPath();
        for (let i = 0; i < trace.length; i++) {
          const x = padL + (i / (trace.length - 1)) * plotW;
          const y = ireToY(trace[i]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }

    // Update numeric telemetry badge
    const st = wfm.stats || {};
    const badge = $id("wfmReadout");
    if (badge) {
      badge.textContent =
        `同步顶 ${st.sync_tip_ire?.toFixed(1) ?? "-40.0"} IRE · ` +
        `黑基座 ${st.setup_ire?.toFixed(1) ?? "7.5"} IRE · ` +
        `色同步 ${st.burst_pp_ire?.toFixed(0) ?? "40"} IREp-p · ` +
        `峰值白 ${st.active_p99_ire?.toFixed(1) ?? "100.0"} IRE`;
    }
    const lineLbl = $id("scopeLineLabel");
    if (lineLbl && st.selected_line !== undefined) {
      lineLbl.textContent = `行 Y=${st.selected_line}/${st.total_lines || 480}`;
    }
  }

  /* ------------------------------------------------------------------
   * 2. Broadcast Chroma Vectorscope (Polar U/V Graticule & 6 Targets)
   * ------------------------------------------------------------------ */
  function drawVectorscope() {
    const canvas = $id("vecCanvas");
    if (!canvas || !ScopeState.data || !ScopeState.data.vectorscope) return;
    const vec = ScopeState.data.vectorscope;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const cx = W * 0.5;
    const cy = H * 0.5;
    const R = Math.min(W, H) * 0.43;

    ctx.fillStyle = "#070c09";
    ctx.fillRect(0, 0, W, H);

    // Outer 100% circle & inner 75% / 25% graticule circles
    ctx.strokeStyle = "rgba(195, 160, 78, 0.45)";
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = "rgba(110, 150, 120, 0.24)";
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.arc(cx, cy, R * 0.75, 0, Math.PI * 2);
    ctx.arc(cx, cy, R * 0.285, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Crosshair axes (+U / -U and +V / -V) + degree ticks every 30 deg
    ctx.strokeStyle = "rgba(175, 145, 72, 0.35)";
    ctx.beginPath();
    ctx.moveTo(cx - R * 1.06, cy);
    ctx.lineTo(cx + R * 1.06, cy);
    ctx.moveTo(cx, cy - R * 1.06);
    ctx.lineTo(cx, cy + R * 1.06);
    ctx.stroke();

    // -I / +Q axis lines (rotated 33 deg for NTSC)
    const iqRad = (123.0 * Math.PI) / 180.0;
    ctx.strokeStyle = "rgba(100, 165, 145, 0.20)";
    ctx.beginPath();
    ctx.moveTo(cx - R * Math.cos(iqRad), cy + R * Math.sin(iqRad));
    ctx.lineTo(cx + R * Math.cos(iqRad), cy - R * Math.sin(iqRad));
    ctx.stroke();

    // Draw 2D polar density cloud + colour-tinted vector scatter
    const scatter = vec.scatter || [];
    for (let i = 0; i < scatter.length; i++) {
      const pt = scatter[i];
      const u = pt[0];
      const v = pt[1];
      const px = cx + u * R;
      const py = cy - v * R;
      ctx.fillStyle = `rgba(65, 235, 125, 0.28)`;
      ctx.fillRect(px - 1, py - 1, 2.2, 2.2);
    }

    // Draw 6 Standard Broadcast Target Boxes (R, Mg, B, Cy, G, Yl)
    const targets = vec.targets || {};
    const boxColors = {
      R: "#ff6262",
      Mg: "#f078ff",
      B: "#68a8ff",
      Cy: "#58e8f5",
      G: "#62f282",
      Yl: "#ffe058",
    };
    ctx.font = "bold 9.5px ui-monospace, SFMono-Regular, Consolas, monospace";
    for (const [code, t] of Object.entries(targets)) {
      const tx = cx + t.u_75 * R;
      const ty = cy - t.v_75 * R;
      const col = boxColors[code] || "#e8a33d";

      // 75% target reticle box + crosshair
      const bs = 5.5;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.15;
      ctx.strokeRect(tx - bs, ty - bs, bs * 2, bs * 2);

      // Small 100% outer bracket dot
      const tx100 = cx + t.u_100 * R;
      const ty100 = cy - t.v_100 * R;
      ctx.fillStyle = col;
      ctx.fillRect(tx100 - 1.5, ty100 - 1.5, 3, 3);
      ctx.fillText(code, tx + 7, ty + 3);
    }

    // Draw Live Colour Burst Vector(s) (180° NTSC / ±135° PAL)
    const bursts = vec.burst_vectors || [];
    for (const b of bursts) {
      const bx = cx + b.u * R;
      const by = cy - b.v * R;
      ctx.strokeStyle = "#ffd166";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(bx, by);
      ctx.stroke();

      ctx.fillStyle = "#ffd166";
      ctx.beginPath();
      ctx.arc(bx, by, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }

    const st = vec.stats || {};
    const badge = $id("vecReadout");
    if (badge) {
      const killer = st.color_killer_active ? " [消色器动作]" : "";
      badge.textContent =
        `${vec.standard || "NTSC"} · 色同步相移 ${st.burst_phase_error_deg >= 0 ? "+" : ""}${st.burst_phase_error_deg?.toFixed(1) ?? "0.0"}° · ` +
        `ACC ${st.acc_chroma_gain?.toFixed(2) ?? "1.00"}× · ` +
        `饱和度 P95 ${st.p95_saturation_pct?.toFixed(1) ?? "0.0"}% · ` +
        `汉诺威分裂 ${st.hanover_split_deg?.toFixed(1) ?? "0.0"}°${killer}`;
    }
  }

  /* ------------------------------------------------------------------
   * 3. Node / Chain A/B Difference Heatmap & 0..6.5 MHz Power Spectrum
   * ------------------------------------------------------------------ */
  function drawAbScope() {
    const canvas = $id("abCanvas");
    if (!canvas || !ScopeState.data || !ScopeState.data.ab_compare) return;
    const ab = ScopeState.data.ab_compare;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.fillStyle = "#070c09";
    ctx.fillRect(0, 0, W, H);

    const mode = ScopeState.abMode;
    const st = ab.stats || {};

    if (mode === "spectrum" && ab.spectrum) {
      const sp = ab.spectrum;
      const freqs = sp.freqs_mhz || [];
      const refDb = sp.ref_luma_db || [];
      const simDb = sp.sim_luma_db || [];
      const compDb = sp.sim_comp_db || [];
      const n = freqs.length;
      if (n < 2) return;

      const padL = 30, padR = 8, padT = 14, padB = 18;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const dbMin = -55, dbMax = 5;

      const fToX = (f) => padL + (f / 6.5) * plotW;
      const dbToY = (db) => padT + ((dbMax - Math.max(dbMin, Math.min(dbMax, db))) / (dbMax - dbMin)) * plotH;

      // Grid lines (dB and MHz)
      ctx.font = "9px ui-monospace, SFMono-Regular, Consolas, monospace";
      ctx.strokeStyle = "rgba(95, 130, 105, 0.22)";
      ctx.lineWidth = 0.75;
      for (const db of [0, -15, -30, -45]) {
        const y = dbToY(db);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
        ctx.fillStyle = "#6c8c74";
        ctx.fillText(`${db}dB`, 2, y + 3);
      }
      for (let mhz = 1; mhz <= 6; mhz++) {
        const x = fToX(mhz);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, H - padB);
        ctx.stroke();
        ctx.fillStyle = "#6c8c74";
        ctx.fillText(`${mhz}M`, x - 7, H - 5);
      }

      // Mark Luma Bandwidth cutoff & Chroma Subcarrier frequency (3.58 / 4.43 MHz)
      const lumaBw = st.luma_bw_nominal_mhz || 3.0;
      const fsc = st.subcarrier_mhz || 3.58;

      const xBw = fToX(lumaBw);
      ctx.strokeStyle = "rgba(105, 195, 255, 0.50)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xBw, padT);
      ctx.lineTo(xBw, H - padB);
      ctx.stroke();
      ctx.fillStyle = "#69c3ff";
      ctx.fillText(`Y带宽 ${lumaBw.toFixed(1)}M`, Math.min(W - 68, xBw + 3), padT + 9);

      const xFsc = fToX(fsc);
      ctx.strokeStyle = "rgba(255, 209, 102, 0.60)";
      ctx.beginPath();
      ctx.moveTo(xFsc, padT);
      ctx.lineTo(xFsc, H - padB);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ffd166";
      ctx.fillText(`fsc ${fsc.toFixed(2)}M`, Math.min(W - 65, xFsc + 3), padT + 20);

      // Draw Reference Luma Spectrum (dashed grey-cyan)
      ctx.strokeStyle = "rgba(160, 195, 210, 0.65)";
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = fToX(freqs[i]);
        const y = dbToY(refDb[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw Simulated Composite Spectrum (amber-gold showing chroma subcarrier spike)
      ctx.strokeStyle = "rgba(232, 163, 61, 0.75)";
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = fToX(freqs[i]);
        const y = dbToY(compDb[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw Simulated Luma Spectrum (bright phosphor green)
      ctx.strokeStyle = "#56f596";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = fToX(freqs[i]);
        const y = dbToY(simDb[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    } else if (ab.heatmap) {
      // Draw A/B Difference Heatmap (Magma / Inferno thermal palette)
      const mapKey = mode === "heat_luma" ? "luma" : (mode === "heat_chroma" ? "chroma" : "total");
      const grid = ab.heatmap[mapKey] || ab.heatmap.total || [];
      const rows = grid.length;
      const cols = rows ? grid[0].length : 0;
      if (rows > 0 && cols > 0) {
        const cellW = W / cols;
        const cellH = H / rows;
        for (let r = 0; r < rows; r++) {
          const rowArr = grid[r];
          const py = r * cellH;
          for (let c = 0; c < cols; c++) {
            const v = Math.min(1, Math.max(0, (rowArr[c] || 0) * 2.4));
            const rr = Math.round(Math.min(255, v * 360));
            const gg = Math.round(Math.max(0, Math.min(255, (v - 0.25) * 320)));
            const bb = Math.round(Math.max(0, Math.min(255, (0.55 - Math.abs(v - 0.45)) * 260)));
            ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
            ctx.fillRect(c * cellW, py, Math.ceil(cellW), Math.ceil(cellH));
          }
        }
      }
    }

    const badge = $id("abReadout");
    if (badge) {
      const cal = data.calibration;
      if (cal && cal.measured_bw_3db_mhz !== undefined) {
        const tbc = cal.tbc_comparison || {};
        badge.textContent =
          `PSNR ${st.psnr_db?.toFixed(1) ?? "0.0"} dB · ` +
          `实测-3dB带宽 ${cal.measured_bw_3db_mhz.toFixed(2)} MHz · ` +
          `DG ${cal.diff_gain_pct.toFixed(1)}% · DP ${cal.diff_phase_deg.toFixed(1)}° · ` +
          `2T K因子 ${cal.k_2t_pct.toFixed(1)}% · ` +
          `TBC抑抖 +${(tbc.suppression_db ?? 0).toFixed(1)} dB`;
      } else {
        badge.textContent =
          `PSNR ${st.psnr_db?.toFixed(1) ?? "0.0"} dB · ` +
          `ΔY ${st.l1_luma_ire?.toFixed(1) ?? "0.0"} IRE · ` +
          `ΔC ${st.l1_chroma_ire?.toFixed(1) ?? "0.0"} IRE · ` +
          `3MHz衰减 ${st.atten_at_3mhz_db?.toFixed(1) ?? "0.0"} dB`;
      }
    }
  }

  window.BroadcastScopes = {
    init: initScopesPanel,
    refresh: refreshScopes,
    state: ScopeState,
  };
})();
