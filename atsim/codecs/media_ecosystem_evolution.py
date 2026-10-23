"""
媒介技术生态演化与历史情景模拟器 (`media_ecosystem_evolution`)
==========================================================================

实现《完整的视听媒介生态模拟器》第六节（方向 C：宏观视听技术史与生态演化模拟器）：

1. **七维宏观媒介生态耦合动力学 (`MediaEcosystemSimulator`, `evaluate_ecosystem_at_year`)**：
   - 在 `1975–2035` 60 年历史时间轴上联动追踪 7 大宏观生态状态变量：
     1. **设备存量 (`hardware_stock`)**：现存可正常运转的放映机/录像机/CRT 存量；
     2. **介质存量 (`media_stock`)**：库房与民间留存的胶片/磁带/光盘存量；
     3. **跨代技术兼容性矩阵 (`compatibility_index`)**：当前主流生产/播放接口对历史载体的原生兼容度；
     4. **备件与技师维护能力 (`maintenance_capability`)**：原厂旋转磁头备件、专用模拟 IC 存量与掌握机械校准的老师傅比例；
     5. **格式迁移与数字化转换成本 (`migration_cost_index`)**：单盘磁带抢救性迁移的综合成本倍增系数；
     6. **制度与版权约束 (`institutional_constraint`)**：孤儿作品版权壁垒与经费限制；
     7. **机构工作习惯惯性 (`workflow_inertia`)**：电视台/档案馆沿用既有带基工作流的组织惯性。

2. **历史反事实情景推演与「可访问性断崖 (Obsolescence Cliff)」预警 (`simulate_media_ecosystem_trajectory`)**：
   - `eco_scenario = 0` (**基准历史轨迹 Historical Baseline**)：1980–1998 模拟磁带鼎盛，2005–2018 数字化转型，2020 年后进入备件枯竭期；
   - `eco_scenario = 1` (**备件与技师断代断崖危机 Spare-Parts & Technician Cliff**)：模拟最后一批原厂视频磁头备件与维修技师在 2012 年前后集中断代，引发存量磁带 **可访问性断崖 (`accessibility_index` 暴跌 >60%)**；
   - `eco_scenario = 2` (**数字无磁带化提前十年普及 Early Tapeless Shift**)：假设非线性固态记录与网络流媒体提前 10 年普及，模拟磁带生态提前萎缩；
   - `eco_scenario = 3` (**国家级抢救性数字化与开源硬件复刻干预 National Preservation & Open-Hardware Intervention**)：通过集中抢救数字化与开源 TBC/激光读取头复刻，成功跨越备件断崖并维持高可访问率。

3. **宏观生态向微观信号链物理参数的实时投射 (`apply_media_ecosystem_evolution`)**：
   - 将所选历史年份 `eco_year` 与情景 `eco_scenario` 解算出的机具失效率、备件老化度与介质衰减度直接投射为当前信号链的磁头磨损、时基漂移与色彩退化，并提供 `2×2` 媒介技术生态演化与反事实推演实验室 (`eco_mode = 4`)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame, horizontal_shift, lowpass
from ..stages.display import _resample_to
from .multiview_wall import _draw_bitmap_text


def draw_text(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    text: str,
    color: tuple[float, float, float],
    scale: int = 1,
) -> None:
    _draw_bitmap_text(canvas, str(text), int(x0), int(y0), color, scale=int(scale))


# 5 Representative Media Eras & Cross-Generational Compatibility Matrix (5x5)
MEDIA_FORMAT_ERAS = (
    "16MM_FILM",
    "UMATIC_BETACAM",
    "VHS_HI8",
    "MINIDV_VCD",
    "FILE_STREAM",
)

COMPATIBILITY_MATRIX = np.array(
    [
        # 16MM  UMAT  VHS   DV    STREAM
        [1.00, 0.72, 0.55, 0.35, 0.18],  # 16MM_FILM
        [0.68, 1.00, 0.85, 0.62, 0.25],  # UMATIC_BETACAM
        [0.52, 0.82, 1.00, 0.74, 0.22],  # VHS_HI8
        [0.35, 0.58, 0.72, 1.00, 0.68],  # MINIDV_VCD
        [0.15, 0.22, 0.20, 0.65, 1.00],  # FILE_STREAM
    ],
    dtype=np.float32,
)


@dataclass
class MediaEcosystemSpec:
    enabled: bool = False
    mode: int = 0
    year: float = 2008.0
    scenario: int = 0
    spare_depletion: float = 1.0
    institution_inertia: float = 0.55
    quad_corner: int = 0
    show_border: int = 2
    show_text: int = 0

    def is_active(self) -> bool:
        return bool(self.enabled and self.mode > 0)

    @classmethod
    def from_params(cls, cfg: dict[str, Any]) -> "MediaEcosystemSpec":
        en = float(cfg.get("eco_enabled", 0.0)) > 0.5
        md = int(round(float(cfg.get("eco_mode", 0.0))))
        if md > 0:
            en = True
        elif en and md == 0:
            md = 4
        return cls(
            enabled=en,
            mode=int(np.clip(md, 0, 4)),
            year=float(np.clip(cfg.get("eco_year", 2008.0), 1975.0, 2035.0)),
            scenario=int(np.clip(round(float(cfg.get("eco_scenario", 0.0))), 0, 3)),
            spare_depletion=float(np.clip(cfg.get("eco_spare_depletion", 1.0), 0.2, 2.0)),
            institution_inertia=float(np.clip(cfg.get("eco_institution_inertia", 0.55), 0.0, 1.0)),
            quad_corner=int(np.clip(round(float(cfg.get("eco_quad_corner", 0.0))), 0, 4)),
            show_border=int(np.clip(round(float(cfg.get("eco_show_border", 2.0))), 0, 2)),
            show_text=int(np.clip(round(float(cfg.get("eco_show_text", 0.0))), 0, 2)),
        )


def evaluate_ecosystem_at_year(
    year: float,
    scenario: int = 0,
    spare_depletion: float = 1.0,
    institution_inertia: float = 0.55,
) -> dict[str, Any]:
    """
    Evaluate the 7-dimensional media ecosystem state at `year` (1975..2035)
    under the specified historical or counterfactual scenario.
    """
    y = float(np.clip(year, 1975.0, 2035.0))
    sc = int(np.clip(scenario, 0, 3))
    depl = float(np.clip(spare_depletion, 0.2, 2.0))
    inertia = float(np.clip(institution_inertia, 0.0, 1.0))

    # Effective timeline shift for Early Tapeless scenario (scenario == 2)
    y_eff = y + 10.0 if sc == 2 else y

    # 1. Analog/Tape Hardware Stock (peaks around 1994, declines as factories close after 2005)
    cliff_accel = 1.65 * depl if sc == 1 else (0.52 * depl if sc == 3 else 1.0 * depl)
    hw_peak = np.exp(-(((y_eff - 1994.0) / 16.0) ** 2))
    post_2005_decay = 1.0 / (1.0 + np.exp((y_eff - (2011.0 if sc == 1 else 2017.0)) * 0.28 * cliff_accel))
    hardware_stock = float(np.clip(0.18 + 0.82 * hw_peak * post_2005_decay + (0.24 if (sc == 3 and y >= 2012) else 0.0), 0.03, 1.0))

    # 2. Recorded Analog Media Stock in Vaults (accumulates through 1980-2004, then slowly decays via hydrolysis)
    accumulation = 1.0 / (1.0 + np.exp(-(y - 1986.0) / 4.5))
    vault_attrition = np.exp(-max(0.0, y - 2000.0) * (0.018 if sc != 3 else 0.006))
    media_stock = float(np.clip(accumulation * vault_attrition, 0.08, 1.0))

    # 3. Spare Parts & Technician Maintenance Skill
    tech_retire_year = 2010.0 if sc == 1 else (2024.0 if sc == 3 else 2016.0)
    maintenance_capability = float(
        np.clip(
            1.0 / (1.0 + np.exp((y_eff - tech_retire_year) * 0.32 * cliff_accel))
            + (0.32 if (sc == 3 and y >= 2010) else 0.0),
            0.04,
            1.0,
        )
    )

    # 4. Cross-Generational Compatibility Index
    compatibility_index = float(np.clip(0.95 - 0.014 * max(0.0, y_eff - 1992.0) + (0.22 if sc == 3 else 0.0), 0.12, 0.98))

    # 5. Institutional & Copyright Constraint + 6. Workflow Inertia
    institutional_constraint = float(np.clip(0.30 + 0.35 * inertia + (0.20 if sc == 1 else (-0.18 if sc == 3 else 0.0)), 0.10, 0.95))
    workflow_inertia = float(np.clip(inertia * np.exp(-max(0.0, y_eff - 1998.0) / 28.0) + 0.12, 0.08, 0.95))

    # 7. Migration Cost Multiplier & Playable Accessibility Index (Obsolescence Cliff indicator)
    Bottleneck = min(hardware_stock, maintenance_capability, compatibility_index)
    # Non-linear cliff when both hardware stock and spare parts fall below critical threshold (~0.25)
    accessibility_index = float(
        np.clip(
            (0.45 * hardware_stock + 0.35 * maintenance_capability + 0.20 * compatibility_index)
            * (1.0 - 0.35 * institutional_constraint),
            0.02,
            1.0,
        )
    )
    migration_cost_index = float(round(1.0 + 14.0 * ((1.0 - accessibility_index) ** 2.1), 2))

    # Format market shares at year `y`
    shares_raw = np.array(
        [
            np.exp(-(((y - 1978.0) / 11.0) ** 2)),  # 16mm Film
            np.exp(-(((y - 1988.0) / 10.0) ** 2)),  # U-matic / Betacam
            np.exp(-(((y - 1995.0) / 11.0) ** 2)),  # VHS / Hi8
            np.exp(-(((y - 2004.0) / 8.5) ** 2)),   # MiniDV / VCD
            1.0 / (1.0 + np.exp(-(y_eff - 2009.0) / 4.2)),  # Network Stream / File
        ],
        dtype=np.float32,
    )
    shares = shares_raw / max(1e-6, float(np.sum(shares_raw)))

    # Projected signal-chain hardware wear & alignment error induced by ecosystem scarcity
    projected_head_wear = float(np.clip(1.0 - 0.85 * maintenance_capability, 0.05, 0.92))
    projected_jitter = float(np.clip((1.0 - hardware_stock) * 0.78, 0.04, 0.88))

    cliff_warning = (
        "OBSOLESCENCE_CLIFF_CRITICAL"
        if accessibility_index < 0.28
        else ("TRANSITION_WINDOW_WARNING" if accessibility_index < 0.55 else "ECOSYSTEM_HEALTHY")
    )

    return {
        "year": round(y, 1),
        "scenario": sc,
        "dimensions": {
            "hardware_stock": round(hardware_stock, 4),
            "media_stock": round(media_stock, 4),
            "compatibility_index": round(compatibility_index, 4),
            "maintenance_capability": round(maintenance_capability, 4),
            "migration_cost_index": migration_cost_index,
            "institutional_constraint": round(institutional_constraint, 4),
            "workflow_inertia": round(workflow_inertia, 4),
        },
        "accessibility_index": round(accessibility_index, 4),
        "bottleneck_factor": round(float(Bottleneck), 4),
        "cliff_warning": cliff_warning,
        "era_shares": {MEDIA_FORMAT_ERAS[i]: round(float(shares[i]), 3) for i in range(len(MEDIA_FORMAT_ERAS))},
        "projected_chain_state": {
            "head_wear": round(projected_head_wear, 4),
            "transport_jitter": round(projected_jitter, 4),
        },
    }


def simulate_media_ecosystem_trajectory(
    spec: MediaEcosystemSpec,
    years: list[int] | None = None,
) -> dict[str, Any]:
    """Simulate full 1975-2035 trajectory for both the selected scenario and counterfactuals."""
    if years is None:
        years = list(range(1975, 2036, 5))
    traj_current = [
        evaluate_ecosystem_at_year(y, spec.scenario, spec.spare_depletion, spec.institution_inertia)
        for y in years
    ]
    traj_cliff = [
        evaluate_ecosystem_at_year(y, 1, max(1.35, spec.spare_depletion), spec.institution_inertia)
        for y in years
    ]
    traj_rescue = [
        evaluate_ecosystem_at_year(y, 3, spec.spare_depletion, spec.institution_inertia)
        for y in years
    ]
    return {
        "years": years,
        "selected_trajectory": traj_current,
        "counterfactual_cliff_trajectory": traj_cliff,
        "counterfactual_rescue_trajectory": traj_rescue,
        "compatibility_matrix": COMPATIBILITY_MATRIX.tolist(),
    }


def _degrade_by_ecosystem_state(rgb: np.ndarray, eco_state: dict[str, Any], seed: int = 2024) -> np.ndarray:
    """Project macro-ecosystem hardware scarcity and maintenance decay onto the video frame."""
    h, w = rgb.shape[:2]
    wear = float(eco_state["projected_chain_state"]["head_wear"])
    jit = float(eco_state["projected_chain_state"]["transport_jitter"])
    acc = float(eco_state["accessibility_index"])

    out = lowpass(rgb, 0.35 + 2.2 * wear, axis=1)
    rng = np.random.default_rng((int(seed) + int(round(eco_state["year"] * 17))) & 0xFFFFFFFF)
    shifts = rng.standard_normal(h).astype(np.float32) * (jit * 4.2)
    for ch in range(3):
        out[..., ch] = horizontal_shift(out[..., ch], shifts)

    # Color desaturation & oxide snow when accessibility falls into the Obsolescence Cliff
    if acc < 0.65:
        sev = float(np.clip((0.65 - acc) / 0.60, 0.0, 1.0))
        luma = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])[..., None]
        out = luma + (out - luma) * (1.0 - 0.65 * sev)
        snow = lowpass(rng.standard_normal((h, w)).astype(np.float32), 1.2, axis=1)[..., None]
        out = out + snow * (0.14 * sev)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _render_trajectory_chart(h: int, w: int, spec: MediaEcosystemSpec, show_text: int = 2) -> np.ndarray:
    """Render Corner 3: 1975-2035 7-Dimensional Ecosystem & Obsolescence Cliff Curves."""
    panel = np.zeros((h, w, 3), dtype=np.float32)
    panel[:, :] = np.array([0.06, 0.08, 0.11], dtype=np.float32)
    ys_grid = np.linspace(1975.0, 2035.0, max(12, w - 16))
    y_top, y_bot = int(h * 0.18), int(h * 0.88)
    span_h = max(10, y_bot - y_top)

    for ix, yr in enumerate(ys_grid):
        px = 8 + ix
        if px >= w - 4:
            break
        st_base = evaluate_ecosystem_at_year(yr, 0, spec.spare_depletion, spec.institution_inertia)
        st_cliff = evaluate_ecosystem_at_year(yr, 1, max(1.35, spec.spare_depletion), spec.institution_inertia)
        st_resc = evaluate_ecosystem_at_year(yr, 3, spec.spare_depletion, spec.institution_inertia)

        py_base = int(np.clip(y_bot - st_base["accessibility_index"] * span_h, y_top, y_bot))
        py_cliff = int(np.clip(y_bot - st_cliff["accessibility_index"] * span_h, y_top, y_bot))
        py_resc = int(np.clip(y_bot - st_resc["accessibility_index"] * span_h, y_top, y_bot))

        panel[max(0, py_resc - 1) : min(h, py_resc + 2), px] = np.array([0.35, 0.95, 0.58], dtype=np.float32)
        panel[max(0, py_base - 1) : min(h, py_base + 2), px] = np.array([0.45, 0.82, 0.96], dtype=np.float32)
        panel[max(0, py_cliff - 1) : min(h, py_cliff + 2), px] = np.array([0.96, 0.42, 0.35], dtype=np.float32)

    # Vertical cursor for current `spec.year`
    cur_x = int(np.clip(8 + ((spec.year - 1975.0) / 60.0) * (w - 16), 8, w - 8))
    panel[y_top:y_bot, max(0, cur_x - 1) : min(w, cur_x + 1)] = np.array([0.96, 0.88, 0.32], dtype=np.float32)

    if show_text >= 1:
        draw_text(panel, 6, 4, f"3.1975-2035 CLIFF CURVE (YR:{int(spec.year)})", (0.95, 0.88, 0.40), scale=1)
    return panel


def _render_compatibility_matrix_tile(
    h: int,
    w: int,
    cur_state: dict[str, Any],
    show_text: int = 2,
) -> np.ndarray:
    """Render Corner 4: 5x5 Cross-Generational Compatibility Matrix & 7D Telemetry."""
    panel = np.zeros((h, w, 3), dtype=np.float32)
    panel[:, :] = np.array([0.07, 0.09, 0.12], dtype=np.float32)

    n = len(MEDIA_FORMAT_ERAS)
    y0_mat, y1_mat = int(h * 0.20), int(h * 0.78)
    x0_mat, x1_mat = int(w * 0.10), int(w * 0.56)
    ch = max(4, (y1_mat - y0_mat) // n)
    cw = max(4, (x1_mat - x0_mat) // n)

    for r in range(n):
        for c in range(n):
            val = float(COMPATIBILITY_MATRIX[r, c])
            ry0 = y0_mat + r * ch
            ry1 = min(h - 2, ry0 + ch - 2)
            cx0 = x0_mat + c * cw
            cx1 = min(w - 2, cx0 + cw - 2)
            col = np.array([0.95 * (1.0 - val), 0.88 * val, 0.35 + 0.45 * val], dtype=np.float32)
            panel[ry0:ry1, cx0:cx1] = col

    # Right-side 7D bar gauges
    dims = list(cur_state["dimensions"].items())
    bx0 = int(w * 0.62)
    bw_max = max(10, w - bx0 - 8)
    for idx_d, (_k, v_raw) in enumerate(dims[:6]):
        by0 = y0_mat + idx_d * max(6, (y1_mat - y0_mat) // 6)
        by1 = min(h - 2, by0 + 5)
        v_norm = float(np.clip(v_raw if v_raw <= 1.0 else (1.0 / v_raw), 0.0, 1.0))
        panel[by0:by1, bx0 : bx0 + bw_max] = np.array([0.16, 0.19, 0.24], dtype=np.float32)
        panel[by0:by1, bx0 : bx0 + int(round(bw_max * v_norm))] = np.array([0.38, 0.88, 0.76], dtype=np.float32)

    if show_text >= 1:
        draw_text(
            panel,
            6,
            4,
            f"4.COMPAT & COST:{cur_state['dimensions']['migration_cost_index']:.1f}x",
            (0.55, 0.95, 0.82),
            scale=1,
        )
        if show_text >= 2 and h >= 90:
            draw_text(
                panel,
                6,
                h - 14,
                f"ACC:{int(cur_state['accessibility_index']*100)}% {cur_state['cliff_warning'][:18]}",
                (0.95, 0.75, 0.42),
                scale=1,
            )
    return panel


def apply_media_ecosystem_evolution(
    rgb: np.ndarray,
    spec: MediaEcosystemSpec,
    frame_index: int = 0,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Option U Media Ecosystem Evolution & Counterfactual Historical Simulator."""
    src = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    cur_state = evaluate_ecosystem_at_year(
        spec.year,
        scenario=spec.scenario,
        spare_depletion=spec.spare_depletion,
        institution_inertia=spec.institution_inertia,
    )
    cliff_state = evaluate_ecosystem_at_year(
        max(2016.0, spec.year),
        scenario=1,
        spare_depletion=max(1.4, spec.spare_depletion),
        institution_inertia=spec.institution_inertia,
    )
    traj = simulate_media_ecosystem_trajectory(spec)

    report = {
        "active": bool(spec.enabled),
        "mode": int(spec.mode),
        "eco_quad_corner": int(spec.quad_corner),
        "eco_show_border": int(spec.show_border),
        "eco_show_text": int(spec.show_text),
        "current_state": cur_state,
        "counterfactual_cliff_state": cliff_state,
        "trajectory": traj,
    }
    if not spec.enabled or spec.mode <= 0:
        return src, report

    h, w = src.shape[:2]
    b1_cur = _degrade_by_ecosystem_state(src, cur_state, seed=seed + int(frame_index))
    b2_cliff = _degrade_by_ecosystem_state(src, cliff_state, seed=seed + 97 + int(frame_index))
    b3_chart = _render_trajectory_chart(h, w, spec, show_text=max(1, spec.show_text))
    b4_matrix = _render_compatibility_matrix_tile(h, w, cur_state, show_text=max(1, spec.show_text))

    if spec.mode == 1:
        out = b1_cur
    elif spec.mode == 2:
        out = b2_cliff
    elif spec.mode == 3:
        out = b3_chart
    else:
        c_sel = int(spec.quad_corner)
        if c_sel in (1, 2, 3, 4):
            out = {1: b1_cur, 2: b2_cliff, 3: b3_chart, 4: b4_matrix}[c_sel].copy()
        else:
            qh, qw = h // 2, w // 2
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            quads = [
                (0, qh, 0, qw, b1_cur, f"1.YR {int(spec.year)} ACC:{int(cur_state['accessibility_index']*100)}%", (0.45, 0.95, 0.82)),
                (0, qh, qw, w, b2_cliff, f"2.SPARE-PART CLIFF ACC:{int(cliff_state['accessibility_index']*100)}%", (0.96, 0.52, 0.42)),
                (qh, h, 0, qw, b3_chart, "3.1975-2035 7D TRAJECTORY", (0.95, 0.88, 0.40)),
                (qh, h, qw, w, b4_matrix, f"4.COMPAT & COST:{cur_state['dimensions']['migration_cost_index']:.1f}x", (0.55, 0.92, 0.95)),
            ]
            for y0, y1, x0, x1, img_q, lbl, col in quads:
                tile = _resample_to(img_q, y1 - y0, x1 - x0)
                if spec.show_text >= 1:
                    tile[:16, :] *= 0.22
                    draw_text(tile, 4, 4, lbl, col, scale=1)
                canvas[y0:y1, x0:x1] = tile
            if spec.show_border >= 1:
                canvas[qh - 1 : qh + 1, :] = np.array([0.20, 0.24, 0.30], dtype=np.float32)
                canvas[:, qw - 1 : qw + 1] = np.array([0.20, 0.24, 0.30], dtype=np.float32)
            out = canvas

    return np.clip(out, 0.0, 1.0).astype(np.float32), report


class EcosystemSimulatorStage(Stage):
    """Node-graph stage for `ecosystem_simulator` (**模块**)."""
    name = "ecosystem_simulator"
    layers = ("overlay",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        cfg = self.cfg_of("overlay")
        self.spec = MediaEcosystemSpec.from_params(cfg)
        if not self.spec.enabled:
            self.spec.enabled = True
            if self.spec.mode <= 0:
                self.spec.mode = 4

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sig: SignalFrame | None = buf.meta.get("signal")
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        out_rgb, info = apply_media_ecosystem_evolution(
            rgb_in,
            self.spec,
            frame_index=int(buf.index),
            seed=int(self.ctx.seed),
        )
        meta = dict(buf.meta)
        meta["media_ecosystem"] = info
        meta["signal"] = SignalFrame.from_rgb(out_rgb, buf.std)
        return FrameBuffer(out_rgb, buf.composite, buf.index, buf.time, buf.std, meta)
