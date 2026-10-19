"""
Graph executor: run a node graph and produce frames.

This connects `atsim.graph.Graph` to the stage implementations, so the wiring a
user draws is what actually determines the signal path.  Three things make it
capable of multi-effector and multi-generation iterative simulation:

**Per-node isolated RenderContext and seed.**  Each node receives its own
`RenderContext` with its per-node parameter overrides applied to its own
`params_layer` and a deterministic node-specific seed `derive_seed(seed, "node", name)`.
When a user wires `tape1 -> deck1 -> tape2 -> deck2`, `tape1` and `tape2` keep
independent wear/jitter/generation parameters and independent random defect
timelines instead of overwriting one shared layer dict.

**Per-node iterative looping (`node.iterations`).**  Any effector node can run
`1..25` iterations in a loop (`for it in range(node.iterations)`), feeding pass
`it`'s output signal into pass `it + 1` with a distinct per-iteration seed so
each pass compounds bandwidth loss, chroma bleed, ringing, and transport jitter
physically rather than repeating identical random numbers.

**Per-node frame memory.**  Each node memoises its output frames, keyed by the
node's own hash (which already includes all its ancestors, parameters, and
iteration count).  Re-rendering after touching one device therefore reuses
every node outside that device's subtree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..core.pipeline import Chain, FrameBuffer, RenderContext, Stage
from ..core.rng import derive_seed
from ..core.signal import SignalFrame, composite_encode
from ..core.standards import VideoStandard
from ..render.renderer import (STAGE_ORDER, STAGE_LAYERS, STAGE_LABELS,
                               CameraStage, CodecStage, DisplayStage,
                               FormatStage, PlayerStage, TapeStage, _hash)
from .graph import CATALOG_BY_KIND, Graph, Node

# node kind -> the stage class that implements it (None = handled specially)
STAGE_FOR_KIND: dict[str, type[Stage] | None] = {
    "camera": CameraStage,
    "encoder": FormatStage,
    "dv_codec": CodecStage,
    "tape": TapeStage,
    "vhs_gen": TapeStage,
    "deck": PlayerStage,
    "display": DisplayStage,
    "source": None,
    "hdmi_out": None,
    "hdmi_to_av": None,
    "rf_modulator": None,
    "rf_tuner": None,
    "vhs_dub_loop": None,
    "vhs_gen": None,
    "vhs_iterative_loop": None,
}


@dataclass
class GraphRun:
    """Result of executing a graph for one frame."""
    rgb: np.ndarray
    node_frames: dict[str, np.ndarray] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class GraphExecutor:
    """
    Executes a `Graph` over a frame source.

    The executor owns the stage instances (and therefore their temporal state:
    transport drift, noise fields, motion blur).  It must be pumped forward one
    frame at a time in order, exactly like the linear renderer, because that
    state is what makes the result reproducible.
    """

    def __init__(self, graph: Graph, std: VideoStandard, params: dict,
                 source, width: int, height: int, fps: float,
                 seed: int = 20240101, scale: float = 1.0):
        self.graph = graph
        self.std = std
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.seed = int(seed)
        self.scale = float(scale)
        self.base_params = {k: dict(v) for k, v in params.items()}
        self.params = {k: dict(v) for k, v in params.items()}

        self.ctx = RenderContext(std=std, params=self.params, seed=self.seed,
                                 frame_count=int(getattr(source, "frame_count", 0) or 0),
                                 fps=fps, width=int(width),
                                 height=int(height), preview=False, scale=scale,
                                 source=source)
        self.stages: dict[str, list[Stage]] = {}
        self._build_node_stages()

        self.order = graph.order()
        self.hashes = graph.node_hashes()
        self._cache: dict[str, dict[int, np.ndarray]] = {n: {} for n in graph.nodes}
        self._cache_key: dict[str, dict[int, str]] = {n: {} for n in graph.nodes}
        self._prev_src: np.ndarray | None = None
        self.node_ms: dict[str, float] = {}
        self.skipped: dict[str, int] = {}
        self.last_rgb: np.ndarray | None = None

    def _node_params(self, node: Node) -> dict:
        """Build an isolated parameter dictionary for `node`."""
        np_dict = {k: dict(v) for k, v in self.base_params.items()}
        layer_name = node.spec.params_layer or node.kind
        if node.kind == "vhs_gen":
            # Default vhs_gen to at least 2 generation passes if not overridden
            np_dict.setdefault("tape", {})
            if "generation_passes" not in (node.params or {}) and float(np_dict["tape"].get("generation_passes", 1.0)) <= 1.0:
                np_dict["tape"]["generation_passes"] = 2.0
        if node.params and layer_name:
            np_dict.setdefault(layer_name, {}).update(node.params)
        return np_dict

    def _build_node_stages(self) -> None:
        """Create isolated Stage instance(s) per node and per iteration pass."""
        self.stages = {}
        frame_cnt = int(getattr(self.source, "frame_count", 0) or 0)
        for name, node in self.graph.nodes.items():
            cls = STAGE_FOR_KIND.get(node.kind)
            if cls is None:
                continue
            iters = max(1, min(25, int(getattr(node, "iterations", 1) or 1)))
            pass_stages: list[Stage] = []
            np_dict = self._node_params(node)
            for it in range(iters):
                node_seed = derive_seed(self.seed, "node", name, it)
                n_ctx = RenderContext(
                    std=self.std, params={k: dict(v) for k, v in np_dict.items()},
                    seed=node_seed, frame_count=frame_cnt, fps=self.fps,
                    width=int(self.width), height=int(self.height),
                    preview=False, scale=self.scale, source=self.source,
                )
                st = cls(n_ctx)
                st.prepare()
                pass_stages.append(st)
            self.stages[name] = pass_stages

    def sync_params(self, params: dict) -> None:
        """Update base parameters and rebuild node stages if hashes changed."""
        self.base_params = {k: dict(v) for k, v in params.items()}
        self.params = {k: dict(v) for k, v in params.items()}
        new_hashes = self.graph.node_hashes()
        if new_hashes != self.hashes or set(self.graph.nodes) != set(self._cache):
            self.order = self.graph.order()
            for n in self.graph.nodes:
                self._cache.setdefault(n, {})
                self._cache_key.setdefault(n, {})
            self._build_node_stages()
            self.hashes = new_hashes

    # ------------------------------------------------------------------
    def _rf(self):
        if getattr(self, "_rflink", None) is None:
            from ..codecs.rf import RFLink, RFParams
            p = self.ctx.layer("rf") or {}
            self._rflink = RFLink(self.std, RFParams(
                channel_mhz=float(p.get("channel_mhz", 6.0)),
                carrier_offset_mhz=float(p.get("carrier_offset_mhz", 1.25)),
                noise_figure_db=float(p.get("noise_figure_db", 8.0)),
                signal_dbm=float(p.get("signal_dbm", -60.0)),
                sync_loss_threshold_db=float(p.get("sync_loss_threshold_db", 30.0)),
                sync_loss_amount=float(p.get("sync_loss_amount", 1.0)),
                adjacent_channel=float(p.get("adjacent_channel", 0.0)),
                broadcast_noise=float(p.get("broadcast_noise", 0.0)),
            ), seed=self.seed)
        return self._rflink

    def _decompose_for_rf(self, comp: np.ndarray):
        from ..core.signal import demod_chroma, split_composite
        window = float(np.clip(
            float(self.ctx.layer("format").get("chroma_bandwidth", 0.75)) * 1.15e6,
            0.35e6, 1.2e6))
        y_band, chroma_band = split_composite(comp, self.std, comb_depth=0.0,
                                              chroma_bandwidth_hz=window)
        i, q = demod_chroma(chroma_band / 0.63, self.std,
                            i_bw_hz=1.5e6, q_bw_hz=1.5e6)
        return y_band, i, q

    def _format_clone(self) -> FormatStage:
        if getattr(self, "_fmt", None) is None:
            self._fmt = FormatStage(self.ctx)
            self._fmt.prepare()
        return self._fmt

    def _prepare_stage_input(self, kind: str, buf: FrameBuffer) -> FrameBuffer:
        """Ensure `buf` carries the representation expected by `kind` before a pass."""
        if kind in ("tape", "vhs_gen", "deck") and buf.composite is None:
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
                buf.meta["signal"] = sig
            buf.composite = composite_encode(sig.y, sig.i, sig.q, self.std, chroma_gain=0.63)
        elif kind in ("encoder", "camera", "display") and buf.meta.get("signal") is not None:
            buf.rgb = buf.meta["signal"].to_rgb()
        return buf

    def _picture_from_output(self, kind: str, out: FrameBuffer) -> np.ndarray:
        """Extract an RGB picture from `out` even when a stage only updates `composite`."""
        if kind == "display":
            return out.rgb
        sig = out.meta.get("signal")
        if sig is not None and kind not in ("tape", "vhs_gen"):
            return sig.to_rgb()
        if out.composite is not None and kind in ("tape", "vhs_gen"):
            y, i, q = self._decompose_for_rf(out.composite)
            sig_c = SignalFrame(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32))
            out.meta["signal"] = sig_c
            out.rgb = sig_c.to_rgb()
            return out.rgb
        if sig is not None:
            return sig.to_rgb()
        return out.rgb

    def _frame_for(self, name: str, index: int, src_rgb: np.ndarray | None
                   ) -> tuple[FrameBuffer, np.ndarray]:
        """Produce this node's output for one frame (including multi-pass iteration)."""
        node = self.graph.nodes[name]
        kind = node.kind
        ins = self.graph.inputs_of(name)

        buf: FrameBuffer | None = None
        if ins:
            edge = next(iter(ins.values()))
            upstream = edge.src
            cached_sig = self._cache[upstream].get(index)
            buf = self._buffer_from_cache(upstream, index, cached_sig, src_rgb)
            if buf is None:
                raise RuntimeError(f"{name}: upstream {upstream} has no frame {index}")

        if kind == "source":
            rgb = src_rgb if src_rgb is not None else self.source.frame(index)
            out = FrameBuffer(np.asarray(rgb, dtype=np.float32), None, index,
                              index / max(1e-6, self.fps), self.std,
                              {"field": index % 2})
            return out, out.rgb

        if buf is None:
            raise RuntimeError(f"节点 {name} 没有输入，无法渲染")

        # If a node is disabled on the canvas (e.g. CRT display turned off), bypass cleanly
        if not getattr(node, "enabled", True):
            sig = buf.meta.get("signal")
            pic = sig.to_rgb() if sig is not None else buf.rgb
            return buf, pic

        pass_stages = self.stages.get(name)
        if pass_stages:
            cur_buf = buf
            for st in pass_stages:
                cur_buf = self._prepare_stage_input(kind, cur_buf)
                if kind == "camera":
                    st.set_previous(self._prev_src)
                cur_buf = st.apply(cur_buf)
                pic = self._picture_from_output(kind, cur_buf)
                cur_buf.rgb = pic
            return cur_buf, pic

        # --- converter nodes -------------------------------------------
        iters = max(1, min(25, int(getattr(node, "iterations", 1) or 1)))
        cur_buf = buf
        picture = cur_buf.rgb
        for _it in range(iters):
            if kind == "hdmi_out":
                sig = cur_buf.meta.get("signal")
                if sig is None:
                    sig = SignalFrame.from_rgb(cur_buf.rgb, self.std)
                cur_buf = FrameBuffer(cur_buf.rgb, None, index, cur_buf.time, self.std,
                                      {**cur_buf.meta, "signal": sig, "format": "hdmi"})
                picture = sig.to_rgb()
            elif kind == "hdmi_to_av":
                sig = cur_buf.meta.get("signal")
                if sig is None:
                    sig = SignalFrame.from_rgb(cur_buf.rgb, self.std)
                fmt = self._format_clone()
                cur_buf = fmt.apply(FrameBuffer(sig.to_rgb(), None, index, cur_buf.time,
                                                self.std, dict(cur_buf.meta)))
                picture = cur_buf.meta["signal"].to_rgb()
            elif kind == "rf_modulator":
                sig = cur_buf.meta.get("signal")
                if sig is None:
                    sig = SignalFrame.from_rgb(cur_buf.rgb, self.std)
                comp = cur_buf.composite
                if comp is None:
                    comp = composite_encode(sig.y, sig.i, sig.q, self.std,
                                            chroma_gain=0.63)
                tx = self._rf().modulate(comp, index)
                cur_buf = FrameBuffer(sig.to_rgb(), tx, index, cur_buf.time, self.std,
                                      {**cur_buf.meta, "format": "rf", "signal": sig})
                picture = sig.to_rgb()
            elif kind == "rf_tuner":
                rf = self._rf()
                comp = cur_buf.composite
                if comp is None:
                    sig0 = cur_buf.meta.get("signal") or SignalFrame.from_rgb(cur_buf.rgb, self.std)
                    comp = composite_encode(sig0.y, sig0.i, sig0.q, self.std,
                                            chroma_gain=0.63)
                base, info = rf.demodulate(
                    comp, index,
                    np.random.default_rng((int(self.seed) * 8191 + index * 131 + _it * 997) & 0xFFFFFFFF))
                y, i, q = self._decompose_for_rf(base)
                ph = rf.chroma_phase_error()
                if ph is not None:
                    ph = np.asarray(ph, dtype=np.float32).reshape(-1, 1)
                    c, s = np.cos(ph), np.sin(ph)
                    i, q = i * c - q * s, i * s + q * c
                sig_rf = SignalFrame(y.astype(np.float32),
                                     i.astype(np.float32),
                                     q.astype(np.float32))
                conn_cfg = self.ctx.layer("connector") or {}
                rf_m = float(conn_cfg.get("rf_moisture", 0.0))
                rf_c = float(conn_cfg.get("rf_corrosion", 0.0))
                rf_sh = float(conn_cfg.get("rf_shielding", 0.0))
                meta_out = {**cur_buf.meta, "format": "composite"}
                if rf_m > 1e-4 or rf_c > 1e-4:
                    from ..codecs.connectors import CONNECTORS, degrade_rf_signal
                    rgb_damp, c_info = degrade_rf_signal(
                        sig_rf.to_rgb(), CONNECTORS["rf"], rf_m, rf_c, rf_sh, index, int(self.seed) + 23 + _it)
                    sig_rf = SignalFrame.from_rgb(rgb_damp, self.std)
                    info.update(c_info)
                    sig_rf._rf_direct = True
                cur_buf = FrameBuffer(sig_rf.to_rgb(), base, index, cur_buf.time, self.std, meta_out)
                cur_buf.meta["signal"] = sig_rf
                cur_buf.meta["rf"] = info
                picture = sig_rf.to_rgb()
            elif kind in ("vhs_dub_loop", "vhs_gen", "vhs_iterative_loop"):
                from ..core.signal import demod_chroma, remod_chroma, split_composite
                comp = cur_buf.composite
                if comp is None:
                    sig0 = cur_buf.meta.get("signal") or SignalFrame.from_rgb(cur_buf.rgb, self.std)
                    comp = composite_encode(sig0.y, sig0.i, sig0.q, self.std, chroma_gain=0.63)
                window = float(np.clip(
                    float(self.ctx.layer("format").get("chroma_bandwidth", 0.75)) * 1.15e6,
                    0.35e6, 1.2e6))
                y_b, c_b = split_composite(comp, self.std, comb_depth=0.0, chroma_bandwidth_hz=window)
                i_b, q_b = demod_chroma(c_b / 0.63, self.std, i_bw_hz=1.5e6, q_bw_hz=1.5e6)
                if name not in self.stages:
                    st_t = TapeStage(self.ctx)
                    st_t.prepare()
                    self.stages[name] = st_t
                st_tape = self.stages[name]
                y_d, i_d, q_d = st_tape._apply_generation_iteration(y_b, i_b, q_b, index, gen_index=int(node.params.get("gen_index", 1)))
                comp_out = (y_d + remod_chroma(i_d, q_d, self.std, chroma_gain=0.63)).astype(np.float32)
                sig_out = SignalFrame(y_d, i_d, q_d)
                cur_buf = FrameBuffer(sig_out.to_rgb(), comp_out, index, cur_buf.time, self.std, dict(cur_buf.meta))
                cur_buf.meta["signal"] = sig_out
                picture = cur_buf.rgb
            else:
                raise RuntimeError(f"no implementation for node kind {kind!r}")
        return cur_buf, picture

    def _buffer_from_cache(self, upstream: str, index: int,
                           cached: np.ndarray | None,
                           src_rgb: np.ndarray | None) -> FrameBuffer | None:
        if cached is None:
            return None
        up = self.graph.nodes[upstream]
        buf = FrameBuffer(np.asarray(cached, dtype=np.float32), None, index,
                          index / max(1e-6, self.fps), self.std,
                          {"field": index % 2})
        if up.kind in ("encoder", "hdmi_to_av", "rf_tuner", "tape", "vhs_gen", "deck"):
            sig = SignalFrame.from_rgb(buf.rgb, self.std)
            if up.kind == "rf_tuner":
                _cc = self.ctx.layer("connector") or {}
                if float(_cc.get("rf_moisture", 0.0)) > 1e-4 or float(_cc.get("rf_corrosion", 0.0)) > 1e-4:
                    sig._rf_direct = True
            buf.composite = composite_encode(sig.y, sig.i, sig.q, self.std,
                                             chroma_gain=0.63)
            buf.meta["signal"] = sig
        else:
            buf.meta["signal"] = SignalFrame.from_rgb(buf.rgb, self.std)
        return buf

    def invalidate(self) -> list[str]:
        old = self.hashes
        self.hashes = self.graph.node_hashes()
        dirty = [n for n in self.graph.nodes if old.get(n) != self.hashes[n]]
        if dirty:
            self._build_node_stages()
        return dirty

    def node_is_cached(self, name: str, index: int) -> bool:
        return (index in self._cache.get(name, {})
                and self._cache_key.get(name, {}).get(index) == self.hashes.get(name))

    # ------------------------------------------------------------------
    def pending_nodes(self) -> list[str]:
        out: list[str] = []
        for name, node in self.graph.nodes.items():
            if not node.spec.inputs:
                continue
            if not self.graph.inputs_of(name):
                out.append(name)
        return out

    def render_frame(self, index: int, use_cache: bool = True) -> GraphRun:
        run = GraphRun(rgb=np.zeros((self.height, self.width, 3), np.float32))

        pending = self.pending_nodes()
        if pending:
            names = "、".join(pending)
            run.errors.append(
                f"以下节点没有接入输入，无法渲染：{names}。"
                f"请在画布上把它们连起来，或删除它们。")
            return run

        src_rgb = None
        for name in self.order:
            self._cache.setdefault(name, {})
            self._cache_key.setdefault(name, {})
            if use_cache and self.node_is_cached(name, index):
                run.node_frames[name] = self._cache[name][index]
                run.trace.append(f"{name}(cached)")
                self.skipped[name] = self.skipped.get(name, 0) + 1
                continue
            import time
            t0 = time.perf_counter()
            try:
                buf, picture = self._frame_for(name, index, src_rgb)
            except Exception as exc:
                run.errors.append(f"{name}: {type(exc).__name__}: {exc}")
                return run
            self._cache[name][index] = picture
            self._cache_key[name][index] = self.hashes[name]
            run.node_frames[name] = picture
            iters = max(1, int(getattr(self.graph.nodes[name], "iterations", 1) or 1))
            run.trace.append(f"{name}×{iters}" if iters > 1 else name)
            self.node_ms[name] = self.node_ms.get(name, 0.0) * 0.7 + \
                (time.perf_counter() - t0) * 1000.0 * 0.3
        outs = self.graph.output_nodes()
        if outs:
            run.rgb = self._cache[outs[0]][index]
        return run

    def render_frames(self, indices: list[int]
                      ) -> Iterator[tuple[int, np.ndarray]]:
        wanted = sorted({int(i) for i in indices})
        if not wanted:
            return
        outs = self.graph.output_nodes()
        if not outs:
            return
        sink = outs[0]
        for index in range(0, max(wanted) + 1):
            if index in self._cache.get(sink, {}):
                if index in wanted:
                    yield index, self._cache[sink][index]
                continue
            run = self.render_frame(index)
            if run.errors:
                raise RuntimeError("; ".join(run.errors))
            if index in wanted:
                yield index, run.rgb

    def describe(self) -> dict:
        d = self.graph.describe()
        d["node_ms"] = {k: round(v, 2) for k, v in self.node_ms.items()}
        d["cached_frames"] = {k: len(v) for k, v in self._cache.items()}
        return d
