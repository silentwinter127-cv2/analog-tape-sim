"""
Graph executor: run a node graph and produce frames.

This connects `atsim.graph.Graph` to the stage implementations, so the wiring a
user draws is what actually determines the signal path.  Two things make it more
than a for-loop over stages:

**Per-node frame memory.**  Each node memoises its output frames, keyed by the
node's own hash (which already includes all its ancestors).  Re-rendering after
touching one device therefore reuses every node outside that device's subtree.
In a graph with two parallel branches, only the touched branch runs.

**Node boundaries are format boundaries.**  A node does not receive "the frame";
it receives the frame *as the signal type of the edge that feeds it*.  That is
what makes a converter meaningful: `hdmi_to_av` is the node that turns a picture
into a composite signal, and `rf_modulator` is the node that puts it on a carrier.

The RF pair is a real round trip through `atsim.codecs.rf`: the modulator applies
channel shaping, and the tuner applies IF noise, adjacent-channel beat and AFC
sync slip (which corrupts hue as well as position, since both come from the same
timing error).  The picture degrades monotonically as the received level falls --
verified across -45 to -80 dBm -- and the tuner's output signal is re-derived from
the *received* composite rather than inherited from upstream, which is what makes
the damage actually reach the downstream nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..core.pipeline import Chain, FrameBuffer, RenderContext, Stage
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
    "deck": PlayerStage,
    "display": DisplayStage,
    "source": None,
    "hdmi_out": None,
    "hdmi_to_av": None,
    "rf_modulator": None,
    "rf_tuner": None,
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
        # node params override the layer params, so a node can be tuned
        # individually instead of only through the shared device layer
        self.params = {k: dict(v) for k, v in params.items()}
        for node in graph.nodes.values():
            if node.params:
                self.params.setdefault(node.kind, {}).update(node.params)

        self.ctx = RenderContext(std=std, params=self.params, seed=self.seed,
                                frame_count=0, fps=fps, width=int(width),
                                height=int(height), preview=False, scale=scale)
        self.stages: dict[str, Stage] = {}
        for name, node in graph.nodes.items():
            cls = STAGE_FOR_KIND.get(node.kind)
            if cls is not None:
                self.stages[name] = cls(self.ctx)
        for st in self.stages.values():
            st.prepare()

        self.order = graph.order()
        self.hashes = graph.node_hashes()
        self._cache: dict[str, dict[int, np.ndarray]] = {n: {} for n in graph.nodes}
        # which node hash produced each cached frame; a node may be skipped only
        # when its hash is unchanged AND its ancestors' hashes are unchanged,
        # which is exactly what the node hash already encodes
        self._cache_key: dict[str, dict[int, str]] = {n: {} for n in graph.nodes}
        self._prev_src: np.ndarray | None = None
        self.node_ms: dict[str, float] = {}
        self.skipped: dict[str, int] = {}
        self.last_rgb: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _rf(self):
        """
        The RF link, built lazily and shared by the modulator and tuner nodes.

        Shared on purpose: it carries the AFC error and the interference beat,
        which are continuous processes across frames.  A per-node instance would
        restart those every frame and the tearing would look like noise.
        """
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
        """
        Composite -> (Y, I, Q) after the RF link has damaged it.

        Uses the same wide-window separation as the medium model: the window has
        to match the recorded chroma width, or luma leaks into chroma and gets
        re-modulated as false colour.
        """
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
        """
        A FormatStage for converter nodes that must encode a picture themselves.

        Kept as a lazily-built singleton so its temporal state (the chroma phase
        processes) advances with the clip rather than restarting per frame.
        """
        if getattr(self, "_fmt", None) is None:
            self._fmt = FormatStage(self.ctx)
            self._fmt.prepare()
        return self._fmt

    def _frame_for(self, name: str, index: int, src_rgb: np.ndarray | None
                   ) -> tuple[FrameBuffer, np.ndarray]:
        """
        Produce this node's output for one frame.

        Returns (buffer, output_picture).  The buffer carries the signal state
        that downstream nodes need; the picture is what gets cached and shown,
        which is what lets a branch's cache entry be one array.
        """
        node = self.graph.nodes[name]
        kind = node.kind
        ins = self.graph.inputs_of(name)

        # --- gather the incoming buffer, respecting the edge type --------
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

        assert buf is not None, f"{name} has no input"

        stage = self.stages.get(name)
        if stage is not None:
            # the stage is given the buffer as produced by its upstream, and
            # its own config comes from the node's parameter layer
            out = stage.apply(buf)
            picture = out.rgb if out.meta.get("signal") is None else \
                out.meta["signal"].to_rgb()
            return out, picture

        # --- converter nodes -------------------------------------------
        if kind == "hdmi_out":
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
            out = FrameBuffer(buf.rgb, None, index, buf.time, self.std,
                              {**buf.meta, "signal": sig, "format": "hdmi"})
            return out, sig.to_rgb()

        if kind == "hdmi_to_av":
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
            # A converter box really does encode the picture itself, so the
            # format layer runs here rather than being borrowed from elsewhere.
            fmt = self._format_clone()
            comp = fmt.apply(FrameBuffer(sig.to_rgb(), None, index, buf.time,
                                         self.std, dict(buf.meta)))
            return comp, comp.meta["signal"].to_rgb()

        if kind == "rf_modulator":
            # put the composite onto a carrier: channel shaping applies here
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
            comp = buf.composite
            if comp is None:
                comp = composite_encode(sig.y, sig.i, sig.q, self.std,
                                        chroma_gain=0.63)
            tx = self._rf().modulate(comp, index)
            out = FrameBuffer(sig.to_rgb(), tx, index, buf.time, self.std,
                              {**buf.meta, "format": "rf"})
            out.meta["signal"] = sig
            return out, sig.to_rgb()

        if kind == "rf_tuner":
            # demodulate: receiver noise, adjacent-channel beat and AFC slip
            rf = self._rf()
            comp = buf.composite
            if comp is None:
                sig0 = buf.meta.get("signal") or SignalFrame.from_rgb(buf.rgb, self.std)
                comp = composite_encode(sig0.y, sig0.i, sig0.q, self.std,
                                        chroma_gain=0.63)
            base, info = rf.demodulate(
                comp, index,
                np.random.default_rng((int(self.seed) * 8191 + index * 131) & 0xFFFFFFFF))
            # The signal handed downstream must be re-derived from the *received*
            # composite.  Keeping the upstream (clean) signal here meant the RF
            # damage was computed and then thrown away -- the picture was
            # bit-identical across a 35 dB range of received level.
            y, i, q = self._decompose_for_rf(base)
            # the same timing slip that moved the lines invalidates the
            # subcarrier reference too, so colour is corrupted with it
            ph = rf.chroma_phase_error()
            if ph is not None:
                # per-line phase error: reshape to (H, 1) so it broadcasts
                # against the (H, W) chroma planes
                ph = np.asarray(ph, dtype=np.float32).reshape(-1, 1)
                c, s = np.cos(ph), np.sin(ph)
                i, q = i * c - q * s, i * s + q * c
            out = FrameBuffer(buf.rgb, base, index, buf.time, self.std,
                              {**buf.meta, "format": "composite"})
            out.meta["signal"] = SignalFrame(y.astype(np.float32),
                                             i.astype(np.float32),
                                             q.astype(np.float32))
            out.meta["rf"] = info
            return out, out.meta["signal"].to_rgb()

        raise RuntimeError(f"no implementation for node kind {kind!r}")

    def _buffer_from_cache(self, upstream: str, index: int,
                           cached: np.ndarray | None,
                           src_rgb: np.ndarray | None) -> FrameBuffer | None:
        """
        Rebuild a FrameBuffer from an upstream node's cached picture.

        The cached value is the node's *output picture*.  Because the medium and
        player stages need a specific signal state (composite carrier, YIQ), the
        buffer is reconstructed from the picture and the upstream node's kind --
        the same information the cache key is built from.
        """
        if cached is None:
            return None
        up = self.graph.nodes[upstream]
        buf = FrameBuffer(np.asarray(cached, dtype=np.float32), None, index,
                          index / max(1e-6, self.fps), self.std,
                          {"field": index % 2})
        # a picture alone cannot carry a composite carrier, so nodes whose
        # downstream needs one must be re-derived rather than reconstructed
        if up.kind in ("encoder", "hdmi_to_av", "rf_tuner"):
            sig = SignalFrame.from_rgb(buf.rgb, self.std)
            buf.composite = composite_encode(sig.y, sig.i, sig.q, self.std,
                                             chroma_gain=0.63)
            buf.meta["signal"] = sig
        elif up.kind in ("camera", "source", "display", "hdmi_out"):
            buf.meta["signal"] = SignalFrame.from_rgb(buf.rgb, self.std)
        elif up.kind in ("deck", "dv_codec", "tape", "rf_modulator"):
            buf.meta["signal"] = SignalFrame.from_rgb(buf.rgb, self.std)
        return buf

    def invalidate(self) -> list[str]:
        """
        Recompute node hashes and report which nodes must re-run.

        Called after any parameter edit.  Because a node's hash covers all of its
        ancestors, a node is dirty exactly when it or something upstream of it
        changed -- so this list is the node editor's "what has to be recomputed"
        indicator, and everything absent from it stays cached.
        """
        old = self.hashes
        self.hashes = self.graph.node_hashes()
        return [n for n in self.graph.nodes if old.get(n) != self.hashes[n]]

    def node_is_cached(self, name: str, index: int) -> bool:
        return (index in self._cache[name]
                and self._cache_key[name].get(index) == self.hashes[name])

    # ------------------------------------------------------------------
    def render_frame(self, index: int, use_cache: bool = True) -> GraphRun:
        """
        Execute the graph for one frame, node by node.

        Nodes whose hash and frame are already cached are skipped outright and
        their stored picture is passed downstream.  Because the hash includes the
        whole ancestor set, this is safe: an unchanged hash means neither this
        node nor anything feeding it has been touched since the frame was made.
        This is the payoff of per-node keys -- after a tweak, only the affected
        subtree runs instead of the whole chain.
        """
        run = GraphRun(rgb=np.zeros((self.height, self.width, 3), np.float32))
        src_rgb = None
        for name in self.order:
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
            run.trace.append(name)
            self.node_ms[name] = self.node_ms.get(name, 0.0) * 0.7 + \
                (time.perf_counter() - t0) * 1000.0 * 0.3
        outs = self.graph.output_nodes()
        if outs:
            run.rgb = self._cache[outs[0]][index]
        return run

    def render_frames(self, indices: list[int]
                      ) -> Iterator[tuple[int, np.ndarray]]:
        """Pump the graph forward, yielding the requested output frames."""
        wanted = sorted({int(i) for i in indices})
        if not wanted:
            return
        outs = self.graph.output_nodes()
        if not outs:
            return
        sink = outs[0]
        last = max(self._cache[sink]) if self._cache[sink] else -1
        for index in range(0, max(wanted) + 1):
            if index in self._cache[sink]:
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
