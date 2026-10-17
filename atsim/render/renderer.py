"""
The renderer: turns a project (source clip + five-layer parameters + seed) into
frames, with the structure that makes interactive adjustment possible.

    ---------------------------------------------------------------------
    | camera | format | tape | player | display |     stage chain
    ---------------------------------------------------------------------
        ^         ^       ^        ^         ^
        |         |       |        |         |
      cached per stage, keyed by the hash of (seed + all upstream layers)

Consequences that matter for the UI:

* Moving a *display* slider re-renders only the display stage -- the expensive
  composite/tape work is reused, so a CRT tweak is cheap even at full
  resolution.
* Moving a *tape* slider invalidates tape/player/display and reuses the camera
  and format output.
* The first dirty stage and frame are reported back to the client, so the
  preview can show which part of the timeline is being recomputed.

The chain carries temporal state (transport drift, motion blur, head noise), so
frames are always produced by *pumping forward* from the last cached frame.  A
backward seek therefore re-pumps the prefix, which keeps the result identical
to a linear render -- reproducibility beats cleverness here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..core.pipeline import Chain, FrameBuffer, RenderContext, Stage
from ..core.profiles import SCHEMA
from ..core.standards import get_standard
from ..stages.camera import CameraStage
from ..stages.codec import CodecStage
from ..stages.connector import ConnectorStage
from ..stages.display import DisplayStage
from ..stages.format import FormatStage
from ..stages.overlay import OverlayStage
from ..stages.player import PlayerStage
from ..stages.tape import TapeStage

STAGE_ORDER = ("camera", "format", "codec", "connector", "tape", "player",
               "display", "overlay")
STAGE_LABELS = {
    "camera": "01 成像",
    "format": "02 编码",
    "codec": "02b DV",
    "connector": "02c 接口",
    "tape": "03 介质",
    "player": "04 读取",
    "display": "05 显示",
    "overlay": "06 字幕机",
}
# which parameter layers feed which stage
STAGE_LAYERS = {
    "camera": ("camera",),
    "format": ("format",),
    "codec": ("codec",),
    # the connectors read only their own layer: they sit between the source and the
    # recorder, so a tape or player change must not dirty them
    "connector": ("connector",),
    "tape": ("tape",),
    "player": ("player", "format"),
    "display": ("display",),
    # the generator reads only its own layer, so editing the text does not dirty
    # the camera or the tape
    "overlay": ("overlay",),
}


def _hash(obj) -> str:
    return hashlib.blake2s(
        json.dumps(obj, sort_keys=True, default=str).encode(), digest_size=10
    ).hexdigest()


@dataclass
class DirtyInfo:
    first_dirty_stage: str | None = None
    first_dirty_frame: int = 0
    cached_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "first_dirty_stage": self.first_dirty_stage,
            "first_dirty_frame": self.first_dirty_frame,
            "cached_frames": self.cached_frames,
        }


class Renderer:
    """
    `source` must provide frames by index: `source.frame(index) -> np.ndarray`
    (H, W, 3) float32 in 0..1 (:class:`atsim.io.source.ClipSource`).
    """

    def __init__(self, source, width: int, height: int, fps: float,
                 params: dict[str, dict[str, float]] | None = None,
                 seed: int = 20240101, std_key: str = "ntsc",
                 preview: bool = False, scale: float = 1.0,
                 max_pump: int = 1200, total_frames: int | None = None):
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.seed = int(seed)
        self.std = get_standard(std_key)
        self.preview = bool(preview)
        self.scale = float(scale)
        self.max_pump = int(max_pump)
        # how many frames the clip actually has; used to bound the tape plan.
        # An unbounded value here is what previously caused a 19 GB allocation.
        self.total_frames = int(total_frames or getattr(source, "frame_count", 0) or 0)
        self.schema = SCHEMA
        self.params = self.schema.resolve_all(params or {})

        self.chain: Chain | None = None
        self.stage: dict[str, Stage] = {}
        # per-frame parameters: keyframe tracks and the machine's condition.  These
        # belong to the renderer, not to the chain -- the chain is rebuilt whenever the
        # parameters change, and anything initialised there is cleared with it.
        self._keyframes = None
        self._environment = None
        self._range_cache = None
        self._layer_cache = None
        self._layer_hashes: dict[str, str] = self._compute_hashes()
        self.dirty_from = STAGE_ORDER[0]
        self.dirty_frame = 0
        self._cache: dict[str, dict[int, np.ndarray]] = {s: {} for s in STAGE_ORDER}
        self._cache_buf: dict[str, dict[int, tuple]] = {s: {} for s in STAGE_ORDER}
        self._prev_src: np.ndarray | None = None
        self._pumps = 0
        self.last_render_ms = 0.0
        self.stage_ms: dict[str, float] = {}

    # ------------------------------------------------------------------
    # hashing / invalidation
    # ------------------------------------------------------------------
    def _compute_hashes(self) -> dict[str, str]:
        h: dict[str, str] = {}
        acc: dict = {"seed": self.seed, "std": self.std.name, "w": self.width,
                     "h": self.height, "fps": round(self.fps, 6)}
        if self._environment is not None:
            acc["env"] = (round(float(self._environment.warmth), 5),
                          round(float(self._environment.age), 5),
                          round(float(self._environment.humidity), 5),
                          round(float(self._environment.cold_sensitivity), 5),
                          round(float(self._environment.age_sensitivity), 5))
        kf_tracks = getattr(self._keyframes, "tracks", None) or []
        kf_by_layer: dict[str, list] = {}
        for tr in kf_tracks:
            kf_by_layer.setdefault(tr.layer, []).append(tr.to_dict())
        if "environment" in kf_by_layer:
            acc["kf_env"] = kf_by_layer["environment"]

        ov_cfg = self.params.get("overlay", {})
        osd_at = int(round(float(ov_cfg.get("at", 0.0)) or 0.0))

        for stage_name in STAGE_ORDER:
            for layer in STAGE_LAYERS[stage_name]:
                acc[f"{stage_name}/{layer}"] = self.params.get(layer, {})
                if layer in kf_by_layer:
                    acc[f"{stage_name}/kf/{layer}"] = kf_by_layer[layer]
            # When `overlay.at == 1` (before tape), `overlay` runs immediately before `tape`,
            # so any change to `overlay` (or switching `at` between 0 and 1) must dirty `tape`
            # and everything downstream of it!
            if stage_name == "tape":
                acc["tape/osd_at"] = osd_at
                if osd_at == 1:
                    acc["tape/overlay_pre"] = ov_cfg
                    if "overlay" in kf_by_layer:
                        acc["tape/kf/overlay"] = kf_by_layer["overlay"]
            h[stage_name] = _hash(acc)
        return h

    def invalidate(self, params: dict[str, dict[str, float]] | None = None) -> DirtyInfo:
        """
        Adopt new parameters and drop the caches that depend on them.

        The stage objects have to be **rebuilt**, not merely re-hashed.  Every stage
        copies its layer out of the render context in `__init__`
        (`self.cfg = {layer: dict(ctx.layer(layer)) ...}`), so a stage keeps
        whatever the parameters were when it was constructed.
        """
        if params is not None:
            self.params = self.schema.resolve_all(params)
            self._build_chain()
        new = self._compute_hashes()
        first: str | None = None
        for stage_name in STAGE_ORDER:
            if new[stage_name] != self._layer_hashes.get(stage_name):
                first = stage_name
                break
        self._layer_hashes = new
        if first is None:
            return DirtyInfo(None, 0, self._cached_count())

        idx = STAGE_ORDER.index(first)
        # drop the dirty stage and everything downstream
        for stage_name in STAGE_ORDER[idx:]:
            self._cache[stage_name] = {}
            self._cache_buf[stage_name] = {}
        self.dirty_from = first
        frames = self._cache[STAGE_ORDER[idx - 1]] if idx > 0 else {}
        self.dirty_frame = min(frames) if frames else 0
        return DirtyInfo(first, self.dirty_frame, self._cached_count())

    def _cached_count(self) -> int:
        return len(self._cache["overlay"])

    def cached_prefix_length(self) -> int:
        """Length of the contiguous cached run starting at frame 0."""
        n = 0
        cache = self._cache["overlay"]
        while n in cache:
            n += 1
        return n

    # ------------------------------------------------------------------
    # chain construction
    # ------------------------------------------------------------------
    def _build_chain(self) -> None:
        ctx = RenderContext(
            std=self.std, params=self.params, seed=self.seed,
            frame_count=self.total_frames, fps=self.fps, width=self.width,
            height=self.height, preview=self.preview, scale=self.scale,
            source=self.source,
        )
        chain = Chain(ctx)
        chain.add(CameraStage(ctx)).add(FormatStage(ctx)).add(CodecStage(ctx))
        # Connectors sit between the source and the recorder, so damage here is
        # recorded onto the tape rather than applied on playback.
        chain.add(ConnectorStage(ctx))
        # The character generator's position is a property of the apparatus, not of the caller: a camera
        # burns its OSD over the formed picture, a titler records onto the tape.  One instance is created,
        # in the slot `overlay.at` selects, and it does nothing when the parameter disagrees with its slot.
        _osd_at = int(round(float(self.params.get("overlay", {}).get("at", 0.0)) or 0.0))
        if _osd_at == 1:
            chain.add(OverlayStage(ctx, slot=OverlayStage.SLOT_PRE_TAPE))
        chain.add(TapeStage(ctx)).add(PlayerStage(ctx)).add(DisplayStage(ctx))
        if _osd_at != 1:
            chain.add(OverlayStage(ctx, slot=OverlayStage.SLOT_TAIL))
        chain.prepare()
        self.chain = chain
        self.stage = {s.name: s for s in chain.stages}
        self._prev_src = None

    def plan(self) -> dict:
        return {
            "stages": [
                {"name": s, "label": STAGE_LABELS[s],
                 "cached": len(self._cache.get(s, {})),
                 "dirty": STAGE_ORDER.index(s) >= STAGE_ORDER.index(self.dirty_from),
                 "hash": self._layer_hashes.get(s, "")[:8]}
                for s in STAGE_ORDER
            ],
            "cached_frames": self._cached_count(),
            "dirty_from": self.dirty_from,
            "dirty_frame": self.dirty_frame,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "last_render_ms": round(self.last_render_ms, 2),
        }

    def timeline(self) -> list[dict]:
        if self.chain is None:
            self._build_chain()
        assert self.chain is not None
        return self.chain.defect_timeline()

    def reset(self) -> None:
        self._cache = {s: {} for s in STAGE_ORDER}
        self._cache_buf = {s: {} for s in STAGE_ORDER}
        self.dirty_from = STAGE_ORDER[0]
        self.dirty_frame = 0
        self._build_chain()

    # ------------------------------------------------------------------
    # frame rendering
    # ------------------------------------------------------------------
    def _fit_source(self, frame: np.ndarray) -> np.ndarray:
        """
        Force the incoming frame to the size this renderer actually renders.
        """
        h, w = frame.shape[:2]
        if (w, h) == (int(self.width), int(self.height)):
            return frame
        arr = np.asarray(frame, dtype=np.float32)
        ys = np.clip((np.arange(int(self.height)) + 0.5) * h / max(1, int(self.height)),
                     0, h - 1).astype(np.int32)
        xs = np.clip((np.arange(int(self.width)) + 0.5) * w / max(1, int(self.width)),
                     0, w - 1).astype(np.int32)
        out = arr[ys][:, xs]
        if out.ndim == 2 and frame.ndim == 3:
            out = np.repeat(out[:, :, None], frame.shape[2], axis=2)
        return out

    def set_time_varying(self, keyframes=None, environment=None) -> None:
        """
        Parameters that change during the clip: keyframe tracks, and the machine's condition.
        """
        self._keyframes = keyframes
        self._environment = environment
        self.invalidate()

    def _apply_frame_params(self, index: int) -> None:
        """
        Push this frame's parameters to the stages.
        """
        if self._keyframes is None and self._environment is None:
            return
        params = {layer: dict(vals) for layer, vals in self.params.items()}
        t = index / max(1e-6, self.fps)
        if self._environment is not None:
            from ..core.environment import apply_environment
            params, _rec = apply_environment(params, self._environment, self._ranges(),
                                             self._layer_of)
        if self._keyframes is not None:
            params, _rec = self._keyframes.apply(params, t, self._ranges())
            if "environment" in params and self._environment is not None:
                from ..core.environment import Environment, apply_environment
                env_d = params["environment"]
                env_dyn = Environment(
                    warmth=float(env_d.get("warmth", self._environment.warmth)),
                    age=float(env_d.get("age", self._environment.age)),
                    humidity=float(env_d.get("humidity", self._environment.humidity)),
                    cold_sensitivity=float(env_d.get("cold_sensitivity", self._environment.cold_sensitivity)),
                    age_sensitivity=float(env_d.get("age_sensitivity", self._environment.age_sensitivity)),
                )
                base_p = {layer: dict(vals) for layer, vals in self.params.items()}
                params, _rec = apply_environment(base_p, env_dyn, self._ranges(), self._layer_of)
                params, _rec = self._keyframes.apply(params, t, self._ranges())
        ctx = self.chain.ctx
        ctx.params = params
        for stage in self.chain.stages:
            stage.refresh_params()

    def _ranges(self) -> dict[str, tuple[float, float]]:
        if getattr(self, "_range_cache", None) is None:
            cache: dict[str, tuple[float, float]] = {}
            for layer in SCHEMA.layer_names():
                for key, spec in SCHEMA.layer(layer).specs.items():
                    cache[key] = (float(spec.minimum), float(spec.maximum))
            self._range_cache = cache
        return self._range_cache

    def _layer_of(self, key: str) -> str | None:
        if getattr(self, "_layer_cache", None) is None:
            cache: dict[str, str] = {}
            for layer in SCHEMA.layer_names():
                for k in SCHEMA.layer(layer).keys():
                    cache.setdefault(k, layer)
            self._layer_cache = cache
        return self._layer_cache.get(key)

    def _render_index(self, index: int, src_frame: np.ndarray | None) -> np.ndarray:
        if self.chain is None:
            self._build_chain()
        assert self.chain is not None

        self._apply_frame_params(index)

        chain_names = [st.name for st in self.chain.stages]
        # Map `dirty_from` (which is a name in STAGE_ORDER) to the actual index in `self.chain.stages`
        dirty_order_idx = STAGE_ORDER.index(self.dirty_from)
        dirty_set = set(STAGE_ORDER[dirty_order_idx:])
        capture_from = 0
        for idx_c, nm in enumerate(chain_names):
            if nm in dirty_set:
                capture_from = idx_c
                break

        # Find the deepest cached upstream stage in `chain_names` before `capture_from`
        start = 0
        buf: FrameBuffer | None = None
        for i in range(capture_from, 0, -1):
            up_name = chain_names[i - 1]
            cached = self._cache[up_name].get(index)
            if cached is not None:
                aux = self._cache_buf.get(up_name, {}).get(index)
                if aux is not None:
                    comp_c, meta_c = aux
                    comp_copy = comp_c.copy() if comp_c is not None else None
                    meta_copy = copy.deepcopy(meta_c)
                else:
                    comp_copy = None
                    meta_copy = {"field": index % 2}
                buf = FrameBuffer(np.asarray(cached, dtype=np.float32).copy(), comp_copy, index,
                                  index / max(1e-6, self.fps), self.std,
                                  meta_copy)
                start = i
                break

        if start == 0:
            if src_frame is None:
                src_frame = self.source.frame(index)
            src_frame = self._fit_source(src_frame)
            cam: CameraStage = self.stage["camera"]  # type: ignore[assignment]
            if self._prev_src is None and index > 0:
                try:
                    self._prev_src = self.source.frame(index - 1)
                except Exception:
                    self._prev_src = None
            if self._prev_src is not None:
                self._prev_src = self._fit_source(self._prev_src)
            cam.set_previous(self._prev_src)
            buf = FrameBuffer(np.asarray(src_frame, dtype=np.float32), None, index,
                              index / max(1e-6, self.fps), self.std,
                              {"field": index % 2})

        t0 = time.perf_counter()
        pending: list[tuple[str, np.ndarray, np.ndarray | None, dict]] = []
        for pos in range(start, len(self.chain.stages)):
            st = self.chain.stages[pos]
            t = time.perf_counter()
            buf = st.apply(buf)  # type: ignore[arg-type]
            dt = (time.perf_counter() - t) * 1000.0
            self.stage_ms[st.name] = self.stage_ms.get(st.name, 0.0) * 0.7 + dt * 0.3
            if pos >= capture_from:
                pending.append((st.name, buf.rgb,
                                buf.composite.copy() if buf.composite is not None else None,
                                copy.deepcopy(buf.meta)))
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        for name, rgb, comp_c, meta_c in pending:
            self._cache[name][index] = rgb
            self._cache_buf[name][index] = (comp_c, meta_c)
        # Always keep the final output under both `"overlay"` and `"display"` if `overlay` ran earlier
        self._cache["overlay"][index] = buf.rgb  # type: ignore[union-attr]
        self._cache["display"][index] = buf.rgb  # type: ignore[union-attr]
        if src_frame is not None:
            self._prev_src = np.asarray(src_frame, dtype=np.float32)
        return buf.rgb  # type: ignore[union-attr]

    def render_frames(self, indices: list[int], limit: int | None = None,
                      should_stop: Callable[[], bool] | None = None
                      ) -> Iterator[tuple[int, np.ndarray]]:
        wanted = sorted({int(i) for i in indices})
        if not wanted:
            return
        if self.chain is None:
            self._build_chain()

        if all(i in self._cache["overlay"] for i in wanted):
            for i in wanted:
                yield i, self._cache["overlay"][i]
            return

        top = max(wanted)
        pumped = 0
        for index in range(0, top + 1):
            if should_stop is not None and should_stop():
                return
            if index in self._cache["overlay"]:
                if index in wanted:
                    yield index, self._cache["overlay"][index]
                continue
            if limit is not None and pumped >= limit:
                return
            src = None
            if STAGE_ORDER.index(self.dirty_from) == 0 \
                    and index not in self._cache["camera"]:
                src = self.source.frame(index)
            try:
                rgb = self._render_index(index, src)
            except Exception:
                for stage_name in STAGE_ORDER:
                    self._cache[stage_name].pop(index, None)
                    self._cache_buf[stage_name].pop(index, None)
                raise
            pumped += 1
            self._pumps += 1
            if index in wanted:
                yield index, rgb

    def render_frame(self, index: int) -> np.ndarray:
        for _, rgb in self.render_frames([index]):
            return rgb
        raise RuntimeError(f"could not render frame {index}")

    # ------------------------------------------------------------------
    def still(self, index: int) -> np.ndarray:
        """Render one frame without disturbing the cache (used by A/B splits)."""
        saved = {s: dict(self._cache[s]) for s in STAGE_ORDER}
        saved_buf = {s: dict(self._cache_buf[s]) for s in STAGE_ORDER}
        try:
            return self.render_frame(index)
        finally:
            self._cache = saved
            self._cache_buf = saved_buf
            self._build_chain()
