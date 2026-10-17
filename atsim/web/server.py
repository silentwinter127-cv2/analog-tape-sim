"""
The WebUI server.

Deliberately built on the standard library's HTTP server: this project's value is
in the signal model, and a framework dependency would buy nothing here while
adding install friction.  Long work (renders) runs on worker threads and reports
progress through a job table the page polls, so the HTTP loop never blocks.

Endpoints
    GET  /                    the app
    GET  /api/catalog         presets, device profiles, layer labels
    GET  /api/schema          the full parameter schema with units/provenance
    GET  /api/project         current project
    GET  /api/export          the whole configuration as a downloadable JSON file
    POST /api/import          apply an exported configuration (resets first, so the file is complete)
    POST /api/project         update project (params / seed / preset / source)
    POST /api/preset          apply a factory preset or a device profile
    POST /api/upload          import a clip
    GET  /api/frame           one rendered still (PNG or JPEG)  ?t=&scale=&mode=
    GET  /api/timeline        the generated defect timeline
    GET  /api/render          start a full-resolution render job
    GET  /api/job             job status / progress
    GET  /api/download        the finished render
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..core.params import merge_values
from ..core.profiles import SCHEMA
from ..io.ffmpeg import VideoWriter, have_ffmpeg, jpeg_bytes
from ..render.renderer import Renderer, STAGE_ORDER
from .session import Project, SessionStore

WEB_DIR = Path(__file__).parent / "static"


def _png(rgb: np.ndarray) -> bytes:
    """
    Encode a frame as PNG.

    Uses the built-in codec rather than Pillow.  Pillow was the only third-party
    dependency in the whole path, and on a machine that lacks it the program
    started, accepted a clip, and then failed at the first preview frame -- the
    worst place for a missing library to surface.
    """
    from ..io.imagecodec import png_encode
    return png_encode(rgb)


def _jpeg(rgb: np.ndarray, quality: int = 88) -> bytes:
    """
    Encode a frame as JPEG through FFmpeg.

    Worth knowing the cost: this spawns one FFmpeg per call, measured at ~50 ms
    for a 360x240 frame, against ~6 ms for the built-in PNG encoder.  So JPEG is
    the right choice for a *file the user keeps* and the wrong choice for a
    preview frame crossing localhost -- see `_auto` below, which picks per use.
    """
    from ..io.imagecodec import jpeg_encode
    return jpeg_encode(rgb, quality=quality)


def _auto(rgb: np.ndarray, prefer_jpeg: bool, quality: int = 88) -> tuple[bytes, str]:
    """
    Pick an encoder and report which was used.

    PNG is the default for interactive frames: the bytes travel over loopback, so
    the larger file costs nothing measurable, while avoiding a per-frame process
    spawn removes roughly half the preview's latency.  JPEG is kept for exports,
    where the file is kept and size matters more than milliseconds.
    """
    if prefer_jpeg:
        return _jpeg(rgb, quality), "image/jpeg"
    return _png(rgb), "image/png"


def _neutral_params(params: dict) -> dict:
    """Every device neutralised, for the canvas A/B reference."""
    from ..core.params import merge_values
    return merge_values(params, {
        "camera": {"sensor_noise": 0.0, "fixed_pattern": 0.0, "chroma_aberration": 0.0,
                   "motion_blur": 0.0, "sharpness": 1.0, "exposure": 0.0,
                   "white_balance_temp": 0.0, "ccd_smear": 0.0, "halation": 0.0,
                   "ccd_gain": 0.0, "color_cast": 0.0, "highlight_tint": 0.0,
                   "lens_wear": 0.0, "lens_flare": 0.0, "sensor_type": 0.0},
        "format": {"edge_boost": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                   "color_noise": 0.0, "burst_phase_error": 0.0,
                   "luma_bandwidth": 5.0, "chroma_bandwidth": 1.5,
                   "chroma_vertical_averaging": 0.0},
        "codec": {"dv_enabled": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0,
                 "micro_jitter": 0.0, "vertical_instability": 0.0, "head_switching": 0.0,
                 "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0,
                 "noise_floor": 0.0, "head_clog": 0.0, "debris": 0.0,
                 "surface_damage": 0.0, "chroma_loss": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0,
                   "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0,
                   "line_delay": 0.0, "output_gain": 0.0, "output_bias": 0.0,
                   "cross_color": 0.0, "cross_luma": 0.0},
        "display": {"enabled": 0.0},
    })


def _neutral_checked(neutral: dict) -> dict:
    """
    Fail loudly if the A/B reference names a parameter that does not exist.

    A hand-written neutral set is exactly the kind of code that rots: parameters
    get added, the dict is not updated, and because unknown keys are ignored the
    comparison silently stops being a comparison.  That had already happened here
    -- five keys named parameters that do not exist, and several newer ones were
    missing entirely.  Checking at construction turns it into an immediate error.
    """
    from ..core.profiles import SCHEMA
    known = {lay: set(SCHEMA.layer(lay).keys()) for lay in SCHEMA.layer_names()}
    for layer, values in neutral.items():
        if layer not in known:
            raise KeyError(f"neutral reference names unknown layer {layer!r}")
        unknown = sorted(k for k in values if k not in known[layer])
        if unknown:
            raise KeyError(
                f"neutral reference names unknown parameter(s) in {layer!r}: {unknown}")
    return neutral


# --- node position units ---------------------------------------------------
# The canvas stores node positions in *pixels*.  Presets are authored in grid columns because
# that is readable in source, so the conversion happens once, here, as a preset is applied.
#
# There used to be three units in circulation at the same time -- preset columns, dragged
# pixels, and a newly added node's pixels sent without the flag that marks them as such -- and
# every resulting bug was an attempt to disambiguate them on the client.  That cannot work
# while the units differ, so the ambiguity is removed instead of being guessed at.


class App:
    """Holds all mutable state; one instance per server."""

    def __init__(self, data_dir: Path, default_clip: str | None = None):
        self.store = SessionStore(data_dir)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self._clean: dict[tuple, Renderer] = {}
        # what the character generator last drew.  Declared here so the payload can read it before any
        # frame has been rendered, rather than raising on a missing attribute.
        self.osd_report: dict = {}
        self.current = Project(name="新工程", seed=20240101)
        if default_clip:
            p = Path(default_clip)
            if p.exists():
                from ..io.ffmpeg import probe
                self.current.source = str(p)
                self.current.source_info = probe(p).to_dict()
                self.current.fps = self.current.source_info.get("fps") or self.current.fps
        self.store.apply_preset(self.current, "family_archive_1992")
        self.store.save(self.current)
        # the apparatus the user wires up; starts as the classic VHS path, which
        # matches the default preset above
        from ..graph import GRAPH_PRESETS, Graph
        # Normalise here as well as in the preset route: the graph is built from a
        # preset, so it arrives in grid columns, and a server that hands the canvas
        # columns renders every node on top of the next one.
        self.graph: Graph = (GRAPH_PRESETS["vhs_consumer"]["build"]()
                             .normalise_positions())
        self._graph_execs: dict[tuple, object] = {}
        # renderer revision counter: lets the client tell whether a tweak needs
        # a full re-render of the preview or can reuse a cached prefix
        self._revision = 0

    # ------------------------------------------------------------------
    def project_dict(self) -> dict:
        info = self.current.source_info or {}
        return {
            "project": self.current.to_dict(),
            "params": self.current.resolved_params(),
            "source_info": info,
            "standard": self.current.standard,
            # The per-frame state, so the interface can show what is set.  `unit_report` is included
            # because the point of an individual machine is that its deviations can be named -- a
            # realism claim about one unit is only checkable if its numbers can be read out.
            "environment": dict(self.current.environment or {}),
            "keyframes": list(self.current.keyframes or []),
            "unit_identities": dict(self.current.unit_identities or {}),
            "unit_report": self.current.unit_report(),
            # what the character generator last drew, and which characters its font could not draw.
            # Empty until a frame has been rendered, which the interface handles by falling back to its
            # own coverage check.
            "osd_report": dict(getattr(self, "osd_report", {}) or {}),
        }

    def set_project(self, payload: dict) -> dict:
        with self.lock:
            p = self.current
            # A payload may name layers in two ways and both must work.  The nested form is the server's own
            # convention (`set_levels` wraps its argument that way), and the bare form is what a caller
            # reasonably sends when it wants to set a layer.  Measured before this: the bare form left
            # `overlay.enabled` at 0.0 with no error, so a caller had no way to tell it had failed.
            #
            # The bare form is detected by the payload naming any *known layer* at the top level.  A payload
            # that only carries `params`/`seed`/`standard` and so on has none, so the nested path is used.
            _layers = {l for l in self.store.schema.layer_names()} if hasattr(self.store, "schema") else set()
            if not _layers:
                from ..core.profiles import SCHEMA as _S
                _layers = set(_S.layer_names())
            _bare = [k for k in payload if k in _layers]
            if "params" in payload and isinstance(payload["params"], dict):
                merged = dict(payload["params"])
                # a caller that nested twice leaves a layer literally called `params`; unfurl it so the
                # values reach the layer they name
                if isinstance(merged.get("params"), dict):
                    inner = dict(merged.pop("params"))
                    inner.update(merged)
                    merged = inner
                p.params = merge_values(p.params or {}, merged)
            elif _bare:
                p.params = merge_values(p.params or {}, {k: payload[k] for k in _bare})
            # **`and payload["seed"] is not None`, because a key that exists holding None is not a value.**  Measured:
            # `import_config` builds its payload with `"seed": payload.get("seed")`, so **any imported configuration that
            # does not carry a `seed` key of its own reaches this line as None and raises**
            # `TypeError: int() argument must be ... not 'NoneType'` -- and the import fails entirely, with the values
            # never applied.  **The user's report "导入失败 配置里没有 params" was this**, not the missing `params`, which
            # was a second and separate fault in the file itself.
            #
            # The same guard is applied to the other optional fields below: each is set only when the payload actually
            # carries a value, so a complete-looking payload with None-filled gaps cannot overwrite a project with
            # `str(None)` or an empty name.
            if payload.get("seed") is not None:
                p.seed = int(payload["seed"])
            if payload.get("standard") is not None:
                p.standard = str(payload["standard"])
            if "environment" in payload and isinstance(payload["environment"], dict):
                p.environment = dict(payload["environment"])
            if "keyframes" in payload and isinstance(payload["keyframes"], list):
                p.keyframes = list(payload["keyframes"])
            if "unit_identities" in payload and isinstance(payload["unit_identities"], dict):
                # an empty string means the nominal model, and the key is the device class
                p.unit_identities = {str(k): str(v) for k, v in payload["unit_identities"].items()}
            if payload.get("name") is not None:
                p.name = str(payload["name"])
            if "source" in payload:
                p.source = payload["source"]
                if p.source:
                    from ..io.ffmpeg import probe
                    try:
                        info = probe(self.store.clip_path(p.source))
                        p.source_info = info.to_dict()
                        p.fps = info.fps or p.fps
                    except Exception as exc:
                        p.source_info = {"error": str(exc)}
                else:
                    p.source_info = None
            if payload.get("reset_params"):
                p.params = {}
            self.store.drop(p.id)
            self.store.save(p)
            self._revision += 1
        return self.project_dict()

    # ------------------------------------------------------------------ export / import
    def export_config(self) -> dict:
        """
        The whole configuration as one JSON-serialisable object, **as a diff against the schema defaults**.

        A full dump of every layer is 400-odd numbers and says nothing about which of them a reproduction depended
        on. The diff says exactly that, and it is small enough to read.

        Everything needed to reproduce a frame is included: the changed parameters, the seed, the standard, the
        environment, the keyframe tracks, the unit identities, and what the source was. A version and a timestamp go
        with it, so an old file can be recognised instead of being applied half-way.
        """
        from ..core.profiles import SCHEMA
        import datetime as _dt

        with self.lock:
            p = self.current
            defaults = SCHEMA.resolve_all({})
            changed: dict[str, dict[str, object]] = {}
            for layer, values in (p.params or {}).items():
                base = defaults.get(layer) or {}
                diff = {}
                for k, v in values.items():
                    d = base.get(k)
                    # a NaN default is how the schema writes "off", and a NaN is not equal to itself
                    if isinstance(d, float) and isinstance(v, float) and d != d and v != v:
                        continue
                    if isinstance(d, float) and isinstance(v, float):
                        if abs(d - v) <= 1e-9:
                            continue
                    elif d == v:
                        continue
                    diff[k] = v
                if diff:
                    changed[layer] = diff
            return {
                "kind": "atsim.config",
                "version": 1,
                "exported": _dt.datetime.now().isoformat(timespec="seconds"),
                "name": p.name,
                "standard": p.standard,
                "seed": p.seed,
                "fps": p.fps,
                "environment": dict(p.environment or {}),
                "keyframes": list(p.keyframes or []),
                "unit_identities": dict(p.unit_identities or {}),
                # recorded but not imported: the path is machine-specific, and it is here so a reader can tell
                # which clip the file was made from
                "source": p.source,
                "source_info": p.source_info,
                "params": changed,
                "param_count": sum(len(v) for v in changed.values()),
            }

    def import_config(self, payload: dict) -> dict:
        """
        Apply an exported configuration.  **The parameters are reset first, so the file means the same thing here as
        it did where it was written.**

        `set_project` already merges correctly, so the only thing an import has to add is determinism. Applying a
        file on top of whatever happens to be loaded would leave every value the file does not mention at its
        current setting, and the same file would then give different results on two machines -- which defeats the
        point of exporting one.
        """
        if not isinstance(payload, dict):
            raise ValueError("配置必须是一个 JSON 对象")
        kind = payload.get("kind")
        if kind not in (None, "atsim.config"):
            raise ValueError("这不是本程序的配置导出文件（kind=%r）" % (kind,))
        version = payload.get("version")
        if isinstance(version, int) and version > 1:
            raise ValueError("配置版本 %d 比本程序支持的版本新" % version)
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("配置里没有 params")

        # reset first, then merge -- so the file is complete on its own
        self.set_project({"reset_params": True})
        # **Only the keys the file actually carries are forwarded.**  Passing `payload.get(k)` for every optional field
        # puts `None` under each key the file omits, and `set_project` then has to defend against a value that is not
        # there -- which is how a configuration without a `seed` failed to import at all.  **A key that is absent should
        # be absent**, so the forwarding is built from what is present rather than from a fixed list.
        forward = {"params": params}
        for opt in ("seed", "standard", "environment", "keyframes", "unit_identities", "name"):
            if payload.get(opt) is not None:
                forward[opt] = payload[opt]
        applied = self.set_project(forward)
        applied["imported"] = {
            "from": payload.get("exported"),
            "version": version,
            "params_applied": sum(len(v) for v in params.values() if isinstance(v, dict)),
        }
        return applied

    def apply_preset(self, key: str) -> dict:
        with self.lock:
            self.store.apply_preset(self.current, key)
            self.store.drop(self.current.id)
            self.store.save(self.current)
            self._revision += 1
        return self.project_dict()

    def set_levels(self, layer: str, values: dict) -> dict:
        return self.set_project({"params": {layer: values}})

    # ------------------------------------------------------------------
    def renderer(self, preview: bool, scale: float):
        with self.lock:
            return self.store.renderer_for(self.current, preview=preview, scale=scale)

    def frame(self, t: float, scale: float, quality: int,
              clean: bool = False) -> bytes:
        r, src = self.renderer(True, scale)
        idx = int(round(t * self.current.fps))
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        if clean:
            # A/B reference: the same frame with every stage neutralised, so the
            # comparison shows what the chain actually did rather than comparing
            # against a differently-scaled decode.
            r = self._clean_renderer(scale)
        rgb = r.render_frame(idx)
        # the character generator fills its report while drawing, so this is the moment it is known.
        # Captured here rather than in the stage so that every path which renders a preview updates it
        # and the interface has one place to read from.
        try:
            st = (r.stage or {}).get("overlay")
            if st is not None and getattr(st, "report", None):
                self.osd_report = dict(st.report)
        except Exception:                                             # noqa: BLE001
            pass
        return _jpeg(rgb, quality) if quality < 100 else _png(rgb)

    def source_frame(self, t: float, scale: float, quality: int = 88) -> bytes:
        """
        The untouched source frame, for the A/B reference.

        This used to be produced by running the whole chain with every parameter
        neutralised, which has two problems.  It cannot actually be clean -- the
        neutral set has to name every parameter the chain can touch, and as soon
        as one is missed the "original" is silently distorted (the camera's
        highlight roll-off alone shifted it by 0.07, which the player's
        composite separation then amplified to 0.31).  And it costs a full
        luma/chroma separation and re-encode to produce a frame that is supposed
        to be unprocessed.

        The honest reference is the decoded clip itself, which is also what the
        chain is fed, so the comparison isolates the chain's contribution exactly.
        """
        r, src = self.renderer(True, scale)
        idx = int(round(t * self.current.fps))
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        rgb = np.clip(src.frame(idx), 0.0, 1.0).astype(np.float32)
        return _jpeg(rgb, quality) if quality < 100 else _png(rgb)

    def sweep(self, layer: str, key: str, values: list[float], t: float,
              scale: float = 0.35, quality: int = 82,
              second: tuple[str, str, list[float]] | None = None) -> dict:
        """
        Render the current project once per parameter value, for one-page review.

        The point is to make "does this control behave the way it is documented"
        answerable by *looking*, on the user's own material, without running a
        script or dragging sliders one at a time.  Each cell is genuinely re-rendered
        through the same chain rather than reusing the preview, so the grid shows
        the mechanism instead of one cached picture.

        A second parameter may be swept as well, giving a small matrix rather than a
        strip -- useful where two mechanisms interact, such as tape speed with
        tracking error.
        """
        import base64
        from ..core.params import merge_values
        from ..io.imagecodec import jpeg_encode

        base = self.current.params or {}
        idx = int(round(t * self.current.fps))
        combos: list[tuple[float, ...]] = [(float(v),) for v in values]
        if second is not None and second[2]:
            combos = [(float(v), float(w)) for v in values for w in second[2]]

        tiles: list[dict] = []
        for combo in combos:
            params = merge_values(base, {layer: {key: float(combo[0])}})
            if second is not None and len(combo) > 1:
                params = merge_values(params, {second[0]: {second[1]: float(combo[1])}})
            r, src = self.renderer(True, scale)
            r.invalidate(params)
            last = max(0, int(getattr(src, "frame_count", 1)) - 1)
            frame = r.render_frame(min(idx, last))
            tile = {
                "value": float(combo[0]),
                "jpeg": base64.b64encode(jpeg_encode(frame, quality=quality)).decode("ascii"),
                "mean": round(float(frame.mean()), 4),
            }
            if second is not None and len(combo) > 1:
                tile["value2"] = float(combo[1])
            tiles.append(tile)

        cols = len(second[2]) if (second is not None and second[2]) else len(values)
        return {
            "layer": layer, "key": key, "t": float(t), "cols": max(1, int(cols)),
            "second": None if second is None else {"layer": second[0], "key": second[1]},
            "tiles": tiles,
        }

    def _clean_renderer(self, scale: float) -> Renderer:
        key = (self.current.id, "clean", round(float(scale), 3))
        r = self._clean.get(key)
        src = self.store.source_for((self.current.id, round(float(scale), 3)), self.current)
        if r is None:
            # The neutral set must cover every parameter the chain can add, or the
            # "original" side is not actually clean.  Several keys had been added
            # after this dict was written and were silently leaking into the
            # reference (recording speed, pre-emphasis, ringing, filter family),
            # and five more named parameters that do not exist at all -- the
            # simulator ignores unknown keys, so nothing complained while the
            # reference quietly kept whatever the project had set.
            #
            # Built here, then verified against the schema in `_neutral_checked`
            # so a typo or a newly added parameter fails loudly instead of
            # silently degrading the comparison.
            neutral = {
                "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                           "chroma_aberration": 0.0, "motion_blur": 0.0,
                           "sharpness": 1.0, "exposure": 0.0, "white_balance_temp": 0.0,
                           "ccd_smear": 0.0, "halation": 0.0, "ccd_gain": 0.0,
                           "color_cast": 0.0, "highlight_tint": 0.0,
                           "lens_wear": 0.0, "lens_flare": 0.0},
                "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0,
                         "micro_jitter": 0.0, "vertical_instability": 0.0,
                         "head_switching": 0.0, "dropout_rate": 0.0, "wear": 0.0,
                         "crease_rate": 0.0, "noise_floor": 0.0,
                         "tape_speed": 0.0,          # SP
                         "head_clog": 0.0, "debris": 0.0, "surface_damage": 0.0,
                         "chroma_loss": 0.0, "tracking_error": 0.0,
                         "tracking_noise_rows": 0.0, "tracking_noise_wave": 0.0,
                         "tracking_noise_snow": 0.0},
                "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0,
                           "transport_jitter": 0.0, "head_wear": 0.0,
                           "frequency_response": 0.0, "line_delay": 0.0,
                           "output_gain": 0.0, "output_bias": 0.0,
                           "preemphasis": 0.0, "ringing": 0.0},
                "display": {"enabled": 0.0},
                "format": {"edge_boost": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                           "color_noise": 0.0, "chroma_vertical_averaging": 0.0,
                           "comb_depth": 0.0, "burst_phase_error": 0.0,
                           "luma_bandwidth": 5.0, "chroma_bandwidth": 1.5,
                           "filter_kind": 0.0},    # gaussian: no group delay
                "codec": {"dv_enabled": 0.0},
                "rf": {"adjacent_channel": 0.0, "sync_loss_amount": 0.0},
                "audio": {"enabled": 0.0},
            }
            neutral = _neutral_checked(neutral)
            r = Renderer(src, getattr(src, "width", 720), getattr(src, "height", 480),
                         self.current.fps, neutral, seed=self.current.seed,
                         std_key=self.current.standard, preview=True, scale=scale)
            self._clean[key] = r
        return r

    # ------------------------------------------------------------------
    # the node graph
    # ------------------------------------------------------------------
    def graph_state(self) -> dict:
        from ..graph import CATALOG, TYPES, GRAPH_PRESETS
        issues = self.graph.validate()
        return {
            "graph": self.graph.to_dict(),
            "describe": self.graph.describe(),
            "catalog": [n.to_dict() for n in CATALOG],
            "types": TYPES,
            "presets": {k: {"label": v["label"], "note": v["note"]}
                        for k, v in GRAPH_PRESETS.items()},
            "issues": [i.to_dict() for i in issues],
        }

    def graph_set(self, payload: dict) -> dict:
        from ..graph import Graph
        with self.lock:
            if "graph" in payload:
                self.graph = Graph.from_dict(payload["graph"])
            self._graph_execs.clear()
        return self.graph_state()

    def graph_load_preset(self, key: str) -> dict:
        from ..graph import GRAPH_PRESETS
        spec = GRAPH_PRESETS.get(key)
        if spec is None:
            return {"error": f"unknown graph preset {key!r}"}
        with self.lock:
            self.graph = spec["build"]()
            self._graph_execs.clear()
            # The preset declares grid columns; store pixels, so that everything downstream --
            # the canvas, a later drag, the renderer -- agrees on one unit.  The conversion
            # lives on the graph so that this route and the start-up constructor cannot drift.
            self.graph.normalise_positions()
            # a graph preset may also declare the reception conditions it was
            # designed around; without this the RF loop loaded a clean signal and
            # showed none of the artefacts it exists to demonstrate
            if spec.get("params"):
                from ..core.params import merge_values
                self.current.params = merge_values(self.current.params or {},
                                                   spec["params"])
                self.store.drop(self.current.id)
                self.store.save(self.current)
        return self.graph_state()

    def graph_add(self, payload: dict) -> dict:
        from ..graph import CATALOG_BY_KIND
        kind = str(payload.get("kind", ""))
        if kind not in CATALOG_BY_KIND:
            return {"error": f"unknown node kind {kind!r}"}
        with self.lock:
            base = kind
            n = 1
            name = base
            while name in self.graph.nodes:
                n += 1
                name = f"{base}{n}"
            self.graph.add(name, kind,
                           position=(float(payload.get("x", 0.0)),
                                     float(payload.get("y", 0.0))))
            # The caller sends `40 + random*260`, which is pixels.  `graph_move` marks its
            # position as pixels and `graph_add` did not, so a node from the palette came back
            # unflagged and could be read as a grid column -- column 300 is 67200 px to the
            # right, which is where a node would vanish to.
            self.graph.nodes[name].px_position = True
            self._graph_execs.clear()
        return self.graph_state()

    def graph_connect(self, payload: dict) -> dict:
        """Add an edge, but only if the types actually match."""
        with self.lock:
            e = None
            try:
                e = self.graph.connect(str(payload["src"]), str(payload["src_port"]),
                                       str(payload["dst"]), str(payload["dst_port"]))
                issues = [i for i in self.graph.validate() if i.severity == "error"]
                if issues:
                    # reject the edge rather than leaving a broken graph behind
                    self.graph.edges.remove(e)
                    return {"error": issues[0].message, **self.graph_state()}
            except KeyError as exc:
                if e is not None and e in self.graph.edges:
                    self.graph.edges.remove(e)
                return {"error": f"invalid connection: {exc}", **self.graph_state()}
            self._graph_execs.clear()
        return self.graph_state()

    def graph_disconnect(self, payload: dict) -> dict:
        with self.lock:
            self.graph.disconnect(str(payload["dst"]), str(payload["dst_port"]))
            self._graph_execs.clear()
        return self.graph_state()

    def graph_remove(self, payload: dict) -> dict:
        with self.lock:
            self.graph.remove(str(payload["name"]))
            self._graph_execs.clear()
        return self.graph_state()

    def graph_move(self, payload: dict) -> dict:
        """
        Record a dragged position.

        The `px` flag matters: a node placed by a preset carries *grid* units,
        while a dragged node carries pixels.  Storing the distinction is what
        stops the canvas from re-scaling a dragged coordinate on the next load,
        which previously threw the node far off screen.
        """
        with self.lock:
            node = self.graph.nodes.get(str(payload.get("name", "")))
            if node is None:
                return {"error": "no such node", **self.graph_state()}
            node.position = (float(payload.get("x", 0.0)),
                             float(payload.get("y", 0.0)))
            node.px_position = True
        return {"ok": True}

    def graph_node_params(self, payload: dict) -> dict:
        """Per-node parameter overrides, so one device can be tuned on its own."""
        with self.lock:
            node = self.graph.nodes.get(str(payload.get("name", "")))
            if node is None:
                return {"error": "no such node"}
            node.enabled = bool(payload.get("enabled", node.enabled))
            if isinstance(payload.get("params"), dict):
                for k, v in payload["params"].items():
                    if isinstance(v, (int, float)):
                        node.params[k] = float(v)
            self._graph_execs.clear()
        return self.graph_state()

    def graph_frame(self, t: float, scale: float, quality: int,
                    clean: bool = False) -> bytes:
        """Render one frame by executing the user's graph."""
        from ..graph import GraphExecutor
        from ..core.standards import get_standard
        idx = int(round(t * self.current.fps))
        with self.lock:
            key = (self.current.id, "graph", round(float(scale), 3),
                   self.current.seed, self.current.standard, clean)
            ex = self._graph_execs.get(key)
            src = self.store.source_for((self.current.id, round(float(scale), 3)),
                                        self.current)
            params = self.current.resolved_params()
            if clean:
                params = _neutral_params(params)
            if ex is None:
                ex = GraphExecutor(self.graph, get_standard(self.current.standard),
                                   params, src, int(getattr(src, "width", 720)),
                                   int(getattr(src, "height", 480)), self.current.fps,
                                   seed=self.current.seed, scale=1.0)
                self._graph_execs[key] = ex
            else:
                ex.params = {k: dict(v) for k, v in params.items()}
                for node in self.graph.nodes.values():
                    if node.params:
                        ex.params.setdefault(node.kind, {}).update(node.params)
                ex.graph = self.graph
                ex.hashes = self.graph.node_hashes()
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        run = ex.render_frame(idx)
        if run.errors:
            raise RuntimeError("; ".join(run.errors))
        return _jpeg(run.rgb, quality) if quality < 100 else _png(run.rgb)

    def graph_perf(self) -> dict:
        """Per-node timing and cache hits, for the canvas readout."""
        out = {}
        for key, ex in self._graph_execs.items():
            out[str(key)] = ex.describe()
        return out

    def timeline(self) -> dict:
        r, _ = self.renderer(True, 0.5)
        return {
            "stages": r.plan()["stages"],
            "defects": r.timeline(),
            "seed": self.current.seed,
            "fps": self.current.fps,
        }

    def _source_audio_window(self, start_s: float, duration_s: float,
                             sample_rate: int, spec) -> tuple[np.ndarray, bool]:
        """
        Fetch or synthesize a stereo float32 PCM window [start_s, start_s + duration_s).
        Returns (pcm_array, is_synthetic).
        """
        from ..codecs.audio import synthesize_test_audio
        from ..io.ffmpeg import read_audio_pcm
        p = self.current
        if p.source and int(round(spec.synth_type)) == 0:
            clip = self.store.clip_path(p.source)
            if clip.exists() and (p.source_info or {}).get("has_audio", False):
                pcm = read_audio_pcm(clip, sample_rate=sample_rate,
                                     start_s=max(0.0, float(start_s)),
                                     duration_s=max(0.1, float(duration_s)))
                if pcm is not None and len(pcm) > 0:
                    target_n = max(1, int(round(duration_s * sample_rate)))
                    if len(pcm) < target_n:
                        pad = np.zeros((target_n - len(pcm), 2), dtype=np.float32)
                        pcm = np.concatenate([pcm, pad], axis=0)
                    else:
                        pcm = pcm[:target_n]
                    return pcm, False
        kind = int(round(spec.synth_type)) if int(round(spec.synth_type)) > 0 else 1
        pcm = synthesize_test_audio(duration_s, sample_rate=sample_rate,
                                    kind=kind, seed=p.seed, start_s=start_s)
        return pcm, True

    def audio_window(self, start_s: float = 0.0, duration_s: float = 4.0,
                     mode: str = "sim", sample_rate: int = 44100):
        """
        Render an audio window [start_s, start_s + duration_s) synchronized with the
        exact video TapeStage defect plan at the same frame indices.
        """
        from ..codecs.audio import AudioSimulator, AudioSpec
        r, src = self.renderer(True, 0.5)
        fps = float(self.current.fps or 29.97)
        start_s = max(0.0, float(start_s))
        duration_s = float(np.clip(duration_s, 0.2, 60.0))
        start_frame = max(0, int(round(start_s * fps)))
        end_frame = start_frame + max(1, int(np.ceil(duration_s * fps))) + 2

        tape_stage = (r.stage or {}).get("tape")
        if tape_stage is not None and hasattr(tape_stage, "_ensure"):
            tape_stage._ensure(end_frame)
        tape_plan = getattr(tape_stage, "plan", None)

        params = self.current.resolved_params()
        spec = AudioSpec.from_params(params.get("audio", {}))
        pcm_in, is_synth = self._source_audio_window(start_s, duration_s, sample_rate, spec)
        sim = AudioSimulator(params, seed=self.current.seed, fps=fps,
                             sample_rate=sample_rate)
        if mode == "source":
            faults = sim.extract_fault_envelope(start_frame, max(1, end_frame - start_frame), tape_plan)
            from ..codecs.audio import AudioRun
            return AudioRun(
                pcm=pcm_in,
                sample_rate=sample_rate,
                start_s=start_s,
                duration_s=duration_s,
                effective_mode="source_clean",
                fault_frames=faults,
                summary={"effective_mode": "source_clean", "synthetic_source": is_synth},
            )
        run = sim.process(pcm_in, start_s=start_s, tape_plan=tape_plan)
        run.summary["synthetic_source"] = is_synth
        return run

    def audio_wav(self, start_s: float = 0.0, duration_s: float = 4.0,
                  mode: str = "sim", sample_rate: int = 44100,
                  clean: bool = False) -> bytes:
        from ..io.ffmpeg import wav_encode
        if clean or mode in ("source", "original", "clean"):
            mode = "source"
        run = self.audio_window(start_s=start_s, duration_s=duration_s,
                                mode=mode, sample_rate=sample_rate)
        return wav_encode(run.pcm, sample_rate=run.sample_rate)

    def audio_timeline(self, start_s: float = 0.0, duration_s: float = 4.0) -> dict:
        """
        Return the synchronized audio waveform envelope and per-frame A/V fault telemetry
        for the WebUI oscilloscope and fault-sync strip.
        """
        run = self.audio_window(start_s=start_s, duration_s=duration_s,
                                mode="sim", sample_rate=22050)
        bins = 180
        pcm = run.pcm
        n = len(pcm)
        step = max(1, n // bins)
        trimmed = pcm[:step * bins].reshape(bins, step, 2) if n >= bins else pcm[None, :, :]
        peak_l = np.max(np.abs(trimmed[:, :, 0]), axis=1).round(4).tolist()
        peak_r = np.max(np.abs(trimmed[:, :, 1]), axis=1).round(4).tolist()
        rms_b = np.sqrt(np.mean(trimmed ** 2, axis=(1, 2))).round(4).tolist()
        wf_list = [
            {"peak": round(float(max(pl, pr)), 4), "rms": round(float(rm), 4),
             "peak_l": round(float(pl), 4), "peak_r": round(float(pr), 4)}
            for pl, pr, rm in zip(peak_l, peak_r, rms_b)
        ]
        ff_dicts = [f.to_dict() for f in run.fault_frames]
        return {
            "start_s": round(run.start_s, 3),
            "duration_s": round(run.duration_s, 3),
            "fps": float(self.current.fps or 29.97),
            "effective_mode": run.effective_mode,
            "summary": run.summary,
            "waveform": wf_list,
            "waveform_channels": {"bins": len(peak_l), "peak_l": peak_l, "peak_r": peak_r, "rms": rms_b},
            "frames": ff_dicts,
            "fault_frames": ff_dicts,
        }

    # ------------------------------------------------------------------
    def resolve_scale(self, raw_scale: str | float, default: float = 1.0) -> float:
        s_val = str(raw_scale).strip().lower()
        if s_val in ("sd", "sd720", "720"):
            info = self.current.source_info or {}
            w = float(info.get("width") or 720.0)
            return round(float(np.clip(720.0 / max(1.0, w), 0.1, 2.0)), 4)
        try:
            return round(float(np.clip(float(s_val), 0.1, 2.0)), 4)
        except (TypeError, ValueError):
            return default

    def start_render(self, scale: float = 1.0, crf: int = 16,
                     keep_audio: bool = True, preset: str = "medium") -> str:
        job_id = uuid.uuid4().hex[:10]
        self.jobs[job_id] = {"state": "queued", "progress": 0.0, "message": "排队中",
                             "path": None, "started": time.time(),
                             "scale": scale, "crf": crf, "preset": preset}
        th = threading.Thread(target=self._run_render, args=(job_id, scale, crf, keep_audio, preset),
                              daemon=True)
        th.start()
        return job_id

    def _run_render(self, job_id: str, scale: float, crf: int, keep_audio: bool,
                    preset: str = "medium") -> None:
        job = self.jobs[job_id]
        try:
            project = Project.from_dict(self.current.to_dict())
            store = SessionStore(self.store.data_dir)
            store.projects[project.id] = project
            # The source already delivers frames at the requested scale, so the
            # renderer must be sized to *its* dimensions.  Scaling the source's
            # nominal size again leaves the stage buffers at a different shape
            # than the frames flowing through them.
            src = store.source_for((project.id, scale), project)
            width = int(getattr(src, "width", 720))
            height = int(getattr(src, "height", 480))
            r = Renderer(src, width, height, project.fps, project.resolved_params(),
                         seed=project.seed, std_key=project.standard,
                         preview=(scale < 1.0), scale=1.0)
            env, kf = project.environment_from(), project.keyframe_set()
            if kf is not None or env is not None:
                r.set_time_varying(kf, env)
            total = max(1, int(getattr(src, "frame_count", 1)))
            out_path = self.store.renders / f"{job_id}.mp4"
            audio_src = None
            sim_wav_path = None
            resolved_p = project.resolved_params()
            audio_cfg = resolved_p.get("audio", {})
            audio_enabled = float(audio_cfg.get("enabled", 1.0)) > 0.5
            if keep_audio:
                try:
                    from ..codecs.audio import AudioSimulator, AudioSpec, synthesize_test_audio
                    from ..io.ffmpeg import read_audio_pcm, wav_encode
                    spec_a = AudioSpec.from_params(audio_cfg)
                    dur_s = max(1.0 / max(1.0, project.fps), total / max(1e-6, project.fps))
                    sr = 44100
                    pcm_in = None
                    if project.source and int(round(spec_a.synth_type)) == 0:
                        clip = self.store.clip_path(project.source)
                        if clip.exists():
                            pcm_in = read_audio_pcm(clip, sample_rate=sr, start_s=0.0, duration_s=dur_s)
                    if pcm_in is None and int(round(spec_a.synth_type)) > 0:
                        pcm_in = synthesize_test_audio(dur_s, sample_rate=sr,
                                                       kind=int(round(spec_a.synth_type)),
                                                       seed=project.seed, start_s=0.0)
                    if pcm_in is not None:
                        if audio_enabled:
                            if r.chain is None:
                                r._build_chain()
                            t_stage = (r.stage or {}).get("tape")
                            if t_stage is not None and hasattr(t_stage, "_ensure"):
                                t_stage._ensure(total + 4)
                            sim_a = AudioSimulator(resolved_p, seed=project.seed,
                                                   fps=project.fps, sample_rate=sr)
                            arun = sim_a.process(pcm_in, start_s=0.0,
                                                 tape_plan=getattr(t_stage, "plan", None))
                            pcm_out = arun.pcm
                        else:
                            pcm_out = pcm_in
                        sim_wav_path = self.store.renders / f"{job_id}_audio.wav"
                        sim_wav_path.write_bytes(wav_encode(pcm_out, sample_rate=sr))
                        audio_src = str(sim_wav_path)
                    elif project.source:
                        clip = self.store.clip_path(project.source)
                        if clip.exists():
                            audio_src = str(clip)
                except Exception:
                    if project.source:
                        clip = self.store.clip_path(project.source)
                        if clip.exists():
                            audio_src = str(clip)

            job.update(state="running", message="渲染中", total=total)
            with VideoWriter(out_path, width, height, project.fps, crf=crf,
                             preset=preset) as vw:
                vw.open(audio_from=audio_src if audio_src else None)
                for i, rgb in r.render_frames(list(range(total))):
                    vw.write(rgb)
                    if i % 5 == 0 or i == total - 1:
                        job["progress"] = (i + 1) / total
                        job["message"] = f"渲染中 {i + 1}/{total}"
                        if job.get("cancel"):
                            break
            job.update(state="done", progress=1.0, message="完成", path=str(out_path))
        except Exception as exc:
            job.update(state="error", message=f"{type(exc).__name__}: {exc}",
                       trace=traceback.format_exc()[-2000:])


# ---------------------------------------------------------------------------
def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "analog-tape-sim"

        def log_message(self, fmt, *args):  # quieter console
            if "/api/frame" in (self.path or ""):
                return
            print(f"  {self.address_string()} {fmt % args}")

        # -- helpers ---------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str,
                  extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        # -- routes ----------------------------------------------------
        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = u.path
            try:
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                # The connector test bench: every interface degraded side by side against a
                # clean baseline, so the differences between them can be read directly
                # instead of inferred from one picture at a time.
                if path in ("/connectors", "/connectors.html"):
                    return self._static("connectors.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/api/catalog":
                    return self._json({"catalog": app.store.catalog()})
                if path == "/api/schema":
                    return self._json(SCHEMA.schema_dict())
                if path == "/api/project":
                    return self._json(app.project_dict())
                if path == "/api/export":
                    # a download rather than a JSON reply, so the browser saves it as a file
                    cfg = app.export_config()
                    body = json.dumps(cfg, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    return self._send(200, body, "application/json; charset=utf-8", {
                        "Content-Disposition":
                            'attachment; filename="atsim-config-%s.json"' % stamp})
                if path == "/api/timeline":
                    return self._json(app.timeline())
                if path == "/api/audio":
                    t = float((q.get("t") or ["0"])[0])
                    dur = float((q.get("dur") or q.get("duration") or ["4.0"])[0])
                    mode = str((q.get("mode") or ["sim"])[0])
                    sr = int(np.clip(int((q.get("sr") or ["44100"])[0]), 8000, 48000))
                    wav_bytes = app.audio_wav(start_s=t, duration_s=dur, mode=mode, sample_rate=sr)
                    dl = (q.get("download") or ["0"])[0] in ("1", "true", "yes")
                    headers = {}
                    if dl:
                        headers["Content-Disposition"] = f'attachment; filename="atsim_audio_{t:.2f}s.wav"'
                    return self._send(200, wav_bytes, "audio/wav", headers)
                if path == "/api/audio_timeline":
                    t = float((q.get("t") or ["0"])[0])
                    dur = float((q.get("dur") or q.get("duration") or ["4.0"])[0])
                    return self._json(app.audio_timeline(start_s=t, duration_s=dur))
                if path == "/api/frame":
                    t = float((q.get("t") or ["0"])[0])
                    scale = app.resolve_scale((q.get("scale") or ["0.5"])[0], default=0.5)
                    qual = int((q.get("q") or ["85"])[0])
                    clean = (q.get("mode") or [""])[0] == "original"
                    source_mode = (q.get("mode") or [""])[0] == "source"
                    use_graph = (q.get("engine") or ["chain"])[0] == "graph"
                    t0 = time.time()
                    if source_mode:
                        # the A/B reference: the decoded clip itself
                        data = app.source_frame(t, scale, qual)
                    else:
                        data = (app.graph_frame(t, scale, qual, clean=clean) if use_graph
                                else app.frame(t, scale, qual, clean=clean))
                    ctype = "image/png" if qual >= 100 else "image/jpeg"
                    return self._send(200, data, ctype, {
                        "X-Render-Ms": f"{(time.time() - t0) * 1000:.0f}"})
                if path == "/api/still":
                    # Export one frame as a full-resolution still.  The still
                    # counterpart of /api/render: same frame, no video container,
                    # so a single result can be filed or compared without a media
                    # player.  `mode=source` exports the untouched input frame,
                    # which is what makes before/after pairs possible.
                    t = float((q.get("t") or ["0"])[0])
                    want = (q.get("mode") or ["sim"])[0]
                    fmt = (q.get("format") or ["png"])[0].lower()
                    scale = app.resolve_scale((q.get("scale") or ["1.0"])[0], default=1.0)
                    r, src = app.renderer(True, scale)
                    idx = int(round(t * app.current.fps))
                    idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
                    if want == "source":
                        frame = np.clip(src.frame(idx), 0.0, 1.0).astype(np.float32)
                    else:
                        frame = r.render_frame(idx)
                    if fmt in ("jpg", "jpeg"):
                        data = _jpeg(frame, int((q.get("q") or ["92"])[0]))
                        ctype, ext = "image/jpeg", "jpg"
                    else:
                        data = _png(frame)
                        ctype, ext = "image/png", "png"
                    return self._send(200, data, ctype, {
                        "Content-Disposition":
                            f'attachment; filename="atsim_{want}.{ext}"'})
                if path == "/api/graph":
                    return self._json(app.graph_state())
                if path == "/api/graph/perf":
                    return self._json(app.graph_perf())
                if path == "/api/render":
                    scale = app.resolve_scale((q.get("scale") or ["1.0"])[0], default=1.0)
                    crf = int(np.clip(int((q.get("crf") or ["16"])[0]), 0, 51))
                    preset = str((q.get("preset") or ["medium"])[0]).strip().lower()
                    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
                        preset = "medium"
                    keep_audio = (q.get("audio") or ["1"])[0] not in ("0", "false", "no")
                    return self._json({"job": app.start_render(scale, crf, keep_audio=keep_audio, preset=preset)})

                if path == "/api/faults":
                    from ..diagnostics import as_dicts, coverage, STAGE_LABELS
                    return self._json({"faults": as_dicts(),
                                       "coverage": coverage(),
                                       "stage_labels": STAGE_LABELS})
                if path == "/api/job":
                    jid = (q.get("id") or [""])[0]
                    return self._json(app.jobs.get(jid, {"state": "unknown"}))
                if path == "/api/download":
                    jid = (q.get("id") or [""])[0]
                    job = app.jobs.get(jid) or {}
                    p = Path(job.get("path") or "")
                    if not p.exists():
                        return self._json({"error": "not ready"}, 404)
                    return self._send(200, p.read_bytes(), "video/mp4", {
                        "Content-Disposition": f'attachment; filename="{p.name}"'})
                if path == "/api/projects":
                    return self._json({"projects": [
                        {k: v for k, v in p.to_dict().items()
                         if k in ("id", "name", "modified", "preset", "seed")}
                        for p in app.store.projects.values()]})
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}",
                                   "trace": traceback.format_exc()[-1500:]}, 500)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            path = u.path
            try:
                if path == "/api/sweep":
                    b = self._body()
                    layer = str(b.get("layer", "tape"))
                    key = str(b.get("key", ""))
                    vals = [float(v) for v in (b.get("values") or [])]
                    if not key or not vals:
                        return self._json({"error": "需要 layer、key 与 values"})
                    if len(vals) > 24:
                        return self._json({"error": "一次最多 24 个取值"})
                    sec = None
                    if b.get("second"):
                        s2 = b["second"]
                        v2 = [float(v) for v in (s2.get("values") or [])]
                        if s2.get("key") and v2 and len(vals) * len(v2) <= 36:
                            sec = (str(s2.get("layer", layer)), str(s2["key"]), v2)
                    t0 = time.time()
                    out = app.sweep(layer, key, vals, float(b.get("t", 0.0)),
                                    scale=float(b.get("scale", 0.35)), second=sec)
                    out["ms"] = int((time.time() - t0) * 1000)
                    return self._json(out)
                if path == "/api/project":
                    return self._json(app.set_project(self._body()))
                if path == "/api/preset":
                    b = self._body()
                    return self._json(app.apply_preset(str(b.get("key", ""))))
                if path == "/api/upload":
                    n = int(self.headers.get("Content-Length") or 0)
                    raw_name = self.headers.get("X-Filename") or "clip.mp4"
                    # Accept stills as well as clips.  A photograph is a clip of
                    # length one downstream, so the only thing that has to differ
                    # is how the file is probed -- and `source_for` decides that
                    # from the extension.
                    from urllib.parse import unquote
                    from ..io.imagecodec import image_size, is_image_path
                    name = Path(unquote(raw_name)).name
                    dest = app.store.uploads / name
                    with open(dest, "wb") as fh:
                        remaining = n
                        while remaining > 0:
                            chunk = self.rfile.read(min(1 << 20, remaining))
                            if not chunk:
                                break
                            fh.write(chunk)
                            remaining -= len(chunk)
                    if is_image_path(dest):
                        size = image_size(dest)
                        if size is None:
                            return self._json({"error": f"无法读取图片：{name}"})
                        w, h = size
                        from ..io.ffmpeg import VideoInfo
                        info = VideoInfo(path=str(dest), width=w, height=h,
                                         fps=app.current.fps, frame_count=1,
                                         duration=1.0 / max(1.0, app.current.fps),
                                         has_audio=False, codec="still")
                        app.set_project({"source": str(dest)})
                        return self._json({"ok": True, "still": True,
                                           "info": info.to_dict(),
                                           "project": app.project_dict()})
                    from ..io.ffmpeg import probe
                    info = probe(dest)
                    app.set_project({"source": str(dest)})
                    return self._json({"ok": True, "info": info.to_dict(),
                                       "project": app.project_dict()})
                if path == "/api/graph":
                    return self._json(app.graph_set(self._body()))
                if path == "/api/graph/preset":
                    return self._json(app.graph_load_preset(str(self._body().get("key", ""))))
                if path == "/api/graph/add":
                    return self._json(app.graph_add(self._body()))
                if path == "/api/graph/connect":
                    return self._json(app.graph_connect(self._body()))
                if path == "/api/graph/disconnect":
                    return self._json(app.graph_disconnect(self._body()))
                if path == "/api/graph/remove":
                    return self._json(app.graph_remove(self._body()))
                if path == "/api/graph/move":
                    return self._json(app.graph_move(self._body()))
                if path == "/api/graph/node":
                    return self._json(app.graph_node_params(self._body()))
                if path == "/api/import":
                    b = self._body()
                    # accept either the parsed object or the raw text of a file
                    if isinstance(b.get("config"), dict):
                        b = b["config"]
                    elif isinstance(b.get("text"), str):
                        b = json.loads(b["text"])
                    return self._json(app.import_config(b))
                if path == "/api/save":
                    b = self._body()
                    if b.get("name"):
                        app.current.name = str(b["name"])
                    p = app.store.save(app.current)
                    return self._json({"ok": True, "project": p.to_dict()})
                if path == "/api/load":
                    b = self._body()
                    p = app.store.projects.get(str(b.get("id", "")))
                    if p is None:
                        return self._json({"error": "no such project"}, 404)
                    with app.lock:
                        app.current = p
                        app.store.drop(p.id)
                    return self._json(app.project_dict())
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}",
                                   "trace": traceback.format_exc()[-1500:]}, 500)

        # -- static ----------------------------------------------------
        def _static(self, rel: str) -> None:
            p = (WEB_DIR / rel).resolve()
            if not str(p).startswith(str(WEB_DIR.resolve())) or not p.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml",
            }.get(p.suffix, "application/octet-stream")
            self._send(200, p.read_bytes(), ctype)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8931, data_dir: str | Path = "runs",
          open_browser: bool = False, default_clip: str | None = None) -> None:
    # Report the tools at startup, with the paths actually resolved.  Discovering
    # a missing FFmpeg only when the user tries to import a clip produces a
    # failure that looks like the *file* is at fault ("系统找不到指定的文件"),
    # when in fact the tool that reads it could not be run.
    from ..io.ffmpeg import missing_tools, tool_paths
    ffmpeg_path, ffprobe_path = tool_paths()
    missing = missing_tools()
    app = App(Path(data_dir), default_clip=default_clip)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print("=" * 70)
    print("  analog-tape-sim  --  模拟磁带信号仿真工具")
    print("=" * 70)
    print(f"  WebUI:  {url}")
    print(f"  数据目录: {Path(data_dir).resolve()}")
    if app.current.source:
        print(f"  素材:   {app.current.source}")
    else:
        print("  素材:   未导入（正在使用内置测试图；可用「导入素材」按钮添加视频）")
    if missing:
        print(f"  !! 缺少工具: {', '.join(missing)} —— 导入/导出/离线渲染会失败")
        print(f"     已尝试的路径: ffmpeg={ffmpeg_path}")
        print(f"                   ffprobe={ffprobe_path}")
        print("     设置 ATSIM_FFMPEG / ATSIM_FFPROBE 指向可执行文件，或装到 C:\\ffmpeg\\bin")
    else:
        print(f"  ffmpeg: {ffmpeg_path}")
        print(f"  ffprobe:{ffprobe_path}")
    print("  按 Ctrl+C 停止")
    print("=" * 70)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="atsim-web", description="analog tape simulator WebUI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--data-dir", default="runs")
    ap.add_argument("--clip", default=None, help="video file to load at startup")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args(argv)
    serve(args.host, args.port, args.data_dir, args.open, args.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
