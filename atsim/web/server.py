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


class App:
    """Holds all mutable state; one instance per server."""

    def __init__(self, data_dir: Path, default_clip: str | None = None):
        self.store = SessionStore(data_dir)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self._clean: dict[tuple, Renderer] = {}
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
        self.graph: Graph = GRAPH_PRESETS["vhs_consumer"]["build"]()
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
        }

    def set_project(self, payload: dict) -> dict:
        with self.lock:
            p = self.current
            if "params" in payload and isinstance(payload["params"], dict):
                p.params = merge_values(p.params or {}, payload["params"])
            if "seed" in payload:
                p.seed = int(payload["seed"])
            if "standard" in payload:
                p.standard = str(payload["standard"])
            if "name" in payload:
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

    # ------------------------------------------------------------------
    def start_render(self, scale: float = 1.0, crf: int = 16,
                     keep_audio: bool = True) -> str:
        job_id = uuid.uuid4().hex[:10]
        self.jobs[job_id] = {"state": "queued", "progress": 0.0, "message": "排队中",
                             "path": None, "started": time.time()}
        th = threading.Thread(target=self._run_render, args=(job_id, scale, crf, keep_audio),
                              daemon=True)
        th.start()
        return job_id

    def _run_render(self, job_id: str, scale: float, crf: int, keep_audio: bool) -> None:
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
            total = max(1, int(getattr(src, "frame_count", 1)))
            out_path = self.store.renders / f"{job_id}.mp4"
            audio_src = None
            if keep_audio and project.source:
                clip = self.store.clip_path(project.source)
                if clip.exists():
                    audio_src = str(clip)

            job.update(state="running", message="渲染中", total=total)
            with VideoWriter(out_path, width, height, project.fps, crf=crf,
                             preset="medium") as vw:
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
                if path == "/api/timeline":
                    return self._json(app.timeline())
                if path == "/api/frame":
                    t = float((q.get("t") or ["0"])[0])
                    scale = float((q.get("scale") or ["0.5"])[0])
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
                    return self._send(200, data, "image/jpeg", {
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
                    scale = float((q.get("scale") or ["1.0"])[0])
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
                    scale = float((q.get("scale") or ["1.0"])[0])
                    crf = int((q.get("crf") or ["16"])[0])
                    return self._json({"job": app.start_render(scale, crf)})

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
