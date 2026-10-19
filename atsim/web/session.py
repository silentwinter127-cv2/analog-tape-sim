"""
Project + session management for the WebUI.

A *project* is the complete, reproducible description of a simulation:

    source clip + seed + standard + five parameter layers + device profiles

Anything that affects a rendered frame lives in here, which is what makes the
"same project renders the same video" guarantee possible (see the plan's
"repeatable defect timeline" requirement).  Sessions wrap a project with a
live Renderer and its caches, so the preview can reuse upstream stages.
"""
from __future__ import annotations

from ..core.profiles import CAMCORDER_PRESETS, FACTORY_PRESETS, LAYER_LABELS, PROFILE_LIBRARY, SCHEMA
from ..core.profiles import CAMCORDER_PRESETS

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..core.params import merge_values
from ..core.profiles import FACTORY_PRESETS, PROFILE_LIBRARY, SCHEMA
from ..io.ffmpeg import VideoInfo, probe
from ..io.source import ClipSource, StillSource, SyntheticSource
from ..render.renderer import Renderer


DEFAULT_PROJECT = {
    "source": None,                 # path to the input clip, or None = demo
    "standard": "ntsc",
    "seed": 20240101,
    "fps": 30000.0 / 1001.0,
    "params": {},
    "preset": None,
}


def _scaled_synthetic(width: int, height: int, frames: int, fps: float,
                      scale: float, variant: str = "chart") -> SyntheticSource:
    """A synthetic clip at the requested preview scale, even dimensions kept."""
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return SyntheticSource(width=w, height=h, frames=frames, fps=fps, variant=variant)


# The state a factory preset is expanded over.  Presets are written as deltas
# against this, so it has to be fixed rather than inherited from the screen.
_PRESET_BASELINE_LAYERS = {
    "camera": {"noise_grain": 0.0, "sensor_type": 0.0},
    "format": {"filter_kind": 1.0, "chroma_gain": 1.0,
               "chroma_vertical_averaging": 0.9},
    "codec": {"dv_enabled": 0.0, "dv_quality": 1.0, "dv_base_step": 0.0125},
    "tape": {"shuttle_speed": 0.0, "shuttle_stale": 0.0, "binder_loss": 0.0,
             "shed_grain": 0.0, "luma_pedestal": 0.0, "drum_speed": 1.0,
             "head_clog": 0.0, "debris": 0.0, "surface_damage": 0.0,
             "chroma_loss": 0.0, "tracking_noise_rows": 0.0,
             "tracking_noise_wave": 0.0, "tracking_noise_snow": 0.0},
    "player": {"blue_screen": 0.0, "blue_osd": 0.0},
}


def _preset_baseline() -> dict:
    """
    Schema defaults, plus the preset-specific baseline above.

    Two sources, because both are needed: the schema defaults guarantee every
    parameter exists and is in range, and the explicit layer values pin the handful
    of settings the presets assume without restating them.
    """
    base: dict = {name: {} for name in SCHEMA.layer_names()}
    for name in SCHEMA.layer_names():
        layer = SCHEMA.layer(name)
        for key in layer.keys():
            _spec = layer.specs[key]
            # keep the default's own type: coercing to float was a no-op for numeric parameters and
            # raises on a text one, which took down every preset application rather than just the
            # preset that mentions the overlay
            _d = _spec.default
            base[name][key] = _d if getattr(_spec, "kind", "number") == "text" else float(_d)
    return merge_values(base, _PRESET_BASELINE_LAYERS)


def _resolve_night_vision(params: dict) -> dict:
    """
    Apply the night-vision mode's own settings to a resolved parameter set, in place of nothing.

    **Why it lives here.**  The renderer and the API both call `resolved_params`, so resolving the mode here means the
    picture and the panel cannot disagree -- which they did: the camera stage applied the mode's values and the
    parameters the page read still held the schema defaults.

    **And it is a no-op when no mode is selected**, so nothing about any existing preset changes: with `nv_mode` at 0 the
    module returns the spec unchanged and the dictionary is returned as it was.
    """
    cam = params.get("camera")
    if not cam:
        return params
    try:
        mode = float(cam.get("nv_mode", 0.0))
    except (TypeError, ValueError):
        return params
    if int(round(mode)) == 0:
        return params
    from ..codecs.nightvision import MODE_PRESETS, mode_of
    from ..codecs.nightvision import NightVisionSpec
    spec = NightVisionSpec.from_params(cam)
    preset = MODE_PRESETS.get(mode_of(spec))
    if not preset:
        return params
    # **The preset's keys are `NightVisionSpec` field names; the parameter set's are `nv_`-prefixed.**  Writing them
    # unprefixed added three keys nothing reads and left the real three at their defaults -- measured, the resolver's
    # output differed from its input by exactly `accumulate`, `green_mix` and `illuminator` appearing as new keys.  The
    # guard that was meant to catch a bad name compared `f"nv_{k}"` against a set while using the bare `k` on the left,
    # so it passed everything through; it now compares like with like.
    updates = {}
    unknown = []
    for k, v in preset.items():
        if f"nv_{k}" in cam:
            updates[f"nv_{k}"] = v
        elif k in cam:
            updates[k] = v
        else:
            unknown.append(k)
    if unknown:
        import warnings
        warnings.warn("the night-vision preset names parameters the schema lacks: %s"
                      % ", ".join(f"nv_{k}" for k in unknown), stacklevel=2)
    out = dict(params)
    out["camera"] = dict(cam, **updates)
    if mode in (1.0, 2.0) or int(round(mode)) in (1, 2):
        cod = dict(out.get("codec") or {})
        if float(cod.get("dv_enabled", 0.0)) < 0.5:
            cod["dv_enabled"] = 1.0
            out["codec"] = cod
            fmt = dict(out.get("format") or {})
            fmt.update({"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0, "chroma_vertical_averaging": 0.0, "color_noise": 0.02})
            out["format"] = fmt
            tpe = dict(out.get("tape") or {})
            tpe.update({"dropout_rate": 0.0, "noise_floor": 0.02, "line_jitter": 0.02, "walk_amplitude": 0.02, "micro_jitter": 0.02, "head_switching": 0.0})
            out["tape"] = tpe
            ply = dict(out.get("player") or {})
            ply.update({"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0, "tbc_enabled": 1.0, "head_wear": 0.05, "transport_jitter": 0.05})
            out["player"] = ply
        dsp = dict(out.get("display") or {})
        if float(dsp.get("enabled", 1.0)) > 0.5 and abs(float(dsp.get("gamma", 2.35)) - 2.35) < 1e-3:
            dsp["enabled"] = 0.0
            out["display"] = dsp
    return out


@dataclass

class Project:
    id: str = ""
    name: str = "未命名工程"
    source: str | None = None
    source_info: dict | None = None
    standard: str = "ntsc"
    seed: int = 20240101
    fps: float = 30000.0 / 1001.0
    params: dict = field(default_factory=dict)
    preset: str | None = None
    # Which physical machine each device is.  An empty string means the nominal model, so a
    # project saved before this dimension existed still loads and renders identically.  The
    # identities are part of the project because a unit that is not saved is a unit that changes
    # when the project is reloaded, which would break reproducibility.
    unit_identities: dict = field(default_factory=dict)
    # The machine's condition -- warmth, ageing, damp -- and the keyframe tracks that drive it and
    # anything else.  Both vary per frame, so unlike `params` they cannot be folded into
    # `resolved_params()`; the renderer is given them instead.  A project saved before these existed
    # loads with none of them, which is what keeps it rendering identically.
    environment: dict = field(default_factory=dict)
    keyframes: list = field(default_factory=list)
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def resolved_params(self) -> dict:
        """
        The parameters this project renders with, including this unit's deviations.

        Device individual variation is applied here rather than in the renderer so that every caller
        goes through it and none can forget.  With no identities set nothing changes, which is what
        keeps every preset and capture from before this dimension rendering identically.

        **The night-vision mode's own settings are resolved here for the same reason.**  They were applied inside the
        camera stage first, so the renderer used them and nothing that *read* the parameters could see them -- measured
        in the browser, mode 3 selected with the engine's state reporting `nv_green_mix = 1` while the renderer used 0,
        i.e. two sources of truth for one picture and a slider editing a parameter the mode owned.  **This function is
        the one place both the renderer and the API pass through**, which is exactly what its own docstring says about
        device variation.
        """
        params = SCHEMA.resolve_all(self.params or {})
        params = _resolve_night_vision(params)
        unit = self._unit()
        if unit is None:
            return params
        from ..core.units import apply_unit, bind_ranges, spreads_from_schema
        bind_ranges(SCHEMA)
        spreads = spreads_from_schema(SCHEMA)
        params, _record = apply_unit(params, unit, spreads.get)
        return params

    def environment_from(self):
        """This project's machine condition, as an `Environment`, or None if it is at nominal."""
        from ..core.environment import Environment
        d = self.environment or {}
        if not d:
            return None
        return Environment(
            warmth=float(d.get("warmth", 1.0)),
            age=float(d.get("age", 0.0)),
            humidity=float(d.get("humidity", 0.0)),
            cold_sensitivity=float(d.get("cold_sensitivity", 1.0)),
            age_sensitivity=float(d.get("age_sensitivity", 1.0)))

    def keyframe_set(self):
        """This project's timeline, as a `KeyframeSet`, or None if there are no tracks."""
        from ..core.keyframes import KeyframeSet, Track
        tracks = self.keyframes or []
        if not tracks:
            return None
        return KeyframeSet(tracks=[Track.from_dict(t) for t in tracks])

    def unit_report(self) -> list[dict]:
        """
        Which parameters this machine deviates in, and by how much.

        Returned rather than logged, because the point of an individual machine is that its
        deviations can be named -- and because a realism claim about a specific unit is only
        checkable if the unit's numbers can be read out.
        """
        unit = self._unit()
        if unit is None:
            return []
        from ..core.units import apply_unit, bind_ranges, spreads_from_schema
        bind_ranges(SCHEMA)
        spreads = spreads_from_schema(SCHEMA)
        base = SCHEMA.resolve_all(self.params or {})
        # The report is computed by the same `apply_unit` call that `resolved_params` makes, from the
        # same merged unit, so the table cannot disagree with what was rendered.  An earlier version
        # built its own `DeviceUnit(identity="probe")` here -- a hardcoded string -- and reported the
        # same machine for every serial number while the render used the real one.
        _p, record = apply_unit(base, unit, spreads.get)
        return record

    def _unit(self):
        """
        This project's machine, with one identity per device class merged into one deviation set.

        Shared by `resolved_params` and `unit_report` so that what is reported is what is rendered.
        """
        ids = self.unit_identities or {}
        if not ids:
            return None
        from ..core.units import DeviceUnit, bind_ranges, spreads_from_schema
        bind_ranges(SCHEMA)
        spreads = spreads_from_schema(SCHEMA)
        merged: dict[str, float] = {}
        for _kind, identity in sorted(ids.items()):
            merged.update(DeviceUnit(identity=str(identity)).draw(spreads).deviations)
        if not merged:
            return None
        unit = DeviceUnit(identity="+".join(sorted(str(v) for v in ids.values())))
        unit.deviations = merged
        return unit


class SessionStore:
    """Projects, sources and renderers, with the preview caches kept warm."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.custom_presets_file = self.data_dir / "custom_presets.json"
        self.custom_presets: dict[str, dict] = {}
        self._load_custom_presets()
        self.uploads = self.data_dir / "uploads"
        self.projects_dir = self.data_dir / "projects"
        self.renders = self.data_dir / "renders"
        for d in (self.uploads, self.projects_dir, self.renders):
            d.mkdir(parents=True, exist_ok=True)
        self.projects: dict[str, Project] = {}
        self._renderers: dict[tuple, Renderer] = {}
        self._sources: dict[tuple, Any] = {}
        self._load_all()

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        for f in sorted(self.projects_dir.glob("*.json")):
            try:
                self.projects[f.stem] = Project.from_dict(
                    json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue

    def save(self, project: Project) -> Project:
        if not project.id:
            project.id = uuid.uuid4().hex[:12]
        project.modified = time.time()
        path = self.projects_dir / f"{project.id}.json"
        path.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        self.projects[project.id] = project
        return project

    def delete(self, pid: str) -> bool:
        p = self.projects.pop(pid, None)
        if p is None:
            return False
        (self.projects_dir / f"{pid}.json").unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    def import_clip(self, filename: str, stream) -> dict:
        safe = Path(filename).name.replace("..", "_")
        dest = self.uploads / safe
        with open(dest, "wb") as fh:
            shutil.copyfileobj(stream, fh)
        info = probe(dest)
        return {"path": str(dest), "info": info.to_dict()}

    def clip_path(self, relative_or_abs: str) -> Path:
        p = Path(relative_or_abs)
        if p.is_absolute() and p.exists():
            return p
        cand = self.uploads / p.name
        if cand.exists():
            return cand
        return p

    # ------------------------------------------------------------------
    def source_for(self, key: tuple, project: Project):
        src = self._sources.get(key)
        if src is not None:
            return src
        # The scale from the cache key must actually reach the source.  It was
        # accepted but never passed, so every "preview at 0.5" was silently
        # rendering full resolution: the badge said 360x240 while the renderer
        # was working at 720x576.  That also broke the A/B view, because the
        # simulated frame and the original ended up with different aspect ratios
        # and `object-fit: contain` then scaled them differently.
        scale = float(key[1]) if len(key) > 1 else 1.0
        if project.source:
            path = self.clip_path(project.source)
            if path.exists():
                # A still and a clip are the same thing downstream, so the choice
                # is made here once rather than in every caller.
                from ..io.imagecodec import is_image_path
                if is_image_path(path):
                    src = StillSource(path, scale=scale, fps=project.fps)
                else:
                    src = ClipSource(path, scale=scale)
            else:
                src = _scaled_synthetic(720, 480, 150, project.fps, scale)
        else:
            # no clip yet: the built-in demo scene, so the tool is always usable
            src = _scaled_synthetic(720, 480, 150, project.fps, scale, variant="chart")
        self._sources[key] = src
        return src

    def renderer_for(self, project: Project, preview: bool = True,
                     scale: float = 0.5) -> tuple[Renderer, Any]:
        """
        Get (or build) the Renderer for a project.

        Preview and final renders use *different* renderers on purpose: the
        preview works on a downscaled proxy while the final render uses full
        resolution, and sharing one cache between them would silently mix the two.
        """
        key = (project.id, bool(preview), round(float(scale), 3))
        src = self.source_for((project.id, round(float(scale), 3)), project)
        r = self._renderers.get(key)
        params = project.resolved_params()
        # size the renderer from the frames the source actually delivers: a
        # scaled ClipSource already returns smaller frames, and scaling twice
        # would put the stage buffers at a different shape than the frames.
        w = int(getattr(src, "width", 720))
        h = int(getattr(src, "height", 480))
        if r is None:
            r = Renderer(src, w, h, project.fps, params, seed=project.seed,
                         std_key=project.standard, preview=preview, scale=scale)
            self._renderers[key] = r
        else:
            r.invalidate(params)
        # The environment and the timeline are per-frame, so they are given to the renderer rather than
        # folded into `params` -- and both paths need it: a cached renderer invalidated with new
        # parameters still holds the old timeline.  Without this, editing a keyframe in the interface
        # changes nothing on screen.
        env, kf = project.environment_from(), project.keyframe_set()
        if getattr(r, "_keyframes", None) is not kf or getattr(r, "_environment", None) is not env:
            r.set_time_varying(kf, env)
        return r, src

    def drop(self, project_id: str) -> None:
        for key in [k for k in self._renderers if k[0] == project_id]:
            self._renderers.pop(key, None)
        for key in [k for k in self._sources if k[0] == project_id]:
            self._sources.pop(key, None)

    # ------------------------------------------------------------------
    def _load_custom_presets(self) -> None:
        if self.custom_presets_file.exists():
            try:
                data = json.loads(self.custom_presets_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.custom_presets = data
            except Exception:
                self.custom_presets = {}

    def _save_custom_presets_disk(self) -> None:
        self.custom_presets_file.write_text(
            json.dumps(self.custom_presets, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_custom_preset(self, project: Project, label: str, note: str = "",
                           key: str | None = None, graph_dict: dict | None = None) -> dict:
        import re as _re
        clean_label = (label or "").strip() or "用户自定义预设"
        if not key:
            slug = _re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fa5]+", "_", clean_label).strip("_")
            key = f"user_{slug}_{int(time.time()) % 100000}"
        layers_copy = {lay: dict(vals) for lay, vals in (project.params or {}).items() if isinstance(vals, dict)}
        layers_copy.setdefault("display", {})
        if "enabled" not in layers_copy["display"]:
            layers_copy["display"]["enabled"] = 0.0
        entry = {
            "label": clean_label,
            "note": (note or "").strip() or f"保存自工程「{project.name}」({project.standard.upper()})",
            "standard": project.standard,
            "layers": layers_copy,
            "environment": dict(project.environment or {}),
            "timeline": list(project.keyframes or []),
            "created": time.time(),
        }
        self.custom_presets[key] = entry
        self._save_custom_presets_disk()
        return {"key": key, **entry}

    def delete_custom_preset(self, key: str) -> bool:
        if key in self.custom_presets:
            del self.custom_presets[key]
            self._save_custom_presets_disk()
            return True
        return False

    def preset_layers(self, key: str) -> dict:
        if key in self.custom_presets:
            return self.custom_presets[key].get("layers", {})
        if key in FACTORY_PRESETS:
            return FACTORY_PRESETS[key]["layers"]
        if key in CAMCORDER_PRESETS:
            return CAMCORDER_PRESETS[key]["layers"]
        for group in PROFILE_LIBRARY.values():
            if key in group:
                return group[key]["layers"]
        return {}

    def _preset_spec(self, key: str) -> tuple[dict | None, bool]:
        if key in self.custom_presets:
            return self.custom_presets[key], True
        if key in FACTORY_PRESETS:
            return FACTORY_PRESETS[key], True
        if key in CAMCORDER_PRESETS:
            return CAMCORDER_PRESETS[key], True
        for group in PROFILE_LIBRARY.values():
            if key in group:
                return group[key], False
        return None, False

    def apply_preset(self, project: Project, key: str) -> Project:
        spec, is_full_preset = self._preset_spec(key)
        if spec is None:
            return project
        layers = spec.get("layers") or spec.get("values") or {}
        if not layers:
            return project
        if is_full_preset:
            base = {lay: dict(vals) for lay, vals in _PRESET_BASELINE_LAYERS.items()}
            project.params = merge_values(base, layers)
            project.preset = key
            if spec.get("standard"):
                project.standard = str(spec["standard"])
        else:
            project.params = merge_values(project.params or {}, layers)
        if "environment" in spec:
            project.environment = dict(spec.get("environment") or {})
        elif is_full_preset:
            project.environment = {}
        if "timeline" in spec:
            project.keyframes = list(spec.get("timeline") or [])
        elif is_full_preset:
            project.keyframes = []
        return project

    def catalog(self) -> dict:
        return {
            "factory": {k: {"label": v["label"], "layers": v["layers"]}
                        for k, v in FACTORY_PRESETS.items()},
            "camcorders": {k: {"label": v["label"], "layers": v["layers"]}
                           for k, v in CAMCORDER_PRESETS.items()},
            "profiles": {
                g: {k: {"label": v["label"], "note": v.get("note", ""),
                        "layers": v.get("layers") or v.get("values") or {}}
                    for k, v in items.items()}
                for g, items in PROFILE_LIBRARY.items()
            },
            "custom": {
                k: {
                    "label": v.get("label", k),
                    "note": v.get("note", ""),
                    "standard": v.get("standard", "ntsc"),
                    "layers": v.get("layers", {}),
                }
                for k, v in self.custom_presets.items()
            },
            "layers": LAYER_LABELS,
        }
