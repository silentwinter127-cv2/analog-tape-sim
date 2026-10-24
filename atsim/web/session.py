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

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..core.params import merge_values
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
    "film": {"enabled": 0.0, "stock": 0.0},
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
    """
    base: dict = {name: {} for name in SCHEMA.layer_names()}
    for name in SCHEMA.layer_names():
        layer = SCHEMA.layer(name)
        for key in layer.keys():
            _spec = layer.specs[key]
            _d = _spec.default
            base[name][key] = _d if getattr(_spec, "kind", "number") == "text" else float(_d)
    return merge_values(base, _PRESET_BASELINE_LAYERS)


def _resolve_night_vision(params: dict, raw_params: dict | None = None) -> dict:
    """
    Apply the night-vision mode's own defaults to a resolved parameter set, WITHOUT
    overwriting any parameter that the user or active preset explicitly set in `raw_params`.
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
    from ..codecs.nightvision import MODE_PRESETS, mode_of, NightVisionSpec
    spec = NightVisionSpec.from_params(cam)
    preset = MODE_PRESETS.get(mode_of(spec))
    if not preset:
        return params

    raw_cam = (raw_params or {}).get("camera") or {}
    updates = {}
    unknown = []
    for k, v in preset.items():
        nv_k = f"nv_{k}"
        if nv_k in cam:
            # Only apply mode default if the user has not explicitly set this key
            if nv_k not in raw_cam:
                updates[nv_k] = v
        elif k in cam:
            if k not in raw_cam:
                updates[k] = v
        else:
            unknown.append(k)
    if unknown:
        import warnings
        warnings.warn("the night-vision preset names parameters the schema lacks: %s"
                      % ", ".join(f"nv_{k}" for k in unknown), stacklevel=2)
    out = dict(params)
    out["camera"] = dict(cam, **updates)
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
    unit_identities: dict = field(default_factory=dict)
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
        """
        params = SCHEMA.resolve_all(self.params or {})
        params = _resolve_night_vision(params, raw_params=self.params)
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

    # Compatibility alias used by server._run_render
    environment_state = environment_from

    def keyframe_set(self):
        """This project's timeline, as a `KeyframeSet`, or None if there are no tracks."""
        from ..core.keyframes import KeyframeSet, Track
        tracks = self.keyframes or []
        if not tracks:
            return None
        return KeyframeSet(tracks=[Track.from_dict(t) for t in tracks])

    # Compatibility alias used by server._run_render
    keyframe_tracks = keyframe_set

    def unit_report(self) -> list[dict]:
        unit = self._unit()
        if unit is None:
            return []
        from ..core.units import apply_unit, bind_ranges, spreads_from_schema
        bind_ranges(SCHEMA)
        spreads = spreads_from_schema(SCHEMA)
        base = SCHEMA.resolve_all(self.params or {})
        _p, record = apply_unit(base, unit, spreads.get)
        return record

    def _unit(self):
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
        scale = float(key[1]) if len(key) > 1 else 1.0
        if project.source:
            path = self.clip_path(project.source)
            if path.exists():
                from ..io.imagecodec import is_image_path
                if is_image_path(path):
                    src = StillSource(path, scale=scale, fps=project.fps)
                else:
                    src = ClipSource(path, scale=scale)
            else:
                src = _scaled_synthetic(720, 480, 150, project.fps, scale)
        else:
            src = _scaled_synthetic(720, 480, 150, project.fps, scale, variant="chart")
        self._sources[key] = src
        return src

    def renderer_for(self, project: Project, preview: bool = True,
                     scale: float = 0.5) -> tuple[Renderer, Any]:
        key = (project.id, bool(preview), round(float(scale), 3))
        src = self.source_for((project.id, round(float(scale), 3)), project)
        r = self._renderers.get(key)
        params = project.resolved_params()
        w = int(getattr(src, "width", 720))
        h = int(getattr(src, "height", 480))
        from ..core.standards import get_standard
        want_std = get_standard(project.standard)
        if r is None or r.seed != int(project.seed) or r.std.name != want_std.name or r.width != w or r.height != h:
            r = Renderer(src, w, h, project.fps, params, seed=project.seed,
                         std_key=project.standard, preview=preview, scale=scale)
            self._renderers[key] = r
        else:
            r.invalidate(params)
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
        if isinstance(graph_dict, dict):
            entry["graph"] = graph_dict
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
                return group[key].get("layers") or group[key].get("values") or {}
        return {}

    def _preset_spec(self, key: str) -> tuple[dict | None, bool]:
        if key in self.custom_presets:
            return self.custom_presets[key], True
        if key in FACTORY_PRESETS:
            return FACTORY_PRESETS[key], True
        if key in CAMCORDER_PRESETS:
            return CAMCORDER_PRESETS[key], True
        if key in PROFILE_LIBRARY.get("digital", {}):
            return PROFILE_LIBRARY["digital"][key], True
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
        # Support profile entries whose `values` is a flat dict for one layer (e.g. CAMERA_PROFILES["docu_1989_telecine"])
        first_val = next(iter(layers.values()), None)
        if first_val is not None and not isinstance(first_val, dict):
            for grp_name, grp_items in PROFILE_LIBRARY.items():
                if key in grp_items:
                    target_layer = {"media": "tape", "player": "player", "display": "display", "camera": "camera"}.get(grp_name, grp_name)
                    layers = {target_layer: dict(layers)}
                    break
        if is_full_preset:
            base = {lay: dict(vals) for lay, vals in _PRESET_BASELINE_LAYERS.items()}
            project.params = merge_values(base, layers)
            project.preset = key
            if spec.get("standard"):
                project.standard = str(spec["standard"])
        else:
            project.params = merge_values(project.params or {}, layers)
            project.preset = key
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
