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
            base[name][key] = float(layer.specs[key].default)
    return merge_values(base, _PRESET_BASELINE_LAYERS)


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
        return SCHEMA.resolve_all(self.params or {})


class SessionStore:
    """Projects, sources and renderers, with the preview caches kept warm."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
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
        return r, src

    def drop(self, project_id: str) -> None:
        for key in [k for k in self._renderers if k[0] == project_id]:
            self._renderers.pop(key, None)
        for key in [k for k in self._sources if k[0] == project_id]:
            self._sources.pop(key, None)

    # ------------------------------------------------------------------
    def preset_layers(self, key: str) -> dict:
        if key in FACTORY_PRESETS:
            return FACTORY_PRESETS[key]["layers"]
        for group in PROFILE_LIBRARY.values():
            if key in group:
                return group[key]["layers"]
        return {}

    def apply_preset(self, project: Project, key: str) -> Project:
        """
        Apply a preset as a complete state, not as an adjustment to the screen.

        Expanding over `_preset_baseline()` rather than over `project.params` is the
        whole point: a partial preset (one that defines only the parameters it is
        about) then produces the same result every time, instead of inheriting the
        previous selection's unmentioned layers.  Applying "Hi8 shuttle" after "worn
        rental tape" used to give the rental tape's camera, format, player and
        display settings plus Hi8 shuttle residue -- and clicking the same preset
        twice in a different order gave a different picture.
        """
        layers = self.preset_layers(key)
        if layers:
            project.params = merge_values(_preset_baseline(), layers)
            project.preset = key
        return project

    def catalog(self) -> dict:
        return {
            "factory": {k: {"label": v["label"]} for k, v in FACTORY_PRESETS.items()},
            "profiles": {
                group: {k: {"label": v["label"], "note": v.get("note", "")}
                        for k, v in items.items()}
                for group, items in PROFILE_LIBRARY.items()
            },
            "layers": {
                "camera": "01 成像系统（摄像机）",
                "format": "02 信号编码（格式）",
                "codec": "02b DV 编解码",
                # These labels duplicate LAYER_LABELS in core/profiles.py, which is how
                # the connector layer came to show as untranslated "connector" in the
                # interface: the UI reads this copy, and adding the layer to the other
                # one was not enough.  Kept in step by the check that every schema layer
                # has a label here.
                "connector": "02c 链路接口（受潮/氧化）",
                "rf": "02r RF 射频链路",
                "tape": "03 记录介质（磁带）",
                "player": "04 读取解码（播放设备）",
                "display": "05 显示设备（CRT）",
            },
        }
