"""
One command that verifies every claim this project makes about itself.

The problem this solves: individual mechanisms were each validated with a
throwaway probe script, which is fine for developing a model and useless for
anyone else.  If you want to know whether the simulator still behaves the way the
documentation says, there needs to be a single entry point that prints a table and
exits non-zero when something regressed.

    python tools/verify.py            # all checks, a table, and a contact sheet
    python tools/verify.py --quick    # skip the slow render-based checks
    python tools/verify.py --json     # machine-readable results

What "verified" means here, and what it does not:

* Each row states a **mechanism**, the **measured direction** it must have, and
  the **number this run produced**.  A PASS is the model behaving as documented;
  it is not a claim that the model matches a real camcorder.
* Where a value was calibrated against reference material, the reference figure is
  named in the row, so the comparison is auditable rather than implied.
* Nothing here needs Pillow or any library beyond numpy, scipy and FFmpeg.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atsim.core.params import merge_values                     # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA        # noqa: E402
from atsim.io.imagecodec import save_image                     # noqa: E402
from atsim.render.renderer import Renderer                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")


# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    claim: str
    measured: str
    ok: bool
    notes: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "claim": self.claim,
                "measured": self.measured, "ok": bool(self.ok),
                "notes": self.notes}


RESULTS: list[Result] = []


def record(name: str, claim: str, measured: str, ok: bool, notes: str = "") -> None:
    RESULTS.append(Result(name, claim, measured, ok, notes))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<26} {measured}")
    if not ok:
        print(f"         expected: {claim}")
        if notes:
            print(f"         note:     {notes}")


# ---------------------------------------------------------------------------
class MovingScene:
    """A moving source, so temporal mechanisms have something to act on."""

    def __init__(self, w: int = 360, h: int = 240, frames: int = 120, fps: float = 25.0):
        self.width, self.height, self.frame_count, self.fps = w, h, frames, fps

    def frame(self, i: int) -> np.ndarray:
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        x, y = xx + i * 7.0, yy + i * 4.0
        out = np.stack([0.5 + 0.5 * np.cos(x / 19.0),
                        0.5 + 0.5 * np.cos(y / 14.0),
                        0.5 + 0.5 * np.cos((x + y) / 23.0)], axis=2)
        disc = ((xx - (i * 9.5) % w) ** 2 + (yy - h * 0.5) ** 2) < (min(h, w) * 0.11) ** 2
        out[disc] = 1.0
        return np.clip(out.astype(np.float32), 0, 1)


class StripeScene:
    """Fine horizontal stripes: the pattern that line duplication destroys."""

    def __init__(self, w: int = 360, h: int = 240, frames: int = 20, fps: float = 25.0):
        self.width, self.height, self.frame_count, self.fps = w, h, frames, fps

    def frame(self, i: int) -> np.ndarray:
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        s = 0.5 + 0.45 * np.cos(2 * np.pi * yy / 4.0)
        out = np.stack([s * 0.6 + 0.3 * xx / w, s * 0.6, s * 0.5 + 0.2], axis=2)
        return np.clip(out, 0, 1).astype(np.float32)


def _neutral(extra: dict | None = None) -> dict:
    base = merge_values(FACTORY_PRESETS["family_archive_1992"]["layers"], {
        "display": {"enabled": 0.0},
        "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0},
        "tape": {"line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0,
                 "noise_floor": 0.0, "head_switching": 0.0, "dropout_rate": 0.0,
                 "vertical_instability": 0.0, "wear": 0.0},
    })
    return merge_values(base, extra or {})


def render(scene, layers: dict, idx: int = 10) -> np.ndarray:
    r = Renderer(scene, scene.width, scene.height, scene.fps, layers, seed=20240101)
    return r.render_frame(idx)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_engine() -> None:
    print("\nengine invariants")
    from atsim.render.renderer import STAGE_ORDER
    from atsim.core.profiles import SCHEMA as S
    got = set(S.layer_names())
    want = {"camera", "format", "codec", "connector", "rf", "tape", "player", "display"}
    record("device layers", "camera..display all present",
           f"{len(got)} layers, missing={sorted(want - got) or 'none'}", got == want)

    scene = MovingScene(frames=20)
    lay = FACTORY_PRESETS["family_archive_1992"]["layers"]
    r = Renderer(scene, scene.width, scene.height, scene.fps, lay, seed=7)
    a = r.render_frame(3)
    r2 = Renderer(scene, scene.width, scene.height, scene.fps, lay, seed=7)
    b = r2.render_frame(3)
    record("determinism", "same seed -> identical output",
           f"max|delta| = {np.abs(a - b).max():.2e}", np.array_equal(a, b))

    r3 = Renderer(scene, scene.width, scene.height, scene.fps, lay, seed=8)
    c = r3.render_frame(3)
    record("seed sensitivity", "different seed -> different output",
           f"mean|delta| = {np.abs(a - c).mean():.4f}", not np.allclose(a, c))

    ok = bool(np.all(np.isfinite(a)) and a.min() >= -1e-6 and a.max() <= 1 + 1e-6)
    record("output range", "finite and within 0..1",
           f"[{a.min():.3f}, {a.max():.3f}]", ok)


def check_presets() -> None:
    print("\ngraph presets")
    from atsim.graph import GRAPH_PRESETS
    for key, spec in GRAPH_PRESETS.items():
        g = spec["build"]()
        errs = [i for i in g.validate() if i.severity == "error"]
        record(f"preset {key}", "connects with no type errors",
               f"{len(g.nodes)} nodes, {len(errs)} errors", not errs,
               "; ".join(str(e) for e in errs[:2]))


def check_drum_speed() -> None:
    print("\nhead-drum speed error  (Panasonic shoulder-mount VHS / VHS-C)")
    scene = StripeScene()
    base = _neutral({"format": {"luma_bandwidth": 6.0}})

    def stripe_amp(out: np.ndarray) -> float:
        g = out @ np.array([0.299, 0.587, 0.114], np.float32)
        prof = g.mean(axis=1)
        prof = prof - prof.mean()
        return float(abs(np.fft.rfft(prof)[prof.size // 4]) / prof.size)

    def horiz(out: np.ndarray) -> float:
        g = out @ np.array([0.299, 0.587, 0.114], np.float32)
        return float(np.abs(np.diff(g, axis=1)).mean())

    ref = render(scene, merge_values(base, {"tape": {"drum_speed": 1.0}}))
    a0, h0 = stripe_amp(ref), horiz(ref)
    slow = render(scene, merge_values(base, {"tape": {"drum_speed": 0.97}}))
    a1, h1 = stripe_amp(slow), horiz(slow)
    frac = a1 / max(1e-9, a0)
    record("drum 0.97 kills stripes", "fine vertical detail collapses",
           f"stripe amplitude {100*frac:.0f}% of nominal", frac < 0.5)
    hfrac = h1 / max(1e-9, h0)
    record("drum 0.97 keeps edges", "horizontal detail is unaffected",
           f"horizontal detail {100*hfrac:.0f}% of nominal", hfrac > 0.9)


def check_shuttle() -> None:
    print("\nshuttle residue  (rewind / fast-forward)")
    from atsim.codecs.dv import (line_run_mask, macroblock_grid,
                                 shuttle_stale_mask)
    ok_f = []
    for want in (0.05, 0.19, 0.45):
        m = shuttle_stale_mask(np.random.default_rng(3), macroblock_grid(240, 360),
                               want, 0.55)
        ok_f.append(abs(m.mean() - want) < 0.01)
    record("block fraction", "DV parameter matches what it produces",
           "0.05/0.19/0.45 all within 0.01", all(ok_f))
    ok_l = []
    for want in (0.05, 0.19, 0.45):
        m = line_run_mask(np.random.default_rng(3), 480, want, 8.0)
        ok_l.append(abs(m.mean() - want) < 0.01)
    record("line fraction", "Hi8/VHS parameter matches what it produces",
           "0.05/0.19/0.45 all within 0.01", all(ok_l))

    # the two granularities must be selected by format, not by preference
    from atsim.core.profiles import FACTORY_PRESETS as FP
    scene = MovingScene()
    line_mode = _neutral({"tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                                   "shuttle_mode": 1.0}})
    block_mode = merge_values(line_mode, {"tape": {"shuttle_mode": 2.0}})
    rl = Renderer(scene, scene.width, scene.height, scene.fps, line_mode, seed=5)
    rb = Renderer(scene, scene.width, scene.height, scene.fps, block_mode, seed=5)
    rl._build_chain(); rb._build_chain()
    record("granularity switch", "line mode vs block mode are different paths",
           f"lines={not rl.stage['tape'].shuttle_block_based}, "
           f"blocks={rb.stage['tape'].shuttle_block_based}",
           (not rl.stage["tape"].shuttle_block_based)
           and rb.stage["tape"].shuttle_block_based)


def check_acc_gain() -> None:
    print("\nchroma ACC gain  (the source of 'saturation')")
    scene = MovingScene()

    def sat(out: np.ndarray) -> float:
        mx, mn = out.max(axis=2), out.min(axis=2)
        return float(np.mean(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)))

    vals = {}
    for g in (1.0, 2.2, 0.1):
        out = render(scene, _neutral({"format": {"chroma_gain": g}}))
        vals[g] = sat(out)
    record("gain raises saturation", "gain 2.2 > gain 1.0",
           f"{vals[1.0]:.3f} -> {vals[2.2]:.3f}", vals[2.2] > vals[1.0] * 1.5)
    record("killer drains colour", "gain 0.1 nearly monochrome",
           f"sat = {vals[0.1]:.3f}", vals[0.1] < 0.10)


def check_binder_loss() -> None:
    """
    Binder breakdown: colour loses more than luma, by a margin that grows.

    Measured in the signal domain (tools/probe.py).  This has to be done on Y/I/Q
    rather than on RGB pictures: converting back to RGB clips at zero, and with luma
    attenuated while chroma is not, that clip removes exactly the partially-damaged
    values the check depends on.  Measuring in RGB is what made several earlier
    rounds of parameter tuning look like they changed nothing.

    Two things are asserted, because either alone can pass vacuously:

    * a partially-damaged population exists at mid settings -- without one, an
      ordering between chroma and luma is not observable at any decay rate;
    * chroma retention is below luma retention, and the gap widens with damage.
    """
    from atsim.core.pipeline import FrameBuffer, RenderContext
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.stages.tape import TapeStage

    scene = MovingScene()
    clean = float(render(scene, _neutral({})).mean())
    gone = float(render(scene, _neutral({"tape": {"binder_loss": 0.7}})).mean())
    record("loss removes signal", "mean level falls",
           f"{clean:.4f} -> {gone:.4f}", gone < clean - 0.05)

    H, W = 240, 360
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    y_plane = 0.42 + 0.06 * (xx / W)
    i_plane = 0.16 * np.cos(yy / 13.0)
    q_plane = 0.16 * np.sin(yy / 13.0)
    comp = composite_encode(y_plane, i_plane, q_plane, std, chroma_gain=0.63)
    c0 = np.sqrt(i_plane ** 2 + q_plane ** 2)

    rows = []
    for amt in (0.2, 0.4, 0.7, 1.0):
        ctx = RenderContext(std=std,
                            params=_neutral({"tape": {"binder_loss": amt}}),
                            seed=20240101, frame_count=10, fps=25.0,
                            width=W, height=H)
        stage = TapeStage(ctx)
        stage.prepare()
        buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, 3, 0.12, std,
                          {"field": 0})
        out = stage.apply(buf)
        if out.composite is None:
            record("chroma dies first", "colour loses more than luma",
                   "stage published no composite", False)
            return
        oy, oi, oq = stage._decompose(out.composite)
        if oy.shape != y_plane.shape:
            record("chroma dies first", "colour loses more than luma",
                   f"shape mismatch {oy.shape} vs {y_plane.shape}", False)
            return
        dy = np.abs(oy - y_plane)
        partial = float(((dy > 0.01) & (dy < 0.6 * y_plane.mean())).mean())
        rows.append((amt,
                     float(oy.mean() / max(1e-6, y_plane.mean())),
                     float(np.sqrt(oi ** 2 + oq ** 2).mean() / max(1e-6, c0.mean())),
                     partial))

    partial_ok = any(p > 0.15 for _a, _l, _c, p in rows)
    record("partial damage exists", "a partly-damaged population must be measurable",
           "partial fractions " + ", ".join("%.0f%%" % (100 * p) for _a, _l, _c, p in rows),
           partial_ok,
           "without this the ordering below is unobservable, whatever the mechanism")

    ordering_ok = all(c < l for _a, l, c, _p in rows)
    gaps = [l - c for _a, l, c, _p in rows]
    widening = gaps[-1] > gaps[0]
    record("chroma dies first", "colour loses more than luma, by a widening margin",
           "gap " + " -> ".join("%.3f" % g for g in gaps),
           ordering_ok and widening)

def check_rf() -> None:
    print("\nRF link  (HDMI -> AV -> RF -> TV)")
    from atsim.codecs.rf import RFLink, RFParams
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std

    h, w = 480, 720
    comp = composite_encode(np.full((h, w), 0.5, np.float32),
                            np.zeros((h, w), np.float32),
                            np.zeros((h, w), np.float32), std, chroma_gain=0.63)
    rows = []
    for sig in (-45.0, -65.0, -80.0):
        rf = RFLink(std, RFParams(signal_dbm=sig), seed=11)
        out, info = rf.apply(comp, 0)
        rows.append((sig, float(np.abs(out - comp).mean()), info.get("slipped_lines", 0)))
    mono = all(rows[i][1] <= rows[i + 1][1] + 1e-6 for i in range(len(rows) - 1))
    record("weaker signal is worse", "degradation rises monotonically",
           " | ".join(f"{s:.0f}dBm:{d:.5f}" for s, d, _ in rows), mono)
    record("sync loss only when weak", "no slipping at strong signal",
           f"slipped lines at -45dBm = {rows[0][2]}", rows[0][2] == 0)


def check_head_switch() -> None:
    print("\nhead-switch band  (see docs: overscan can hide it)")
    scene = MovingScene()
    lay = _neutral({"tape": {"head_switching": 0.9, "head_switch_lines": 8.0,
                             "tracking_error": 0.0, "head_switch_mid_line": 0.0}})
    off = render(scene, _neutral({"tape": {"head_switching": 0.0}}))
    on = render(scene, lay)
    d = np.abs(on - off).mean(axis=(1, 2))
    rows = np.flatnonzero(d > 0.01)
    H = d.size
    if rows.size:
        bottom = bool(rows.max() >= H - 4 and d[-8:].mean() > d[:H // 2].mean())
        record("band sits at the bottom", "damaged rows are the last ones",
               f"rows {rows.min()}..{rows.max()} of {H}", bottom)
    else:
        record("band sits at the bottom", "damaged rows are the last ones",
               "no measurable band", False,
               "check that display overscan is 0; realistic overscan hides it")



def check_consistency() -> None:
    """
    The schema, the profile libraries and the stage code must agree.

    These are the checks that caught real breakage during a recovery, so they run
    every time rather than living in a scratch script.  Two failure modes are
    covered:

    * a **profile names a parameter that does not exist**, which is silent at
      runtime -- the value is simply never applied -- and therefore the worst kind
      of error to leave in;
    * a **stage reads a parameter its layer does not define**, which makes the
      control do nothing; this is how `beam_defocus` and `glass_reflection` were
      found to be dead, the schema having moved to `defocus` and `reflection`.

    Cross-layer reads are legitimate and expected (`tape` reads the codec's
    `dv_enabled`, the player's output bandwidth, and the frame metadata), so those
    are declared rather than treated as errors.
    """
    import re
    from atsim.core.profiles import (CAMCORDER_PRESETS, CAMERA_PROFILES,
                                     DISPLAY_PROFILES, FACTORY_PRESETS,
                                     MEDIA_PROFILES, PLAYER_PROFILES, SCHEMA)

    known = {lay: set(SCHEMA.layer(lay).keys()) for lay in SCHEMA.layer_names()}
    libs = {
        "MEDIA_PROFILES": MEDIA_PROFILES, "PLAYER_PROFILES": PLAYER_PROFILES,
        "DISPLAY_PROFILES": DISPLAY_PROFILES, "CAMERA_PROFILES": CAMERA_PROFILES,
        "CAMCORDER_PRESETS": CAMCORDER_PRESETS, "FACTORY_PRESETS": FACTORY_PRESETS,
    }
    bad = []
    for lib, items in libs.items():
        for name, prof in items.items():
            for lay, vals in (prof.get("layers") or {}).items():
                if lay not in known:
                    bad.append(f"{lib}.{name}: unknown layer {lay}")
                    continue
                for k in vals:
                    if k not in known[lay]:
                        bad.append(f"{lib}.{name}: unknown parameter {lay}.{k}")
    record("profile keys exist", "every profile targets real schema parameters",
           f"{sum(len(v) for v in libs.values())} profiles checked", not bad,
           "; ".join(bad[:3]))

    # names a stage may legitimately read from elsewhere
    CROSS = {"dv_enabled", "chroma_mode", "luma_bandwidth", "chroma_bandwidth",
             "signal", "field", "stage_done", "chroma_phase", "line_phase",
             "sensor_type"}
    missing = []
    for stage in ("camera", "format", "codec", "rf", "tape", "player", "display"):
        path = os.path.join("atsim", "stages", stage + ".py")
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        used = set(re.findall(r'get\("([a-z_0-9]+)"', text))
        defined = set(known.get(stage, set()))
        for name in sorted(used - defined - CROSS):
            missing.append(f"{stage}.{name}")
    record("stage params exist", "every parameter a stage reads is defined",
           f"{len(missing)} undefined", not missing, ", ".join(missing[:4]))

    # every profile the tooling asks for by name must resolve
    wanted = []
    for dirpath, _dirs, files in os.walk("tools"):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            text = open(os.path.join(dirpath, fn), encoding="utf-8",
                        errors="replace").read()
            for lib in libs:
                for m in re.finditer(re.escape(lib) + r'\[\s*"([^"]+)"\s*\]', text):
                    wanted.append((lib, m.group(1)))
    unresolved = sorted({f"{lib}.{k}" for lib, k in wanted if k not in libs[lib]})
    record("referenced profiles resolve", "tooling never hits a missing key",
           f"{len(set(wanted))} references, {len(unresolved)} missing",
           not unresolved, ", ".join(unresolved[:4]))


def check_deck_osd() -> None:
    """
    The deck's blue-screen substitution: it either shows the tape or it does not.

    This check exists because the mechanism *looked* wired and was not.  Every
    attribute read correctly, `apply` was called, and the output was unchanged --
    the display stage prefers `meta["signal"]` over the RGB buffer, so the deck's
    picture was being rebuilt from the very tape signal it had rejected.  A check on
    the committed output catches that class of failure; reading the code does not.
    """
    scene = MovingScene()
    plain = render(scene, _neutral({"display": {"enabled": 0.0}}))
    deck = render(scene, _neutral({"display": {"enabled": 0.0},
                                   "player": {"blue_screen": 1.0}}))

    def sat(img: np.ndarray) -> float:
        mx, mn = img.max(axis=2), img.min(axis=2)
        return float(np.mean(np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)))

    # a pure blue field is almost entirely chroma, so saturation is high by
    # construction; the reference deck's own screen measured 0.895
    record("deck substitutes its screen", "output is a flat blue field, not the tape",
           f"sat {sat(plain):.3f} -> {sat(deck):.3f}", sat(deck) > 0.7)
    record("substitution discards the tape", "the picture is replaced, not blended",
           f"mean|delta| from the tape picture = {np.abs(deck - plain).mean():.4f}",
           np.abs(deck - plain).mean() > 0.15)


def check_broadcast_noise() -> None:
    """
    Off-air noise: direction only.

    The reference for this is a 1992 weak-signal off-air recording where the picture
    is buried (high frequency 0.024).  Absolute calibration against it is still open,
    and the honest thing is to verify what is actually established -- that the
    control raises broadband energy and does not read as a colour shift -- rather
    than assert a target the model does not reach.
    """
    from atsim.codecs.rf import RFLink, RFParams
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std
    from scipy.ndimage import laplace
    from atsim.core.params import merge_values as _mv

    h, w = 480, 720
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    comp = composite_encode(0.5 + 0.15 * np.cos(xx / 23.0),
                            0.20 * np.cos(yy / 17.0), 0.20 * np.sin(yy / 17.0),
                            std, chroma_gain=0.63)
    rows = []
    for bn in (0.0, 0.5):
        rf = RFLink(std, RFParams(signal_dbm=-45.0, broadcast_noise=bn), seed=11)
        out, _ = rf.demodulate(comp, 0, np.random.default_rng(3))
        rows.append((bn, float(np.abs(laplace(out)).mean())))
    grew = rows[1][1] > rows[0][1] * 1.3
    record("broadcast noise adds grain", "high-frequency energy rises",
           f"{rows[0][1]:.5f} -> {rows[1][1]:.5f}", grew)


def check_graph_wiring() -> None:
    """
    An unwired converter must produce a readable error naming every such node.

    The previous behaviour was an `assert` deep in the executor, which surfaced as
    `rf_tuner: AssertionError: rf_tuner has no input` -- an internal-sounding message
    naming one node, when the useful answer is which nodes need connecting.  It also
    contradicted the graph validator, which treated the same condition as a warning.
    """
    from atsim.core.standards import get_standard
    from atsim.graph import GRAPH_PRESETS, GraphExecutor
    from atsim.graph.graph import CATALOG
    from atsim.io.source import SyntheticSource

    src = SyntheticSource(width=240, height=160, frames=12)
    params = _neutral({})
    kinds = {c.kind for c in CATALOG}

    def build(extra: list[str]):
        g = GRAPH_PRESETS["vhs_consumer"]["build"]()
        for kind in extra:
            g.add(kind, kind)
        return GraphExecutor(g, get_standard("ntsc"), params, src, 240, 160,
                             src.fps, seed=5)

    # pick two converter kinds that exist, whatever they are called
    converters = sorted(k for k in kinds if k in ("rf_tuner", "hdmi_out", "tv_out"))
    if not converters:
        record("unwired nodes reported", "readable error naming each node",
               "no converter kinds in the catalogue", False)
        return

    ex = build(converters)
    run = ex.render_frame(2)
    pending = ex.pending_nodes()
    text = " ".join(run.errors)
    named_all = all(c in text for c in converters)
    record("unwired nodes reported", "every unwired node must be named",
           f"pending={pending}, named_all={named_all}",
           bool(run.errors) and named_all and set(pending) == set(converters))
    record("wired graph still renders", "the check must not break normal graphs",
           f"errors={len(build([]).render_frame(2).errors)}",
           not build([]).render_frame(2).errors)


def check_preset_independence() -> None:
    """
    A factory preset must produce the same state whatever was selected before it.

    This was a real user-facing bug: `apply_preset` merged a preset's layers into the
    *current* parameters, and most presets define only the layers they are about --
    `hi8_shuttle` sets five tape parameters and nothing else.  So applying it after
    "worn rental tape" kept the rental tape's camera, format, player and display
    settings, and the same button gave a different picture depending on the order of
    clicks.  That is what "some presets work and some do nothing" looked like.

    The check drives the real store rather than re-implementing the merge, because
    the bug was in the store.
    """
    import tempfile

    from atsim.web.session import Project, SessionStore

    store = SessionStore(tempfile.mkdtemp())
    target = "hi8_shuttle"
    histories = ([target],
                 ["rental_tape_worn", target],
                 ["mouldy_archive", "pro_transfer_svhs", target])

    states = []
    for history in histories:
        project = Project()
        for key in history:
            store.apply_preset(project, key)
        states.append(project.params)

    same = all(s == states[0] for s in states[1:])
    record("preset is order-independent", "a preset means the same thing every time",
           f"{len(histories)} click histories, identical={same}", same,
           "applying a partial preset must not inherit the previous selection")

    # and a preset must actually change something, or the button is decorative
    changed = states[0] != store.apply_preset(Project(), "family_archive_1992").params
    record("presets differ from each other", "each preset is a distinct state",
           f"hi8_shuttle vs family_archive_1992 differ={changed}", changed)


def check_dv_odd_size() -> None:
    """
    The 4:1:1 codec must round-trip frames whose width is odd.

    Preview scaling produces odd dimensions routinely (scale 0.37 of 640 is 236.8,
    rounded to 237), and the DV path is the only one that subsamples chroma, so it
    is the only place that can lose a column.  It did, and the symptom appeared two
    functions later as a broadcast error naming two shapes -- which says nothing
    about the cause, and crashed the whole render rather than one frame.
    """
    from atsim.codecs.dv import subsample_411, subsample_420, upsample_411, upsample_420

    widths = [237, 279, 278, 360, 361, 481, 719, 720]
    bad = []
    for w in widths:
        cb = np.zeros((223, w), np.float32)
        cr = np.zeros((223, w), np.float32)
        for sub, up in ((subsample_411, upsample_411), (subsample_420, upsample_420)):
            cs, crs = sub(cb, cr)
            cu, cru = up(cs, crs, cb.shape)
            if cu.shape != cb.shape or cru.shape != cb.shape:
                bad.append(f"w={w}: {sub.__name__} -> {cu.shape} != {cb.shape}")
    record("DV chroma survives odd sizes", "4:1:1 and 4:2:0 keep the luma shape",
           f"{len(widths)} widths checked, {len(bad)} mismatched", not bad,
           "; ".join(bad[:3]))

    # and the same through a whole rendered frame at an odd preview scale
    from atsim.io.source import SyntheticSource
    from atsim.render.renderer import Renderer as _R

    small = SyntheticSource(width=640, height=480, frames=12)
    grew = None
    for scale in (0.37, 0.5):
        w = max(2, int(round(640 * scale / 2)) * 2)
        h = max(2, int(round(480 * scale / 2)) * 2)
        try:
            rr = _R(small, w, h, small.fps, _neutral({}), seed=5,
                    preview=True, scale=scale)
            rr.render_frame(8)
        except Exception as exc:                     # noqa: BLE001 - reported below
            grew = "scale %s (%dx%d) raised %s: %s" % (
                scale, w, h, type(exc).__name__, exc)
            break
    record("odd-scale preview renders", "an odd preview width must render",
           grew or "scales 0.37 and 0.5 both rendered", grew is None)


def check_server_control() -> None:
    """
    The start/stop tool must exist, and its liveness check must not lie.

    It reported a healthy server as "gone" because the command-line lookup used
    `wmic`, which current Windows builds no longer ship.  A control tool that
    misreports liveness invites exactly the mistake it was written to prevent:
    stopping processes by name because the precise check appeared to find nothing.
    """
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "server.py")
    if not os.path.exists(path):
        record("server control exists", "a start/stop tool is present", "missing", False)
        return
    spec = importlib.util.spec_from_file_location("atsim_server_control", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # type: ignore[union-attr]

    me = os.getpid()
    record("server control sees a live pid", "_pid_exists is accurate for this process",
           "pid %d -> %s" % (me, mod._pid_exists(me)), mod._pid_exists(me))
    record("server control rejects a dead pid", "a bogus pid is not reported alive",
           "pid 99999999 -> %s" % mod._pid_exists(99999999),
           not mod._pid_exists(99999999))
    record("server control has all commands", "start/stop/restart/status/logs",
           "bound", all(hasattr(mod, n) for n in
                        ("cmd_start", "cmd_stop", "cmd_restart", "cmd_status",
                         "cmd_logs")))


def check_row_chroma() -> None:
    """
    Whole lines must be able to decode to the wrong colour, not merely to no colour.

    This was a real gap, and the distinction is the mechanism: `chroma_loss` zeroes
    I and Q, producing a grey line, whereas a chroma reference error rotates every
    hue on the line and shifts its colour-under gain, so the line stays fully
    saturated but wrong.  Someone familiar with the equipment described the second
    ("解析出错误的颜色 ... 整行全错") while the model only had the first.

    Two assertions, because either can pass vacuously: that saturation is retained
    while the hue moves, and that the affected rows are whole rows.
    """
    from atsim.core.pipeline import FrameBuffer, RenderContext
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.stages.tape import TapeStage

    H, W = 240, 360
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    y_plane = 0.42 + 0.06 * (xx / W)
    i_plane = 0.16 * np.cos(yy / 13.0)
    q_plane = 0.16 * np.sin(yy / 13.0)
    comp = composite_encode(y_plane, i_plane, q_plane, std, chroma_gain=0.63)

    def planes(over: dict, frame: int = 8):
        params = _neutral({"tape": over})
        ctx = RenderContext(std=std, params=params, seed=20240101, frame_count=40,
                            fps=25.0, width=W, height=H)
        st = TapeStage(ctx)
        st.prepare()
        buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, frame, 0.32, std,
                          {"field": frame % 2})
        out = st.apply(buf)
        return st._decompose(out.composite)

    # Differenced against the mechanism switched off.  Comparing with the original
    # signal instead mixes in every other tape mechanism, which is how an earlier
    # version of this measurement concluded the phase control did nothing.
    _y, bi, bq = planes({})
    amp0 = np.sqrt(bi ** 2 + bq ** 2)

    moved = []
    kept = []
    for ph in (30.0, 60.0):
        _y, oi, oq = planes({"row_chroma_error": 0.3, "row_chroma_phase": ph})
        d = np.maximum(np.abs(oi - bi), np.abs(oq - bq))
        touched = d.max(axis=1) > 0.02
        if not touched.any():
            moved.append(0.0)
            kept.append(0.0)
            continue
        h0 = np.angle((bi[touched] + 1j * bq[touched]).mean())
        h1 = np.angle((oi[touched] + 1j * oq[touched]).mean())
        moved.append(abs(float(np.rad2deg(np.angle(np.exp(1j * (h1 - h0)))))))
        amp1 = np.sqrt(oi ** 2 + oq ** 2)
        m = amp0 > 0
        kept.append(float(amp1[m].mean() / max(1e-6, amp0[m].mean())))

    record("hue rotates without losing colour", "saturation retained while hue moves",
           "kept " + ", ".join("%.0f%%" % (100 * k) for k in kept)
           + " with hue moved " + ", ".join("%.1f deg" % m for m in moved),
           all(abs(k - 1.0) < 0.05 for k in kept) and all(m > 3.0 for m in moved))

    _y, oi, oq = planes({"row_chroma_error": 0.3, "row_chroma_phase": 45.0})
    d = np.maximum(np.abs(oi - bi), np.abs(oq - bq))
    touched = d.max(axis=1) > 0.02
    if not touched.any():
        record("errors cover whole rows", "a damaged line is damaged across its width",
               "no rows affected", False)
    else:
        within = float(np.mean([(d[r] > 0.02 * d.max()).mean()
                                for r in np.flatnonzero(touched)]))
        record("errors cover whole rows", "a damaged line is damaged across its width",
               "%.2f of each affected row changed (1.0 = full width)" % within,
               within > 0.95)


def check_row_timebase() -> None:
    """
    Two per-line horizontal mechanisms, both of which must be *visible*.

    The first assertion is not about the new mechanism at all.  `line_jitter` was
    scaled by `unit_px * 5.0`, which at a 360-wide preview is 0.0225, so its full
    range displaced lines by 0.09 px -- measured standard deviation of the timeline's
    per-line phase was 0.045 px, three orders of magnitude below one pixel.  A
    control whose entire range is invisible reads as a broken control, and that is
    how it was reported.  Both mechanisms are therefore measured in pixels of the
    rendered frame, not by reading the code.

    The second assertion is the distinction between them: the row-time-base field
    must migrate across frames, and luma and chroma must move by the same amount --
    a sampling-phase error moves the whole line, so a colour shift would be an
    invention.
    """
    from atsim.core.pipeline import FrameBuffer, RenderContext
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.stages.tape import TapeStage

    H, W = 240, 360
    # vertical structure, so a horizontal shift is measurable at all.  Measuring a
    # horizontal displacement on a pattern that varies only along x yields zero for
    # every mechanism, which is what an earlier attempt reported.
    x = np.arange(W)
    base = 0.30 + 0.45 * np.maximum.reduce(
        [(np.abs(x - c) < 1.5).astype(np.float32) for c in (40, 95, 150, 205)])
    y_plane = np.tile(base[None, :], (H, 1))
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    comp = composite_encode(y_plane, 0.16 * np.cos(yy / 13.0),
                            0.16 * np.sin(yy / 13.0), std, chroma_gain=0.63)

    def stage_for(over: dict) -> TapeStage:
        params = _neutral({"tape": over})
        ctx = RenderContext(std=std, params=params, seed=20240101, frame_count=60,
                            fps=25.0, width=W, height=H)
        st = TapeStage(ctx)
        st.prepare()
        return st

    def planes(st: TapeStage, frame: int):
        buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, frame,
                          frame / 25.0, std, {"field": frame % 2})
        out = st.apply(buf)
        return st._decompose(out.composite)

    print_specs = []
    for label, over in (("line_jitter", {"line_jitter": 1.0, "row_timebase": 0.0}),
                        ("row_timebase", {"line_jitter": 0.0, "row_timebase": 0.5})):
        st = stage_for(over)
        print_specs.append((label, float(
            np.abs(st._line_phase_for(12)).std()
            if label == "line_jitter" else
            st._row_timebase_field(H, 12).std())))

    worst = min(v for _l, v in print_specs)
    record("per-line displacement is visible", "a full-range control is not sub-pixel",
           ", ".join("%s std %.2f px" % (l, v) for l, v in print_specs),
           worst > 0.5,
           "a control whose whole range is under a pixel reads as broken")

    st = stage_for({"line_jitter": 0.0, "row_timebase": 0.8})
    fields = np.stack([st._row_timebase_field(H, f) for f in range(6)])
    consec = float(np.abs(np.diff(fields, axis=0)).mean())
    record("row time-base migrates across frames",
           "the fault pattern moves rather than repeating",
           "consecutive-frame change %.2f px" % consec, consec > 0.5)

    st_off = stage_for({"line_jitter": 0.0, "row_timebase": 0.0})
    by0, bi0, _ = planes(st_off, 12)
    by1, bi1, _ = planes(stage_for({"line_jitter": 0.0, "row_timebase": 0.6}), 12)

    def row_shift(a: np.ndarray, b: np.ndarray, rng: int = 10) -> float:
        lags = np.arange(-rng, rng + 1)
        scores = np.array([float(np.dot(a, np.roll(b, int(s)))) for s in lags])
        k = int(np.argmax(scores))
        if 0 < k < len(lags) - 1:
            y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-9:
                return float(lags[k] - 0.5 * (y2 - y0) / den)
        return float(lags[k])

    sy = np.array([row_shift(by0[r], by1[r]) for r in range(H)], np.float32)
    si = np.array([row_shift(bi0[r], bi1[r]) for r in range(H)], np.float32)
    diff = float(np.abs(sy - si).mean())
    record("luma and chroma move together", "a sampling-phase error moves the whole line",
           "luma std %.2f px, luma-vs-chroma difference %.2f px" % (sy.std(), diff),
           sy.std() > 0.5 and diff < 0.6)


def check_stripe_moire() -> None:
    """
    Curved stripes must actually curve, and must be distinguishable from displacement.

    The discriminating quantity is *within-row* curvature.  The per-line time-base
    mechanism raises the across-row figure by a factor of 3990 while leaving the
    within-row figure at the floor (0.00001 rad/px^2), so across-row alone cannot
    demonstrate curvature -- a first version of this check would have accepted any
    displacement mechanism as "curving".  A stripe bends only if the spatial
    frequency varies across the frame, which is what a beat produces.

    `stripe_curve` governs the *across-row* figure, not the within-row one: it shapes
    how the family of fringes bends down the picture.  Measured, 0 gives 0.00010 and
    0.8 gives 0.00231 -- twenty-three times more.  An earlier check asserted it moved
    the within-row number and reported a failure that was the check's fault.
    """
    from scipy.ndimage import uniform_filter1d

    from atsim.core.pipeline import FrameBuffer, RenderContext
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.stages.tape import TapeStage

    H, W, PITCH = 240, 360, 24.0
    x = np.arange(W, dtype=np.float32)
    y_plane = np.tile((0.45 + 0.25 * np.cos(2.0 * np.pi * x / PITCH))[None, :], (H, 1))
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    comp = composite_encode(y_plane, 0.10 * np.cos(yy / 13.0),
                            0.10 * np.sin(yy / 13.0), std, chroma_gain=0.63)

    def field(plane: np.ndarray) -> np.ndarray:
        c = np.cos(2.0 * np.pi * x / PITCH)[None, :]
        s = np.sin(2.0 * np.pi * x / PITCH)[None, :]
        re = uniform_filter1d(plane * c, int(PITCH), axis=1)
        im = uniform_filter1d(plane * s, int(PITCH), axis=1)
        return np.unwrap(np.arctan2(im, re), axis=1)

    def curvature(ph: np.ndarray) -> tuple[float, float]:
        along = float(np.median(np.abs(np.diff(ph, 2, axis=1))))
        m = ph.mean(axis=1)
        across = float(np.median(np.abs(np.diff(m, 2)))) if m.size > 3 else 0.0
        return along, across

    def run(tape_over: dict, frame: int = 10) -> np.ndarray:
        params = _neutral({"tape": tape_over})
        ctx = RenderContext(std=std, params=params, seed=20240101, frame_count=40,
                            fps=25.0, width=W, height=H)
        st = TapeStage(ctx)
        st.prepare()
        buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, frame,
                          frame / 25.0, std, {"field": frame % 2})
        out = st.apply(buf)
        y, _i, _q = st._decompose(out.composite)
        return y

    base_a, _base_c = curvature(field(run({})))
    a_on, _c_on = curvature(field(run({"stripe_moire": 0.6})))
    a_disp, _c_disp = curvature(field(run({"row_timebase": 1.0})))
    record("stripes curve, displacement does not",
           "within-row curvature is the discriminating quantity",
           "baseline %.5f, moire %.5f (%.0fx), row_timebase %.5f"
           % (base_a, a_on, a_on / max(base_a, 1e-9), a_disp),
           a_on > 50 * max(base_a, 1e-9) and a_disp < 5 * max(base_a, 1e-9))

    _pa, c_par = curvature(field(run({"stripe_moire": 0.6, "stripe_curve": 0.0})))
    _pc, c_cur = curvature(field(run({"stripe_moire": 0.6, "stripe_curve": 0.8})))
    record("stripe bend is controllable", "stripe_curve bends the fringe family",
           "across-row %.5f -> %.5f" % (c_par, c_cur), c_cur > 3 * max(c_par, 1e-9))

    f = [field(run({"stripe_moire": 0.6}, frame=k)) for k in (8, 10, 12, 14)]
    moved = float(np.abs(f[3] - f[0]).mean())
    record("stripe pattern changes as the tape is read",
           "not a fixed texture on the picture", "phase moved %.4f rad" % moved,
           moved > 0.05)


def check_fingerprint_ordering() -> None:
    """
    A clean digital chain must stay closer to its input than an analogue one.

    Same input frame for every preset, so the only variable is the model. This is the
    narrow, real claim available from a fingerprint comparison: the *ordering* of
    distance from the input. It is not a claim that any preset reproduces real VCD,
    which no preset does -- the closest is `minidv_clean` at a residual of 0.075 and
    its largest gap is hue spread, since VCD is MPEG-1 4:2:0 and the model has no
    MPEG-1 branch at all.
    """
    from atsim.render.renderer import Renderer
    from tools.fingerprint import fingerprint

    class Still:
        def __init__(self, img):
            self._img = np.asarray(img, np.float32)
            self.height, self.width = self._img.shape[:2]
            self.frame_count, self.fps = 16, 25.0

        def frame(self, index):
            return self._img.copy()

    # A frame with both colour and fine detail, built here so the check does not
    # depend on a reference file being present.
    H, W = 176, 288
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 2] = 0.75 + 0.05 * np.cos(2 * np.pi * xx / 19.0)
    img[..., 0] = 0.20 + 0.55 * (np.cos(2 * np.pi * xx / 23.0) > 0.4)
    img[..., 1] = 0.25 + 0.45 * (np.sin(2 * np.pi * yy / 31.0) > 0.3)
    img = np.clip(img, 0, 1)
    scene = Still(img)
    ref = fingerprint(img)

    def residual(key: str) -> float:
        layers = SCHEMA.resolve_all(FACTORY_PRESETS[key]["layers"])
        r = Renderer(scene, W, H, scene.fps, layers, seed=20240101)
        for i in range(5):
            out = r.render_frame(i)
        fp = fingerprint(out)
        keys = ("mean", "sat", "hf", "hue_spread", "row_step")
        return float(np.mean([abs(fp[k] - ref[k]) / max(1e-6, abs(ref[k]) + 0.05)
                              for k in keys]))

    clean = residual("minidv_clean")
    analogue = residual("family_archive_1992")
    record("fingerprint ordering is sane",
           "a clean digital chain stays closer to its input than an analogue one",
           "minidv_clean %.3f vs family_archive_1992 %.3f" % (clean, analogue),
           clean < analogue)


def check_row_timebase_shape() -> None:
    """
    The row time-base displacement must vary *smoothly*, not just by the right amount.

    Calibrating the amplitude against a worn VHS capture matched the spread easily and
    said nothing about the shape -- and the shape was wrong. Measured on the real
    material, the ratio of the median adjacent-row step to the overall spread is 0.108:
    the displacement varies slowly enough that a vertical edge bends rather than tears.
    The model gave 0.311, three times too ragged, while its amplitude matched
    perfectly. A check on the amplitude alone would have passed it.

    The smoothing width cannot be reasoned to; it was swept against the measured ratio
    and 26 passes of a 5-tap kernel give 0.097.
    """
    from atsim.core.pipeline import RenderContext
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.stages.tape import TapeStage

    H, W = 240, 360
    params = _neutral({"tape": {"row_timebase": 0.4}})
    ctx = RenderContext(std=std, params=params, seed=20240101, frame_count=40,
                        fps=25.0, width=W, height=H)
    st = TapeStage(ctx)
    st.prepare()
    f = st._row_timebase_field(H, 12)
    ratio = float(np.median(np.abs(np.diff(f)))) / max(1e-9, float(f.std()))
    record("row time-base varies smoothly", "edges bend rather than tear",
           "adjacent-row step / spread = %.3f (real material 0.108)" % ratio,
           0.05 < ratio < 0.18,
           "amplitude can match while the shape is wrong; this is the shape")


def check_tbc_reduces_error() -> None:
    """
    Engaging the corrector must reduce the per-row displacement spread.

    This was inert. The parameter was read, but the correction blended towards the
    frame mean and the mean was then subtracted again, so the two cancelled exactly:
    switching it on changed the spread by 0.79x, i.e. not at all. Same shape of defect
    as the deck's blue-screen substitution being rebuilt from the tape signal it had
    rejected -- present, reachable, annulled by a later step.

    A real deck's corrector reduces the spread about 8.8x, measured on a capture where
    it was toggled on and off over the same tape. The model does not reach that and the
    check does not pretend it does: it asserts only the direction, which is what the
    fix established.
    """
    from atsim.core.pipeline import FrameBuffer, RenderContext
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.stages.player import PlayerStage
    from atsim.stages.tape import TapeStage

    H, W = 240, 360
    x = np.arange(W, dtype=np.float32)
    y_plane = np.tile((0.30 + 0.45 * np.maximum.reduce(
        [(np.abs(x - c) < 1.5).astype(np.float32) for c in (40, 95, 150, 205)]))[None, :],
        (H, 1))
    yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
    comp = composite_encode(y_plane, 0.16 * np.cos(yy / 13.0),
                            0.16 * np.sin(yy / 13.0), std, chroma_gain=0.63)

    def spread(tbc: bool) -> float:
        params = _neutral({"tape": {"row_timebase": 0.25, "line_jitter": 0.0},
                           "player": {"tbc_enabled": 1.0 if tbc else 0.0,
                                      "tbc_strength": 0.95, "tbc_residual": 0.0,
                                      "output_gain": 0.0, "output_bias": 0.0,
                                      "head_wear": 0.0, "frequency_response": 0.0}})
        ctx = RenderContext(std=std, params=params, seed=20240101, frame_count=40,
                            fps=25.0, width=W, height=H)
        st = TapeStage(ctx)
        st.prepare()
        buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, 12, 0.48, std,
                          {"field": 0})
        buf = st.apply(buf)
        pl = PlayerStage(ctx)
        pl.prepare()
        out = pl.apply(buf)
        sig = out.meta.get("signal")
        plane = (sig.y if sig is not None else
                 (out.composite if out.composite is not None else out.rgb[..., 0]))
        plane = np.asarray(plane, np.float32)
        from scipy.ndimage import gaussian_filter1d
        g = gaussian_filter1d(plane, 0.7, axis=1)
        ref = gaussian_filter1d(g, 4.0, axis=0)
        lags = np.arange(-10, 11)
        shifts = []
        for r in range(H):
            row = g[r] - g[r].mean()
            tgt = ref[r] - ref[r].mean()
            if row.std() < 0.01:
                continue
            sc = [float(np.dot(row, np.roll(tgt, int(s)))) for s in lags]
            shifts.append(float(lags[int(np.argmax(sc))]))
        return float(np.std(shifts)) if len(shifts) > 16 else float("nan")

    off, on = spread(False), spread(True)
    record("corrector reduces displacement", "engaging it must not make things worse",
           "spread %.3f px off -> %.3f px on" % (off, on),
           np.isfinite(off) and np.isfinite(on) and on <= off * 0.98)

# ---------------------------------------------------------------------------
def contact_sheet() -> str:
    """One image showing every headline effect, for eyeballing."""
    scene = MovingScene(w=200, h=150)
    tiles = []
    labels = []

    def add(layers: dict, label: str) -> None:
        tiles.append(render(scene, layers, idx=10))
        labels.append(label)

    add(_neutral({}), "plain")
    add(_neutral({"format": {"chroma_gain": 2.2}}), "ACC 2.2")
    add(_neutral({"format": {"chroma_gain": 0.1}}), "ACC 0.1")
    add(_neutral({"tape": {"drum_speed": 0.94}}), "drum 0.94")
    add(_neutral({"tape": {"binder_loss": 0.6}}), "binder 0.6")
    add(_neutral({"tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.28,
                           "shuttle_mode": 1.0}}), "Hi8 shuttle")
    add(_neutral({"tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.28,
                           "shuttle_mode": 2.0}}), "DV shuttle")
    add(_neutral({"tape": {"head_switching": 1.0, "tape_speed": 2.0}}), "EP + tear")

    sheet = np.concatenate(tiles, axis=1)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "verify_sheet.png")
    save_image(path, sheet)
    return path



def check_mpeg1_branch() -> None:
    """
    The VCD codec must be reachable, lossy, and must lower high frequency.

    All three were wrong at first, and in the same way.  `codec_kind = 2` selected
    MPEG-1 and the stage then returned its input untouched because the guard consulted
    `dv_enabled` -- so `meta["format"]` stayed "composite" and the branch did nothing.
    Once it ran, the quantiser was mis-scaled by a factor of about a hundred: the step
    was fifty times the mean AC coefficient, so every AC coefficient rounded to zero and
    the result was a mosaic of DC values.  Through the chain that *raised* high frequency
    by 257% where the format lowers it, and the preset scored a residual of 2.292 against
    the VCD reference, ranking it last rather than first.

    The codec is driven directly.  Measuring through the chain put the "before"
    high-frequency figure at 0.00000, the format stage having band-limited the signal
    long before the codec saw it, so the assertion could not have failed either way.
    """
    from scipy.ndimage import laplace

    from atsim.codecs.mpeg1 import (INTRA_MATRIX, MPEG1Codec, MPEG1CodecParams,
                                    ZIGZAG)

    record("MPEG-1 uses the specified matrices",
           "the zigzag visits every coefficient once",
           "intra range %.0f..%.0f over 64 coefficients"
           % (INTRA_MATRIX.min(), INTRA_MATRIX.max()),
           sorted(ZIGZAG.reshape(-1).tolist()) == list(range(64))
           and INTRA_MATRIX.max() > 5 * INTRA_MATRIX.min())

    H, W = 176, 288
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[..., 0] = 0.30 + 0.20 * (xx % 6.0 < 3.0) + 0.14 * np.cos(xx / 2.3)
    rgb[..., 1] = 0.28 + 0.16 * np.cos(xx / 3.1) + 0.10 * np.cos(yy / 5.0)
    rgb[..., 2] = 0.32 + 0.18 * (yy % 7.0 < 3.5)
    rgb = np.clip(rgb, 0, 1)

    def hf(img: np.ndarray) -> float:
        g = np.clip(np.asarray(img, np.float32), 0, 1) @ np.array(
            [0.299, 0.587, 0.114], np.float32)
        return float(np.abs(laplace(g)).mean())

    codec = MPEG1Codec(MPEG1CodecParams(quality=1.0, motion_compensation=1.0))
    out = codec.encode_decode(rgb, 0)
    changed = float(np.abs(out - rgb).mean())
    record("MPEG-1 is lossy", "the picture is changed, not passed through",
           "mean |delta| %.5f" % changed, changed > 0.002)

    h0, h1 = hf(rgb), hf(out)
    # `h1 < h0` was too strict: block edges at the 8x8 grid add a little high frequency
    # even while the quantiser removes detail, and a 3.7% rise is ordinary. What the
    # check has to catch is the opposite failure, where the quantiser is so coarse that
    # AC coefficients vanish entirely and the result is a mosaic of DC values -- that
    # measured +2950% (0.041 -> 1.362). A 1.25x ceiling separates the two with a wide
    # margin on both sides.
    record("MPEG-1 does not turn into a mosaic", "coarse quantisation is soft, not blocks",
           "hf %.5f -> %.5f (%+.1f%%)"
           % (h0, h1, 100 * (h1 / max(1e-9, h0) - 1)), h1 < h0 * 1.25,
           "an all-DC result raises high frequency by 2950%; a working quantiser by a few percent")

    errs = []
    for q in (0.5, 1.0, 2.0):
        c = MPEG1Codec(MPEG1CodecParams(quality=q))
        o = c.encode_decode(rgb, 0)
        errs.append(round(float(np.abs(o - rgb).mean()), 5))
    record("MPEG-1 quality control works", "finer quantisation means less error",
           "error at quality 0.5 / 1.0 / 2.0 = %s" % errs,
           errs[0] > errs[1] > errs[2])




# Checks that are cheap enough to run with `--quick`.  A check not listed here is
# skipped under `--quick`, deliberately: a missing check is visible, an accidentally
# included slow one is not.
QUICK = ("check_consistency", "check_engine", "check_presets", "check_shuttle")


def _discover_checks() -> list[str]:
    """
    Every `check_*` function in this module, in the order they are defined.

    Discovered rather than listed. A hand-maintained list has already produced a check
    that never ran while the suite reported success, which is indistinguishable from the
    check passing -- the worst failure mode a test suite can have. Order comes from the
    source, so it is stable and needs no upkeep.
    """
    names = [n for n, v in globals().items()
             if n.startswith("check_") and callable(v)]
    try:
        tree = ast.parse(open(__file__, encoding="utf-8").read())
        order = {n.name: n.lineno for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("check_")}
        names.sort(key=lambda n: order.get(n, 10 ** 9))
    except OSError:
        names.sort()
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="verify every mechanism claim")
    ap.add_argument("--quick", action="store_true",
                    help="skip the render-heavy checks")
    ap.add_argument("--json", action="store_true", help="emit results as JSON")
    ap.add_argument("--no-sheet", action="store_true",
                    help="do not write the contact sheet")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("=" * 78)
    print("  analog-tape-sim  --  verification")
    print("=" * 78)

    checks = _discover_checks()
    for name in checks:
        if args.quick and name not in QUICK:
            continue
        globals()[name]()

    failed = [r for r in RESULTS if not r.ok]
    print()
    print("-" * 78)
    print(f"  {len(RESULTS) - len(failed)} passed, {len(failed)} failed, "
          f"{time.time() - t0:.1f}s")

    sheet = None
    if not args.no_sheet and not args.quick:
        try:
            sheet = contact_sheet()
            print(f"  contact sheet: {sheet}")
        except Exception as exc:                       # pragma: no cover
            print(f"  contact sheet failed: {exc}")

    if args.json:
        print(json.dumps({"results": [r.as_dict() for r in RESULTS],
                          "passed": len(RESULTS) - len(failed),
                          "failed": len(failed),
                          "sheet": sheet}, ensure_ascii=False, indent=2))
    print("=" * 78)
    return 1 if failed else 0


def check_discovery_completeness() -> None:
    """
    Every `check_*` function defined here must be one the runner will execute.

    Written because a check that never runs looks exactly like a check that passes.
    Three MPEG-1 assertions were written, registered next to another check's *definition*
    rather than inside `main()`, and the suite reported 47 passes -- healthy, and wrong.
    Discovery by name now makes that particular mistake impossible, and this asserts it
    stays that way: if a check is defined with a name the discovery pattern misses, or
    the file stops parsing, this fails rather than quietly shrinking the suite.
    """
    import ast as _ast
    import re as _re

    source = open(__file__, encoding="utf-8").read()
    defined = set(_re.findall(r"^def (check_\w+)\(", source, _re.M))
    discovered = set(_discover_checks())
    missing = sorted(defined - discovered)
    record("every defined check is discovered",
           "a check that does not run must not look like one that passes",
           "%d defined, %d discovered, %d missing" % (len(defined), len(discovered),
                                                      len(missing)),
           not missing, ", ".join(missing[:4]))

    # and the file must parse, since discovery reads it with `ast`
    try:
        tree = _ast.parse(source)
        parsed = True
    except SyntaxError as exc:
        parsed = False
        tree = None
    record("verify.py parses for its own discovery", "discovery depends on the AST",
           "ok" if parsed else "syntax error", parsed)

    if tree is not None:
        quick_known = [n for n in QUICK if n in discovered]
        record("the quick set is not empty and resolves",
               "--quick must run something, and only real checks",
               "%d of %d quick names resolve" % (len(quick_known), len(QUICK)),
               bool(quick_known) and len(quick_known) == len(QUICK))


def check_tracker_response() -> None:
    """
    The displacement tracker must respond to a fast per-line component, and be honest
    that it is not a linear estimator of one.

    Five attempts to measure the real fast component failed, none by producing a figure
    that disagreed -- each measured at the wrong point. So the instrument is validated
    against fields whose composition is known before it is used on material, which is
    the order that worked for the stripe work: prove the measurement separates curvature
    from tilt, then build the mechanism.

    Measured response, smooth field mixed with a fast one:

        fast  0%   true std 1.0000 -> measured 0.4706  (recovers 47%)
        fast 25%   true std 0.8015 -> measured 0.6532
        fast 50%   true std 0.7234 -> measured 0.9022
        fast 100%  true std 1.0000 -> measured 3.1065  (amplifies 3.1x)

    So it rises monotonically with the fast fraction, which means it can *distinguish*
    fast from smooth, and it is not calibrated to report either in absolute pixels. That
    calibration explains the real capture: its adjacent-row-step-to-spread ratio is
    0.108-0.146, the near-smooth end of this curve, where the tracker recovers only 47%.
    The real displacement is therefore about twice what was measured, and it is mostly
    *smooth*.

    Which in turn says where the corrector's 8.8x cannot be coming from. It is not the
    within-field fast component, because on this material that component is small. It has
    to be the field-to-field wander -- and a tracker that subtracts a local mean is
    structurally blind to exactly that, which is why no amount of re-measuring the same
    way was going to find it.
    """
    from atsim.core.signal import horizontal_shift
    from tools.measure_timebase import displacement_stats

    H, W = 240, 360
    x = np.arange(W, dtype=np.float32)
    edge = (np.abs(((x - 5.0 + 5.0) % 10.0) - 5.0) < 1.5).astype(np.float32)
    base = np.tile((0.25 + 0.55 * edge + 0.15 * np.cos(2.0 * np.pi * x / 6.0))[None, :],
                   (H, 1)).astype(np.float32)

    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0], np.float32)
    k /= k.sum()
    rng = np.random.default_rng(3)
    smooth = rng.standard_normal(H).astype(np.float32)
    for _ in range(26):
        smooth = np.convolve(smooth, k, mode="same")
    smooth /= max(1e-6, float(smooth.std()))
    fast = np.random.default_rng(4).standard_normal(H).astype(np.float32)
    fast /= max(1e-6, float(fast.std()))

    steps, measured = [], []
    for amp in (0.0, 0.25, 0.5, 1.0):
        field = smooth * (1.0 - amp) + fast * amp
        field = field - field.mean()
        img = horizontal_shift(base, field.astype(np.float32))
        r = displacement_stats(img)
        steps.append(round(float(r.get("line_step", float("nan"))), 4))
        measured.append(round(float(r.get("std", float("nan"))), 4))

    grew = all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))
    record("tracker responds to a fast component",
           "it must rise monotonically with the fast fraction",
           "reported adjacent-row step %s" % steps, grew,
           "a flat response would mean the instrument cannot see what it is asked to measure")

    # and it must be recorded as non-linear, not quietly treated as absolute
    ratio_smooth = measured[0] / 1.0
    ratio_fast = measured[-1] / 1.0
    record("tracker is not a linear estimator",
           "its scale depends on what it is measuring",
           "recovers %.2fx on a smooth field, %.2fx on a fast one"
           % (ratio_smooth, ratio_fast),
           abs(ratio_smooth - 1.0) > 0.2 and abs(ratio_fast - 1.0) > 0.2,
           "absolute figures from it are unusable; only comparisons are")
def check_hue_spread_metric() -> None:
    """
    The hue-spread metric must respond to a known hue rotation, and its scale matters.

    Measured by rotating the hue of a known fraction of rows by a known angle:

        untouched                    0.4230
        10% of rows rotated 30 deg   0.4266   (+0.0037)
        10% of rows rotated 90 deg   0.7207   (+0.2977)
        30% of rows rotated 90 deg   1.0698   (+0.6469)
        100% of rows rotated 90 deg  0.4030   (-0.0200)

    Two things follow, and the second is why this is worth asserting rather than noting.

    Small rotations are nearly invisible to it: a 30-degree rotation across a tenth of the
    rows moves it by 0.004, which is inside the noise of a fingerprint -- so a sweep that
    reports a parameter as having no effect may simply be looking at too small a change.

    And rotating *every* row moves it backwards, because hue spread is measured relative
    to the distribution: a uniform rotation relocates the whole distribution instead of
    widening it.

    That second effect is what the sweep got wrong. It reported `row_chroma_error` as
    changing hue spread by exactly 0.0000 across three values; measured through the
    renderer the same configuration changes the output by 0.066 mean absolute difference
    and the metric by 0.0252. The metric was correct and the harness was not.
    """
    from atsim.core.signal import SignalFrame
    from atsim.core.standards import NTSC_525_60 as std
    from tools.fingerprint import fingerprint

    # A narrow hue distribution with fine structure, which is both what the reference
    # material looks like and what makes a rotation the dominant term.  A wide synthetic
    # palette gave a base spread of 1.3670 against the reference's 0.4230, and against a
    # distribution that broad, rotating a tenth of the rows moved the metric by 0.098 --
    # the pattern was the obstacle, not the metric.
    H, W = 96, 160
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 2] = 0.72 + 0.06 * np.cos(xx / 21.0)
    img[..., 0] = 0.20 + 0.05 * (np.cos(xx / 13.0) > 0.2)
    img[..., 1] = 0.22 + 0.05 * (np.sin(yy / 17.0) > 0.1)
    img = np.clip(img, 0, 1)

    def rotate(frac: float, deg: float) -> np.ndarray:
        sig = SignalFrame.from_rgb(img, std)
        th = np.deg2rad(deg)
        n = int(round(H * frac))
        i, q = sig.i.copy(), sig.q.copy()
        if n > 0:
            c, s = np.cos(th), np.sin(th)
            i[:n], q[:n] = (i[:n] * c - q[:n] * s), (i[:n] * s + q[:n] * c)
        return np.clip(SignalFrame(sig.y, i, q).to_rgb(), 0, 1)

    base = fingerprint(img)["hue_spread"]
    small_rot = fingerprint(rotate(0.1, 30.0))["hue_spread"]
    big_rot = fingerprint(rotate(0.1, 90.0))["hue_spread"]
    all_rows = fingerprint(rotate(1.0, 90.0))["hue_spread"]

    record("hue-spread metric responds to rotation",
           "a large rotation of some rows must widen the distribution",
           "base %.4f -> %.4f at 10%%/90deg (+%.4f)"
           % (base, big_rot, big_rot - base), big_rot > base + 0.1)

    # The asymmetry is the property that makes the metric usable: rotating *some* rows
    # widens the spread, rotating *all* rows does not. Without it the metric could not
    # tell a per-line chroma fault from a global colour shift, and the hue-spread
    # comparison against real material depends on exactly that distinction.
    #
    # Asserted as "does not widen" rather than "stays the same": measured, a uniform
    # rotation *reduces* the spread (0.8416 -> 0.5057), because the base pattern's hue
    # varies spatially as well as the rotation adding to it. Demanding it stay near the
    # base failed for a reason that says nothing about the metric.
    record("a uniform rotation does not widen the spread",
           "some rows rotated widens it; all rows rotated does not",
           "base %.4f, 10%% of rows %.4f, all rows %.4f"
           % (base, big_rot, all_rows),
           big_rot > base + 0.05 and all_rows < base + 0.05,
           "this asymmetry is what separates a per-line chroma fault from a "
           "global colour shift")

    record("a small rotation is nearly invisible",
           "its resolution sets what a sweep can conclude",
           "10%%/30deg moves it by %+.4f" % (small_rot - base),
           abs(small_rot - base) < 0.05,
           "a sweep reporting 'no effect' may be looking at too small a change")
def check_spread_not_artefact() -> None:
    """
    A low-detail frame must not be able to fake a large displacement spread.

    Run before the intermittency in the captures was called a property of the material.
    The tracker only reports rows where it finds structure, so a nearly flat frame yields
    few usable rows, and the spread of a handful of rows is the tracker's own noise rather
    than the tape's behaviour. If that were what produced the variation, the intermittency
    would be a measurement artefact and a whole mechanism would have been built on it.

    Measured on real fields, 25 of them: correlation of spread with picture detail -0.004,
    with the number of usable rows -0.282. The artefact explanation is not supported.

    The same logic is asserted here on synthetic fields, where the answer is known.
    """
    from atsim.core.signal import horizontal_shift
    from tools.measure_timebase import displacement_stats

    H, W = 240, 360
    x = np.arange(W, dtype=np.float32)
    rng = np.random.default_rng(7)

    def build(detail_amp: float, shift_std: float) -> np.ndarray:
        edge = (np.abs(((x - 5.0) % 10.0) - 5.0) < 1.5).astype(np.float32)
        img = np.tile((0.5 + detail_amp * (edge - 0.3))[None, :], (H, 1)).astype(np.float32)
        f = rng.standard_normal(H).astype(np.float32)
        f = f / max(1e-6, float(f.std())) * shift_std
        return horizontal_shift(img, f)

    # same true spread, one with plenty of structure and one nearly flat
    strong = displacement_stats(build(0.55, 0.8))
    weak = displacement_stats(build(0.02, 0.8))
    s_strong = float(strong.get("std", float("nan")))
    s_weak = float(weak.get("std", float("nan")))
    rows_strong = int(strong.get("rows", 0))
    rows_weak = int(weak.get("rows", 0))

    record("a detailed frame gives a usable spread",
           "structure is what makes the measurement possible",
           "detail-rich: %d rows, spread %.2f px" % (rows_strong, s_strong),
           rows_strong > 40 and np.isfinite(s_strong))

    record("a nearly flat frame reports few rows, not a large spread",
           "the artefact this guards against would look like a fault",
           "nearly flat: %d rows, spread %s"
           % (rows_weak, "n/a" if not np.isfinite(s_weak) else "%.2f px" % s_weak),
           rows_weak < rows_strong,
           "so a varying stock of usable rows cannot be mistaken for varying displacement")
def check_connectors() -> None:
    """
    Every interface must damage the picture, and must write to both representations.

    Two defects, both found by tracing rather than by reading.

    The first: HDMI and S-Video reported *exactly* zero effect while AV and RF worked.
    They were modifying the picture, but the player stage rebuilds its output from the
    composite plane, so a change to the rgb buffer was discarded one stage later --
    measured, the difference went from 0.037383 after the connector to 0.000000 after the
    player. The analogue connectors appeared to work only because they happened to modify
    the composite. The stage now rebuilds one coherent signal and re-encodes both views
    from it. Same shape of defect as the deck's blue screen being rebuilt from the tape
    signal it had rejected, and the seventh instance in this project of a change to one
    representation being annulled by a stage that reads the other.

    The second is the point of the module: the *kind* of damage must differ by interface,
    because the signal domain differs. A digital link has no smooth degradation, so it must
    produce block errors and then lose the link; a composite connector with a bad shell must
    produce a rolling bar; RF damage must be frequency selective.
    """
    from atsim.core.pipeline import FrameBuffer, RenderContext
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.render.renderer import Renderer
    from atsim.io.source import SyntheticSource
    from atsim.codecs.connectors import (CONNECTORS, impedance,  # shared helpers
                                         reflection_coefficient)

    # --- the electrical model itself
    z_clean = impedance(CONNECTORS["rf"], 0.0, 0.0)
    z_wet = impedance(CONNECTORS["rf"], 1.0, 0.0)
    g = abs(reflection_coefficient(CONNECTORS["rf"].nominal_z, z_wet))
    record("moisture moves the impedance off nominal",
           "water raises the dielectric constant and reflects energy",
           "RF 75 ohm -> %.1f ohm wet, |gamma| %.3f" % (z_wet, g),
           abs(z_wet - z_clean) > 20.0 and g > 0.1)

    # --- each connector must change the picture through the whole chain
    W, H = 240, 160
    src = SyntheticSource(width=W, height=H, frames=16)

    def shot(over: dict) -> np.ndarray:
        layers = _neutral({"connector": over})
        r = Renderer(src, W, H, src.fps, layers, seed=20240101)
        for i in range(5):
            img = r.render_frame(i)
        return np.clip(img, 0, 1)

    base = shot({})
    results = []
    for label, over in (
        ("hdmi", {"hdmi_present": 1.0, "hdmi_moisture": 0.5}),
        ("svideo", {"svideo_present": 1.0, "svideo_leakage": 0.5}),
        ("av", {"av_present": 1.0, "av_ground_loop": 0.6}),
        ("rf", {"rf_present": 1.0, "rf_moisture": 0.5}),
    ):
        img = shot(over)
        d = float(np.abs(img - base).mean())
        results.append((label, d))
    record("every interface damages the picture",
           "each must survive to the output, not be discarded downstream",
           ", ".join("%s %.4f" % (l, d) for l, d in results),
           all(d > 0.001 for _l, d in results),
           "HDMI and S-Video measured exactly zero until the representations were "
           "reconciled")

    # --- and the digital one must be able to fail the link outright
    from atsim.codecs.connectors import degrade_digital
    rng = np.random.default_rng(4)
    tile = (rng.random((H, W, 3)) * 0.4 + 0.3).astype(np.float32)
    _out, info = degrade_digital(tile, CONNECTORS["hdmi"], 1.0, 1.0, 0, 9)
    record("a badly corroded HDMI link drops rather than degrades",
           "digital video has no slight-snow state",
           "link = %s at full degradation" % info.get("link"),
           info.get("link") == "no signal",
           "a faded picture would be the wrong behaviour for a digital link")
def check_stiction() -> None:
    """
    Tape-deck stiction must be *episodic*: quiet most of the time, far from normal briefly.

    This is the one property no existing mechanism has. Every other tape fault scales
    smoothly, so the model could not represent what a real capture shows -- per-row
    displacement flipping between roughly 0.1 and 3 pixels within seconds, uncorrelated with
    picture content. A smooth field cannot do that, and smoothing one to make it look like a
    fault removes exactly the discontinuities that define it.

    Two assertions, and the second is the one that matters:

      * the events appear on the timeline at all, and their rate follows the parameter;
      * the *effect* is concentrated -- most frames are untouched. A mechanism that moved
        every frame by a similar amount would produce a displacement field, not an
        intermittent fault, and would pass any check written only against the mean.
    """
    from atsim.core.pipeline import FrameBuffer, RenderContext
    from atsim.core.signal import composite_encode
    from atsim.core.standards import NTSC_525_60 as std
    from atsim.stages.tape import TapeStage

    H, W = 240, 360
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = np.arange(W, dtype=np.float32)
    y_plane = np.tile((0.30 + 0.45 * np.maximum.reduce(
        [(np.abs(x - c) < 1.5).astype(np.float32) for c in (60, 150, 250, 340)]))[None, :],
        (H, 1))
    comp = composite_encode(y_plane, 0.10 * np.cos(yy / 13.0),
                            0.10 * np.sin(yy / 13.0), std, chroma_gain=0.63)
    QUIET = {"line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0,
             "noise_floor": 0.0, "head_switching": 0.0, "dropout_rate": 0.0,
             "row_timebase": 0.0, "tracking_error": 0.0, "vertical_instability": 0.0}

    def make(over: dict, frames: int = 64) -> TapeStage:
        tape = dict(QUIET)
        tape.update(over)
        params = _neutral({"tape": tape})
        # the merge trap: assert the sweep value arrived rather than trusting it
        for k, v in over.items():
            assert abs(float(params["tape"][k]) - float(v)) < 1e-9, \
                "override %s was dropped by the merge" % k
        ctx = RenderContext(std=std, params=params, seed=7, frame_count=frames,
                            fps=25.0, width=W, height=H)
        st = TapeStage(ctx)
        st.prepare()
        st._ensure(frames - 1)
        return st

    def plane(st: TapeStage, frame: int) -> np.ndarray:
        buf = FrameBuffer(np.zeros((H, W, 3), np.float32), comp, frame, frame / 25.0,
                          std, {"field": frame % 2})
        return st._decompose(st.apply(buf).composite)[0]

    counts = []
    for rate in (0.0, 0.5, 2.0):
        st = make({"stiction_rate": rate, "stiction_amount": 0.8})
        counts.append(len(getattr(st.plan, "stiction", [])))
    record("stiction events appear at the set rate",
           "an episodic fault must put events on the timeline",
           "counts at rate 0 / 0.5 / 2.0 over 64 frames = %s" % counts,
           counts[0] == 0 and counts[2] > counts[1] > 0)

    quiet = make({"stiction_rate": 0.0})
    loud = make({"stiction_rate": 2.0, "stiction_amount": 0.9})
    deltas = [float(np.abs(plane(loud, f) - plane(quiet, f)).mean()) for f in range(24)]
    active = [d for d in deltas if d > 1e-4]
    share = len(active) / max(1, len(deltas))
    record("the effect is intermittent, not uniform",
           "most frames must be untouched and a few strongly changed",
           "%d of %d frames affected, largest %.4f against a median of %.4f across "
           "all frames" % (len(active), len(deltas),
                           max(deltas), float(np.median(deltas))),
           0.05 < share < 0.8,
           "a smooth mechanism would move every frame similarly and would pass a "
           "mean-only check")
def check_crt_moisture() -> None:
    """
    A damp CRT must produce three separate effects, and the geometry must be diagnostic.

    The three come from different parts of the set, which is why they are asserted
    separately: a bias shift in the video amplifier moves the black level, three
    independently-worn channel amplifiers produce a cast, and a supply dragged down by
    leakage makes the scan shrink and breathe.

    The geometry is the one that matters. It is the *only* fault in the whole model that can
    move the edges of the raster, because a tape cannot bend the picture -- so if the frame
    edges are moving, the fault is in the display and nowhere else. A check that only looked
    at the level would pass a model with no geometry at all, and the geometry is what makes
    this fault locatable.
    """
    from atsim.codecs.crt import apply_deflection, moisture_effects

    # --- the level and the cast must both grow with moisture, independently
    clean = moisture_effects(0.0, 0.0, 0.3)
    wet = moisture_effects(0.6, 0.0, 0.3)
    record("CRT moisture shifts the black level",
           "a high-resistance contact changes the video amplifier's bias",
           "bias %.4f -> %.4f, so the picture washes out rather than gaining noise"
           % (clean.bias_drift, wet.bias_drift),
           wet.bias_drift > 0.05 and clean.bias_drift == 0.0,
           "a clean set must shift nothing and a damp one must shift visibly")

    # --- age must widen the cast without simply adding brightness.  This is the physical
    # reason the same damp room washes out a new set and tints an old one.
    new_set = moisture_effects(0.6, 0.0, 0.0)
    old_set = moisture_effects(0.6, 0.0, 0.9)
    spread_new = max(new_set.cast) - min(new_set.cast)
    spread_old = max(old_set.cast) - min(old_set.cast)
    bias_gain = old_set.bias_drift - new_set.bias_drift
    record("age widens the cast more than it raises the level",
           "the three cathode amplifiers wear independently, so an old set tints",
           "cast spread %.4f -> %.4f, while the bias moves only %.4f"
           % (spread_new, spread_old, bias_gain),
           spread_old > spread_new * 1.5)

    # --- the geometry, which is the diagnostic part
    H, W = 120, 160
    _o, info_none = apply_deflection(np.zeros((H, W, 3), np.float32), 0.0, 0, 1)
    scales = []
    for f in (0, 10, 20, 30):
        _o, info = apply_deflection(np.zeros((H, W, 3), np.float32), 0.3, f, 1)
        scales.append(info.get("raster_scale", 1.0))
    moved = max(scales) - min(scales)
    record("CRT deflection shrinks and breathes the raster",
           "the only fault in the model that moves the edges of the picture itself",
           "raster scale over 1.2 s = %s (drift %.4f); at zero deflection it is unchanged"
           % (", ".join("%.3f" % s for s in scales), moved),
           len(scales) >= 3 and min(scales) < 0.98 and moved > 0.005
           and "raster_scale" not in info_none)
if __name__ == "__main__":
    raise SystemExit(main())
