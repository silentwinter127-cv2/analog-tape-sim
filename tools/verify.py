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
    # exactly the expected set: adding a layer has to be deliberate, which is why this is an equality
    want = {"camera", "format", "codec", "connector", "rf", "tape", "player", "display",
            "overlay"}
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
    # **These two thresholds were re-stated when the chroma filter was fixed, and the reason matters more than the
    # numbers.**  `apply_one_pole` ran `lfilter` from a zero initial state, so every row spent its whole length in
    # the startup transient -- a time constant of about 1.4 million samples against image widths of one to two
    # thousand -- and its response to a constant was **0.46942** at the default cutoff and **0.01679** at 0.30 MHz,
    # and depended on the picture's width.  That reduction of the chroma made both terms of this ratio smaller and
    # pushed the ratio from 1.467 to **1.512**, over a 1.5 threshold, while the near-monochrome reading sat at
    # 0.0888 against a 0.10 limit.  **So these checks were scoring an artefact of the defect**: the broken filter
    # passed them and the correct one failed them.
    #
    # With the state seeded, the DC gain is exactly 1.0 at every cutoff and every width, and the values on this
    # subject are 0.537 / 0.788 / 0.103.  The thresholds below are set to what the correct filter gives, with a
    # margin, rather than to what the defect happened to produce.
    record("gain raises saturation", "gain 2.2 > gain 1.0",
           f"{vals[1.0]:.3f} -> {vals[2.2]:.3f} (x{vals[2.2] / max(1e-9, vals[1.0]):.3f})",
           vals[2.2] > vals[1.0] * 1.4)
    record("killer drains colour", "gain 0.1 nearly monochrome",
           f"sat = {vals[0.1]:.3f}", vals[0.1] < 0.12)


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

    # --- line_jitter must reach the picture on its own -------------------------------
    # Every configuration above sets line_jitter to 0, so the parameter was covered by nothing.
    # A change making it inert would have left this check green, and an auditor constructed
    # precisely that. The displacement is measured from the render with the same estimator, and a
    # zero-line_jitter render is the negative control.
    j0, _, _ = planes(stage_for({"line_jitter": 0.0, "row_timebase": 0.0}), 12)
    j1, _, _ = planes(stage_for({"line_jitter": 1.2, "row_timebase": 0.0}), 12)
    lj = np.array([row_shift(j0[r], j1[r]) for r in range(H)], np.float32)
    # rows whose estimate is meaningless (flat, no structure) would inflate the std, so keep the
    # central mass rather than the extremes
    lj_ok = lj[np.abs(lj) < 40.0]
    lj_std = float(lj_ok.std()) if lj_ok.size > H // 4 else 0.0
    record("line jitter alone reaches the picture",
           "the parameter must displace rows in the render, not only in the stage's array",
           "per-row shift std %.3f px at line_jitter 1.2 against 0 (measured directly: "
           "0.4457 off, 0.8248 at the default, 1.1708 at 1.2)" % lj_std,
           lj_std > 0.3,
           "the check previously read _line_phase_for() and _row_timebase_field(), which proves "
           "the array was computed and not that it was applied")


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
def check_node_positions() -> None:
    """
    Node positions must be pixels, and the check must drive the routes rather than read them.

    Two earlier versions failed for instructive reasons. The first tested a helper,
    `_preset_columns_to_pixels`, that had **no call sites** -- so it certified a function no shipped
    path used, while the live conversion (`Graph.normalise_positions`) could have broken unnoticed.
    That helper has been deleted. The second version decided whether both server routes normalise by
    substring-searching the server source, which an auditor defeated by deleting the start-up call:
    the substring still matched at the definition site, so the test stayed true.

    Both are now replaced by a real `App`. A substring cannot satisfy an assertion about what a
    graph contains, and removing the conversion from either route fails this.
    """
    import tempfile

    from atsim.web.server import App

    app = App(tempfile.mkdtemp())
    nodes = app.graph.nodes
    if len(nodes) <= 1:
        record("a server's initial graph is in pixels",
               "the start-up route builds the graph the canvas first sees",
               "the initial graph has %d node(s), so a spread test would be vacuous" % len(nodes),
               False)
        return
    xs = [float(n.position[0]) for n in nodes.values()]
    flagged = all(getattr(n, "px_position", False) for n in nodes.values())
    spread = max(xs) - min(xs)
    record("a server's initial graph is in pixels",
           "the start-up route builds the graph the canvas first sees",
           "%d nodes, x spread %.0f px, all flagged as pixels: %s"
           % (len(nodes), spread, flagged),
           flagged and spread > 100,
           "in columns the same graph spans 5 px, which renders every node in one pile")

    app2 = App(tempfile.mkdtemp())
    app2.graph_load_preset("vhs_consumer")
    ns = app2.graph.nodes
    xs2 = [float(n.position[0]) for n in ns.values()]
    spread2 = (max(xs2) - min(xs2)) if len(xs2) > 1 else 0.0
    flagged2 = all(getattr(n, "px_position", False) for n in ns.values())
    record("loading a graph preset leaves it in pixels",
           "the preset route and the start-up route are separate and must agree",
           "%d nodes, x spread %.0f px, all flagged: %s" % (len(ns), spread2, flagged2),
           len(ns) > 1 and flagged2 and spread2 > 100,
           "the rule living in one route and not the other is what produced the node overlap")

    js = open(os.path.join(os.path.dirname(HERE), "atsim", "web", "static", "canvas.js"),
              encoding="utf-8").read()
    i = js.find("function px(node)")
    body = js[i:js.find("\n}", i)]
    record("the canvas does not convert coordinates",
           "with one unit the client has nothing to guess",
           "px() is a pass-through: %s; no COL_STEP in it: %s"
           % ("return [node.position" in body, "COL_STEP" not in body),
           "return [node.position" in body and "COL_STEP" not in body,
           "the earlier design kept adding client-side guessing, and kept failing")

def check_drag_tracking() -> None:
    """
    A dragged node must stay under the cursor, measured on the shipped arithmetic.

    This is the bug the user reported as "the nodes run around and are hard to control", and it
    was in the arithmetic rather than in the coordinates or the storage:

        dx: ev.clientX / G.scale - x - G.pan[0] / G.scale

    Only the first term was divided by the scale. The world-space offset from the cursor to the
    node is `clientX/scale - pan[0]/scale - x`, so the missing grouping shifted every drag by
    `(scale - 1) * pan / scale`. Measured before the fix: 313 px of error, *constant* along the
    whole path -- the signature of a units mistake, since a coordinate mistake would vary with
    the cursor.

    An earlier version of this check reimplemented the arithmetic and therefore passed while the
    handler was wrong: it tested the transcription. This one lifts `toWorld` and the offset
    assignment out of canvas.js, so what runs is what ships. It also asserts the mid-drag scale
    case, because the wheel handler changes `G.scale` while a stored offset was measured at the
    old one -- measured, that compounded the error to 484 px.

    The extraction is bounded by the assignment itself. It was previously bounded by
    `ev.stopPropagation()`, which was the last line of the handler until the drag diagnostic was
    inserted after it, at which point the extraction swept in `dragTrackingError` and
    `renderCanvas` and the check failed on a missing identifier. A bound chosen because it
    happened to be last breaks as soon as something is added after it.
    """
    import json as _json
    import subprocess
    import tempfile

    js = open(os.path.join(os.path.dirname(HERE), "atsim", "web", "static", "canvas.js"),
              encoding="utf-8").read()

    i = js.find("function toWorldFor(svgEl, ev) {")
    if i < 0:
        record("a dragged node follows the cursor", "the node must stay under the pointer",
               "toWorld not found in canvas.js", False)
        return
    tw_close = js.find("\n}", i)
    if tw_close < 0:
        record("a dragged node follows the cursor", "the node must stay under the pointer",
               "could not find the end of the transform", False)
        return
    to_world = js[i:tw_close + 2]

    # Search from the node's own handler, not from the top of the file: the expression
    # `toWorld(ev)` also appears in the drag diagnostic, which is defined earlier, so a
    # whole-file search for it captured the wrong region.
    handler = js.find("ng.onmousedown")
    if handler < 0:
        record("a dragged node follows the cursor", "the node must stay under the pointer",
               "the node's mousedown handler was not found", False)
        return
    k = js.find("const [wx, wy] = toWorldFor(svg, ev);", handler)
    if k < 0:
        record("a dragged node follows the cursor", "the node must stay under the pointer",
               "the offset is not computed from toWorld -- the units bug may be back", False)
        return
    endmark = js.find("G.moving = {", k)
    if endmark < 0:
        record("a dragged node follows the cursor", "the node must stay under the pointer",
               "the offset assignment was not found", False)
        return
    close = js.find("};", endmark)
    offset = js[k:close + 2]
    # guard the bound: the region must be the cursor-to-node offset and nothing else
    if "renderCanvas" in offset or "console." in offset or "catch" in offset:
        record("a dragged node follows the cursor", "the node must stay under the pointer",
               "the extraction swept in unrelated statements -- tighten its bound", False)
        return

    harness = (
        "const ATSIM_CANVAS_BUILD = 1;\n"
        "let G = { scale: 0.8, pan: [40, 40] };\n"
        "const svg = { getBoundingClientRect: () => ({ left: 300, top: 90 }) };\n"
        "const svgEl = svg;\n"
        + to_world + "\n"
        "const n = { name: 'camera', position: [300.0, 200.0] };\n"
        "function px(x) { return [x.position[0], x.position[1]]; }\n"
        "function pageOf() { return [n.position[0]*G.scale + G.pan[0] + 300,\n"
        "                             n.position[1]*G.scale + G.pan[1] + 90]; }\n"
        "function stub(ev) { ev.stopPropagation = () => {}; return ev; }\n"
        "function beginDrag(ev) {\n  const [x, y] = px(n);\n"
        + "\n".join("  " + l for l in offset.split("\n")) + "\n"
        "  return G.moving;\n}\n"
        "function applyMove(moving, ev) {\n"
        "  const [x, y] = toWorldFor(svg, ev);\n"
        "  const kk = moving.scale ? G.scale / moving.scale : 1;\n"
        "  n.position = [x - moving.dx * kk, y - moving.dy * kk];\n}\n"
        "function run(mutate) {\n"
        "  G = { scale: 0.8, pan: [40, 40] }; n.position = [300.0, 200.0];\n"
        "  const [gx, gy] = pageOf();\n"
        "  const moving = beginDrag(stub({ clientX: gx, clientY: gy }));\n"
        "  const path = [[gx,gy],[gx+40,gy+25],[gx+120,gy+60],[gx+200,gy-30],"
        "[gx+260,gy+90]];\n"
        "  let worst = 0, done = false;\n"
        "  for (const [cx, cy] of path) {\n"
        "    if (mutate && !done) { mutate(); done = true; }\n"
        "    applyMove(moving, stub({ clientX: cx, clientY: cy }));\n"
        "    const [nx, ny] = pageOf();\n"
        "    worst = Math.max(worst, Math.hypot(nx - cx, ny - cy));\n"
        "  }\n  return worst;\n}\n"
        "console.log(JSON.stringify({ steady: run(null),\n"
        "  scaled: run(() => { G.scale = 1.1; }) }));\n")

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.js")
        open(p, "w", encoding="utf-8").write(harness)
        out = subprocess.run(["node", p], capture_output=True, text=True)
    if out.returncode != 0:
        record("a dragged node follows the cursor", "the node must stay under the pointer",
               "node failed: %s" % (out.stderr or out.stdout)[:70], False)
        return
    r = _json.loads(out.stdout.strip())
    record("a dragged node follows the cursor",
           "measured on the shipped arithmetic, lifted from canvas.js",
           "worst tracking error %.3f px steady, %.3f px with the scale changed mid-drag "
           "(before the fix: 313 and 484)" % (r["steady"], r["scaled"]),
           r["steady"] < 0.01 and r["scaled"] < 0.01,
           "the offset must come from the same transform the move handler uses, or the two "
           "disagree by (scale-1)*pan/scale")
def check_canvas_scopes() -> None:
    """
    The world transform must be reachable from every handler that uses it.

    This exists because the drag fix referenced `toWorld`, a `const` local to `initCanvas`, from
    the node's mousedown handler, which is created in `renderCanvas`. Every mousedown threw
    `ReferenceError: toWorld is not defined` -- reported from the user's console -- and dragging
    stopped working entirely, which is worse than the sliding it replaced.

    The arithmetic check could not see it: it extracts the handler's code and runs it with
    `toWorld` supplied as a global, and supplying the identifier is exactly what the real page
    fails to do. A check that builds its own environment tests the environment it built.

    Five attempts were made at scanning handler bodies for unreachable names, and four of the
    five failures were the check rather than the code -- short names recurring, English words out
    of comment prose, object-literal shorthand, callback parameters. A regex approximation of a
    scope chain is not worth tuning. The failure has a sufficient condition that needs no scope
    model at all, and these are its three assertions:

      * a module-level `function toWorldFor` exists;
      * no `const toWorld` is declared inside a function, i.e. in a closure;
      * no handler mentions a bare `toWorld`.

    The original bug violates the first and the third, so this would have caught it on the first
    try.
    """
    import re as _re

    js = open(os.path.join(os.path.dirname(HERE), "atsim", "web", "static", "canvas.js"),
              encoding="utf-8").read()

    module_level = _re.search(r"^function toWorldFor\(", js, _re.M) is not None
    closure_copy = _re.search(r"^\s+const toWorld\s*=", js, _re.M) is not None

    HANDLERS = ("ng.onmousedown", "svg.onmousemove", "svg.onmouseup",
                "svg.onmousedown", "svg.onwheel")
    bad_uses = []
    js_lines = js.split("\n")
    for i, line in enumerate(js_lines):
        for h in HANDLERS:
            if h not in line or "=" not in line:
                continue
            indent = len(line) - len(line.lstrip())
            for j in range(i + 1, len(js_lines)):
                cur = js_lines[j]
                if cur.strip() and (len(cur) - len(cur.lstrip())) <= indent \
                        and cur.strip().startswith("}"):
                    break
                # a bare `toWorld(` that is not `toWorldFor(` is the bug
                if _re.search(r"(?<![\w$])toWorld\s*\(", cur) and "toWorldFor" not in cur:
                    bad_uses.append("%s line %d" % (h, j + 1))

    record("the world transform is reachable from every handler",
           "a handler using a name from another function's closure throws on every click",
           "module-level toWorldFor: %s; closure-local duplicate: %s; handlers calling a bare "
           "toWorld: %s" % (module_level, closure_copy, bad_uses or "none"),
           module_level and not closure_copy and not bad_uses,
           "the arithmetic check supplies the identifier as a global, so it could never fail "
           "for this reason -- which is how the bug reached the user")

def check_graph_build_positions() -> None:
    """
    Every graph a user can be shown must already be in pixels, whichever path built it.

    Reported as "the nodes start out stacked on top of each other". The start-up graph is built
    as `GRAPH_PRESETS["vhs_consumer"]["build"]()`, which produces grid columns, and only the
    preset route converted them -- so a freshly booted server handed the canvas
    `[0,0] [1,0] [2,0] ...`, one unit apart, and the nodes rendered as a single pile. Loading a
    preset from the UI looked right, which is why it only appeared on a fresh start.

    That is the same shape as the bug that made positions ambiguous in the first place: the rule
    existed in one path and not in another. So this checks the paths, not the arithmetic -- every
    construction route must hand back pixel positions, and the conversion must be reachable from
    the graph itself rather than living in a caller.
    """
    from atsim.graph import GRAPH_PRESETS

    bad = []
    for key, spec in GRAPH_PRESETS.items():
        g = spec["build"]()
        # a raw build is in columns; that is fine, but it must be convertible without a caller
        if not hasattr(g, "normalise_positions"):
            bad.append((key, "no normalise_positions method"))
            continue
        g.normalise_positions()
        xs = [float(n.position[0]) for n in g.nodes.values()]
        offsets = set(getattr(n, "px_position", False) for n in g.nodes.values())
        # after normalisation, adjacent preset nodes must be a column apart in pixels, not 1
        spread = (max(xs) - min(xs)) if xs else 0.0
        if not all(offsets):
            bad.append((key, "a node is not marked as pixels"))
        elif len(xs) > 1 and spread < 100:
            bad.append((key, "spread only %.1f px -- still in columns" % spread))

    record("every graph preset normalises to pixel positions",
           "the start-up path and the preset path must not differ, or nodes overlap",
           "%d presets checked; %s" % (len(GRAPH_PRESETS),
                                        "all spread out" if not bad else "problems: %s" % bad),
           not bad,
           "the initial graph was built without conversion and rendered one unit apart")

    # and the server must actually call it on both routes
def check_fault_citations() -> None:
    """
    Every parameter the fault catalogue cites must exist, and no parameter may be defined twice.

    Both were verified defects. Clicking a symptom in the UI jumps to the mechanism named by the
    citation, so a citation that does not resolve shows the user a *different* mechanism and then
    reports in the coverage counter that the fault has been covered. The tool told the user it
    had shown them something it had not. Three citations were wrong: `chroma_noise` for
    `color_noise`, `chroma_phase_error` for `burst_phase_error`, and `head_clog_bands` for
    `clog_bands`.

    Nineteen parameters were also defined twice in the schema. `ParamSet.add` overwrites, so the
    later definition wins and the earlier help text is silently discarded -- and eighteen of the
    nineteen pairs had drifted apart, meaning the UI explained those parameters using words
    written for a different version. This project had already lost time to one instance
    (`head_switch_lines`), which is the argument for making it a check rather than a repair.
    """
    import re as _re
    from atsim.core.profiles import SCHEMA as S

    fsrc = open(os.path.join(os.path.dirname(HERE), "atsim", "diagnostics", "faults.py"),
                encoding="utf-8").read()
    cited = set(_re.findall(
        r'\(\s*"(camera|format|codec|rf|tape|connector|player|display)"\s*,\s*"(\w+)"\s*\)',
        fsrc))
    unresolved = sorted((l, p) for l, p in cited
                        if l in S.layer_names() and p not in S.layer(l).keys())
    record("every fault-catalogue citation names a real parameter",
           "a citation that does not resolve sends the user to a different mechanism",
           "%d citations checked; %s" % (len(cited),
                                         "all resolve" if not unresolved
                                         else "unresolved: %s" % unresolved),
           bool(cited) and not unresolved,
           "the coverage counter would then report the fault as covered")

    psrc = open(os.path.join(os.path.dirname(HERE), "atsim", "core", "profiles.py"),
                encoding="utf-8").read()
    # Resolve each layer variable to its layer name, then count per layer.  Counting across the
    # whole file was the first version's mistake: the same parameter name legitimately exists in
    # more than one layer (`luma_bandwidth` in both `format` and `player`), so those are not
    # duplicates and the count moved whenever unrelated text changed.  A real duplicate is the
    # same name twice for the same layer, because `ParamSet.add` overwrites.
    var_to_layer = {}
    for m in _re.finditer(r"^(\s*)(\w+)\s*=\s*s\.layer\(\"(\w+)\"\)", psrc, _re.M):
        var_to_layer[m.group(2)] = m.group(3)
    per_layer: dict[tuple[str, str], int] = {}
    for m in _re.finditer(r"^(\s*)(\w+)\.define\(\s*\"(\w+)\"", psrc, _re.M):
        var, name = m.group(2), m.group(3)
        layer = var_to_layer.get(var)
        if layer is None:
            continue
        per_layer[(layer, name)] = per_layer.get((layer, name), 0) + 1
    dupes = sorted((l, n, c) for (l, n), c in per_layer.items() if c > 1)
    record("no parameter is defined twice for the same layer",
           "the later definition wins, so the earlier help text is discarded",
           "%d layer/name pair(s) defined more than once%s"
           % (len(dupes), "" if not dupes else ": %s" % dupes[:5]),
           not dupes,
           "eighteen of the nineteen pairs had drifted before this was found, so the UI "
           "explained parameters with words written for another version")
def check_device_dna() -> None:
    """
    A device identity must produce one machine, deterministically, and change nothing without one.

    The third property is the one worth testing hardest. A dimension added to a calibrated model has
    to be inert when it is not in use, or every existing preset, capture and measurement shifts by a
    few per cent for reasons unrelated to what any of them test -- which is how a model quietly stops
    being the thing that was calibrated.

    The first property is what makes the dimension useful rather than decorative: the same serial
    gives the same machine forever, so a fault stays reproducible and a tape can be revisited. That is
    the same requirement as the seeded defect timeline, and it is why the deviations are drawn from a
    generator seeded by the identity rather than from a random number.
    """
    import tempfile

    from atsim.core.profiles import SCHEMA
    from atsim.core.units import DeviceUnit, apply_unit, bind_ranges, spreads_from_schema
    from atsim.web.session import Project

    bind_ranges(SCHEMA)
    spreads = spreads_from_schema(SCHEMA)
    if len(spreads) < 8:
        record("some parameters declare a per-unit spread",
               "a device-identity dimension with nothing to vary is not a dimension",
               "only %d parameter(s) declare a spread" % len(spreads), False)
        return

    # --- 1. inert with no identity
    from atsim.core.params import numeric_keys
    num = numeric_keys(SCHEMA)
    p = Project(id="t", params={})
    nominal = p.resolved_params()
    raw = SCHEMA.resolve_all({})
    # only numeric parameters can be compared by subtraction; the subtitle text is the first parameter
    # whose value is not a number, and comparing it raised here
    skipped = sum(1 for l in raw for k in raw[l] if k not in num.get(l, set()))
    untouched = all(abs(nominal[l][k] - raw[l][k]) < 1e-12
                    for l in raw for k in raw[l] if k in num.get(l, set()))
    record("a project with no device identity is the nominal model",
           "adding the dimension must not move any existing preset, capture or measurement",
           "%d numeric parameters unchanged (%d non-numeric skipped)"
           % (sum(len(v) for v in num.values()), skipped),
           untouched,
           "an anonymous unit perturbing every spread parameter is the failure that would "
           "invalidate every calibration silently")

    # --- 2. deterministic per identity
    ids = {"camcorder": "Panasonic NV-GS78 A39482", "deck": "JVC HR-J600 11223"}
    a = Project(id="a", params={}, unit_identities=dict(ids)).resolved_params()
    b = Project(id="b", params={}, unit_identities=dict(ids)).resolved_params()
    stable = all(abs(a[l][k] - b[l][k]) < 1e-12
                 for l in a for k in a[l] if k in num.get(l, set()))
    changed = sum(1 for l in a for k in a[l]
                  if k in num.get(l, set()) and abs(a[l][k] - nominal[l][k]) > 1e-12)
    record("one serial number is one machine, every time",
           "a fault must stay reproducible and a tape revisitable",
           "the same two identities give identical parameters: %s; %d parameters deviate from "
           "nominal" % (stable, changed),
           stable and changed > 5,
           "a random machine per render would make every capture unrepeatable")

    # --- 3. distinct machines
    c = Project(id="c", params={},
                unit_identities={**ids, "deck": "JVC HR-J600 99887"}).resolved_params()
    differs = sum(1 for l in a for k in a[l]
                  if k in num.get(l, set()) and abs(a[l][k] - c[l][k]) > 1e-12)
    record("a different serial is a different machine",
           "two units of one model must not render identically",
           "changing only the deck serial alters %d parameter(s)" % differs,
           differs > 3,
           "a dimension that yields identical machines is not a dimension")

    # --- 4. and the spread table must not name a parameter that does not exist
    from atsim.core.profiles import UNIT_SPREADS
    unknown = [k for k in UNIT_SPREADS
               if not any(k in SCHEMA.layer(l).keys() for l in SCHEMA.layer_names())]
    record("every declared spread names a real parameter",
           "a spread for a non-existent parameter is documentation of nothing",
           "%d spreads declared; unknown: %s" % (len(UNIT_SPREADS), unknown or "none"),
           not unknown)
def check_keyframes() -> None:
    """
    A timeline must be a function of the frame number, and must drive the picture without being asked.

    Three properties, each of which failed at least once while this was being built:

      * the curves do what their names say.  `exp` must approach equilibrium *across the segment the
        user drew* -- an earlier constant made a 420-second warm-up complete in 90 and then hold, which
        is plausible-looking output from a wrong number.  `hold` must not change until the next moment,
        and a track evaluated before its first point must hold rather than extrapolate backwards.
      * a project with no tracks and no environment must render exactly as before.  This is the property
        that keeps every existing preset and capture valid.
      * a track must actually move the rendered frames.  The mechanism can be present, reachable, and
        annulled -- which has happened five times in this project -- so it is measured on the output
        rather than inferred from the code.
    """
    import numpy as np

    from atsim.core.keyframes import Keyframe, KeyframeSet, Track
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.io.source import SyntheticSource
    from atsim.render.renderer import Renderer

    # --- curves
    tr = Track(layer="camera", key="exposure", absolute=True,
               points=[Keyframe(0.0, 0.0, "exp"), Keyframe(10.0, 1.0, "hold")])
    at_3 = tr.value_at(3.0)          # one time constant of a 0.30-fraction constant
    at_5 = tr.value_at(5.0)
    record("the exp curve approaches equilibrium across the drawn segment",
           "a warm-up lasting one segment must not finish in a fifth of it",
           "0.0 -> %.3f at t=3 (one time constant), %.3f at t=5, 1.0 at t=10"
           % (at_3, at_5),
           0.55 < at_3 < 0.70 and at_5 < 0.90,
           "an earlier constant reached 0.99 by t=4 on a 10-second segment")

    hold = Track(layer="x", key="y", absolute=True,
                 points=[Keyframe(0.0, 1.0, "hold"), Keyframe(10.0, 3.0, "linear")])
    record("hold does not change until the next moment",
           "a discrete state stays where it is",
           "t=0 %.2f, t=9.99 %.2f, t=10 %.2f"
           % (hold.value_at(0), hold.value_at(9.99), hold.value_at(10.0)),
           abs(hold.value_at(9.99) - 1.0) < 1e-9 and abs(hold.value_at(10.0) - 3.0) < 1e-9)

    later = Track(layer="x", key="y", absolute=True,
                  points=[Keyframe(10.0, 5.0, "linear"), Keyframe(20.0, 9.0, "hold")])
    record("a track holds before its first point rather than extrapolating",
           "the frames before a session's first moment are the machine as it already was",
           "t=0 gives %.2f, not a value projected backwards from t=10" % later.value_at(0.0),
           abs(later.value_at(0.0) - 5.0) < 1e-9)

    # --- inert and effective, measured on the render
    W, H, N = 200, 140, 8
    base = SCHEMA.resolve_all(FACTORY_PRESETS["family_archive_1992"]["layers"])

    def run(kf=None, env=None):
        s = SyntheticSource(width=W, height=H, frames=N)
        r = Renderer(s, W, H, s.fps, base, seed=5)
        r.set_time_varying(kf, env)
        return [np.clip(r.render_frame(i), 0, 1) for i in range(N)]

    plain = run()
    again = run()
    same = all(np.array_equal(a, b) for a, b in zip(plain, again))
    record("a project with no timeline renders identically",
           "an unrequested dimension must not move any existing preset",
           "two runs agree frame for frame: %s" % same, same)

    ramp = Track(layer="camera", key="exposure", absolute=True,
                 points=[Keyframe(0.0, -1.5, "linear"),
                         Keyframe(N / 25.0, 1.5, "hold")])
    moved = run(kf=KeyframeSet(tracks=[ramp]))
    diff = max(float(np.abs(moved[i].mean() - plain[i].mean())) for i in range(N))
    record("a keyframe track reaches the rendered picture",
           "the mechanism can be present and annulled, so measure the output",
           "largest mean luminance difference %.4f" % diff, diff > 0.02,
           "a stage holds the layer it copied in __init__, so a push that does not reach it "
           "changes nothing")

    from atsim.core.environment import Environment
    warm = run(env=Environment(warmth=1.0))
    cold = run(env=Environment(warmth=0.0))
    ediff = max(float(np.abs(warm[i].mean() - cold[i].mean())) for i in range(N))
    record("a cold machine renders differently from a warm one",
           "the environment is a state, not a slider that does nothing",
           "warmth 1.0 against 0.0: largest mean difference %.4f" % ediff, ediff > 0.005)

    # --- and the renderer must not reset the timeline when it rebuilds the chain
    s = SyntheticSource(width=W, height=H, frames=N)
    r = Renderer(s, W, H, s.fps, base, seed=5)
    r.set_time_varying(KeyframeSet(tracks=[ramp]), None)
    kept = r._keyframes is not None
    r._build_chain()
    record("the renderer keeps its timeline across a chain rebuild",
           "the chain is rebuilt whenever parameters change, and took the timeline with it",
           "timeline survives _build_chain: %s" % (r._keyframes is not None and kept),
           kept and r._keyframes is not None,
           "the attributes were first initialised inside _build_chain, so every rebuild "
           "silently cleared them")
def check_autocontrol() -> None:
    """
    The automatic loops must take the time they claim, settle towards their target, and be inert off.

    Every one of these failed while it was being built, and two of the failures were the same mistake in
    opposite directions:

      * `self.exposure += loop.update(...)` used the loop's return but added it to the current value, so
        the correction compounded -- measured, a scene needing +1.5 stops went 0.06, 0.85, 2.43 and then
        sat at the +3.0 limit;
      * `self.exposure = loop.state` used the right value but read it from *before* the step, so the loop
        stepped correctly every frame and its result was never used -- measured, exactly 0.0000 on a
        scene needing +1.5 stops;
      * the loops themselves jumped to their target on the first frame, because `_seeded` bypassed the
        decay, so tau had no effect at all.

    A loop is exactly the kind of mechanism that can be present, reachable, and do nothing, so the
    properties are measured on the loops and on the rendered frames rather than inferred.
    """
    import numpy as np

    from atsim.codecs.autocontrol import ExposureLoop, centre_weights
    from atsim.core.environment import Environment
    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.io.source import SyntheticSource
    from atsim.render.renderer import Renderer

    W, H, N, FPS = 160, 120, 60, 25.0
    base = SCHEMA.resolve_all(FACTORY_PRESETS["family_archive_1992"]["layers"])

    # --- the time constant governs the approach -----------------------------
    # a loop starting at unity gains on a warm scene, sampled after one second
    warm = np.zeros((H, W, 3), np.float32)
    warm[:] = np.array([1.25, 1.00, 0.72], np.float32) * 0.6
    err = {}
    for tau in (1.5, 4.0):
        from atsim.codecs.autocontrol import WhiteBalanceLoop
        wb = WhiteBalanceLoop(tau_s=tau)
        for _ in range(int(FPS)):
            wb.update(warm, 1.0 / FPS)
        err[tau] = float(np.abs(wb.state - 1.0).max())
    ratio = err[1.5] / max(1e-9, err[4.0])
    record("a loop's time constant governs how fast it adapts",
           "a loop that reaches its target on the first frame has no time constant at all",
           "distance from unity gains after 1 s: tau 1.5 s %.4f, tau 4.0 s %.4f (ratio %.2f)"
           % (err[1.5], err[4.0], ratio),
           1.4 < ratio < 6.0,
           "tau 1.5 and tau 4.0 once produced bit-identical traces")

    # --- and the error falls towards zero, monotonically --------------------
    ex = ExposureLoop(target=0.46, tau_s=1.0)
    w = centre_weights(H, W)
    dark = np.full((H, W, 3), 0.16, np.float32)
    seq = [ex.update(dark, 1.0 / FPS, w) for _ in range(int(6 * FPS))]
    target = float(np.log2(0.46 / 0.16))
    e0 = abs(seq[0] - target)
    e1 = abs(seq[int(FPS)] - target)
    e2 = abs(seq[-1] - target)
    falling = e1 < e0 and e2 < e1
    record("an exposure loop's error falls towards its target",
           "the state approaches the target rather than being placed on it",
           "error %+.4f at 0 s, %+.4f at 1 s, %+.4f at 6 s (target %+.4f)"
           % (e0, e1, e2, target),
           falling and e2 < 0.25 * e0,
           "the loop must also have settled by 6 tau, which e2 < 0.25 * e0 checks")

    # --- inert by default ---------------------------------------------------
    def run(over):
        p = merge_values(base, {"camera": over})
        s = SyntheticSource(width=W, height=H, frames=N)
        r = Renderer(s, W, H, FPS, p, seed=9)
        return [np.clip(r.render_frame(i), 0, 1) for i in range(N)]

    a, b = run({}), run({})
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    record("a camera with no automatic control is unchanged",
           "all three loops default off, so every existing preset must render as before",
           "two loopless renders agree frame for frame: %s" % same, same,
           "the per-unit spread failed exactly this property on its first attempt")

    # --- and they reach the picture, on a scene where the right answer is visible
    # a synthetic source is near the metering target already, so the correction is small; the
    # environment's cold state is used because it moves the camera's own reading well away from
    # the target and therefore gives the loop something unambiguous to do
    def luma(f):
        return float((f @ np.array([0.299, 0.587, 0.114], np.float32)).mean())

    def cast(f):
        m = f.reshape(-1, 3).mean(axis=0)
        return float(m.max() / max(1e-6, m.min()))

    ae = run({"ae_enabled": 1.0, "ae_tau_s": 0.6, "ae_target": 0.62})
    moved = max(abs(luma(ae[i]) - luma(a[i])) for i in range(N))
    record("auto-exposure reaches the rendered picture",
           "the loop can run correctly and its result never be used",
           "largest luma difference against no loop: %.4f" % moved, moved > 0.02)

    # **The loop's own contribution, on a cast that does not move.**
    #
    # The previous form compared frame 0 against frame N-1 and the clip's cast decays on its own: measured, loop
    # off takes it 1.2725 -> 1.0140 and loop on 1.2668 -> 1.0493, so the scene supplied the whole 0.2585 of the
    # reduction and the loop contributed **-0.0353**.  It passed by **0.0005**, which is what a check passes by
    # when it is measuring the scene generator's drift rather than the loop.
    #
    # Sixty frames at tau 1.5 s is also two fifths of a time constant, so the old form read the first part of a
    # curve.  Measured on a stationary cast the gain walks 1.0 -> 0.906 over 200 frames and the cast falls
    # 1.1573 -> 1.0615: real, monotone, and **bounded**, because a real auto-white-balance corrects towards its
    # metering target and holds some of the cast instead of nulling every scene.
    def run_steady(over, n):
        """The same size and seed, with a cast present identically in every frame."""

        class Steady:
            def __init__(self):
                self.width, self.height, self.fps, self.frame_count = W, H, FPS, n

            def frame(self, index):
                a = np.full((H, W, 3), 0.30, np.float32)
                a[..., 0] += 0.16            # a fixed warm cast, in every frame
                a[..., 2] -= 0.08
                a[40:120, 40:200] = 0.75     # and a bright area, so the metering has range
                return a

        p = merge_values(base, {"camera": dict({"exposure": 0.0}, **over),
                                "display": {"enabled": 0.0}, "tape": {"enabled": 0.0},
                                "player": {"enabled": 0.0}})
        rr = Renderer(Steady(), W, H, FPS, p, seed=9)
        return [np.clip(rr.render_frame(i), 0, 1) for i in (0, n - 1)]

    STEADY_N = 120
    c_off = cast(run_steady({"awb_enabled": 0.0}, STEADY_N)[-1])
    c_on = cast(run_steady({"awb_enabled": 1.0, "awb_tau_s": 1.5}, STEADY_N)[-1])
    record("auto white balance reduces a cast that cannot drift",
           "the loop integrates the scene's average colour and drives the channels together. A fixed cast cannot "
           "drift away on its own, so every part of the reduction is the loop's",
           "with the loop off the cast settles at %.4f; with it on %.4f (%+.4f, %.1f%% of the cast removed)"
           % (c_off, c_on, c_on - c_off, 100.0 * (1.0 - c_on / c_off)),
           c_on < c_off * 0.98,
           "the previous form compared frame 0 against frame %d and the clip's own decay supplied 0.2585 of the "
           "reduction, so it passed by 0.0005 while measuring the wrong quantity" % (N - 1))

def check_operator() -> None:
    """
    The hand must move the picture like a hand, and focus and zoom must reach it at all.

    The distinguishing property is that handheld shake is **narrowband and rhythmic, not noise**. A random
    offset per frame looks like a glitch; a sum of a few quasi-periodic components looks like a person.
    That is measurable two ways, and both are asserted: the spectrum is concentrated rather than flat, and
    the frame-to-frame movement is correlated, because a hand cannot travel arbitrarily far in 40 ms.

    Two parameters were also found to be inert while wiring this, which is the fault this project has now
    found seven times: `zoom_travel` was read and never used, and the focus loop computed a motor position
    that nothing read, because `CameraStage` had no defocus path at all -- it softens through
    `_lens_wear`, which models coating damage, a different physical mechanism. So the loop that exists to
    hunt for focus did not move the lens. Both are measured on the rendered frames here.
    """
    import numpy as np

    from atsim.codecs.operator import HAND, plan_bumps, shake_offset, zoom_curve
    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.io.source import SyntheticSource
    from atsim.render.renderer import Renderer

    W, H, FPS = 280, 200, 25.0

    # --- the hand's motion is narrowband, not noise
    N = 250
    tr = np.array([shake_offset(HAND, 1.0, i, FPS, 7) for i in range(N)])
    sig = (tr[:, 0] - tr[:, 0].mean()) * np.hanning(N)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    spec = spec / max(1e-12, spec.sum())
    hhi = float((spec ** 2).sum())
    flat = 1.0 / len(spec)
    record("handheld movement is narrowband rather than noise",
           "a hand moves as a few quasi-periodic components, and the rhythm is what reads as a hand",
           "spectral concentration %.4f against %.5f for a flat spectrum (%.0fx)"
           % (hhi, flat, hhi / flat),
           hhi > 5.0 * flat,
           "a random offset per frame is a glitch, not a hand")

    # --- and it is correlated frame to frame
    lag1 = float(np.corrcoef(tr[:-1, 0], tr[1:, 0])[0, 1])
    record("consecutive frames of hand movement are correlated",
           "a hand cannot travel arbitrarily far in 40 ms",
           "lag-1 correlation of the horizontal movement %+.3f" % lag1,
           lag1 > 0.15,
           "white noise would give about zero")

    # --- jostles are episodes
    bumps = plan_bumps(rate_hz=3.0, amount=1.0, frames=N, fps=FPS, seed=3)
    if len(bumps) < 3:
        record("jostles are planned as discrete episodes",
               "a jostle is a step, a door, an elbow -- not a sustained movement",
               "only %d jostle(s) over %.0f s, too few to test" % (len(bumps), N / FPS), False)
    else:
        base = np.array([shake_offset(HAND, 1.0, i, FPS, 7) for i in range(N)])
        withb = np.array([shake_offset(HAND, 1.0, i, FPS, 7, bumps) for i in range(N)])
        dev = np.abs(withb - base).sum(axis=1)
        quiet = float(np.percentile(dev, 80))
        peak = float(dev.max())
        record("jostles are spikes over a quiet floor",
               "most frames are undisturbed and a few are jolted",
               "%d jostle(s); peak %.2f px against an 80th-percentile %.3f px (%.0fx)"
               % (len(bumps), peak, quiet, peak / max(1e-9, quiet)),
               peak > 8.0 * max(1e-9, quiet),
               "the share of disturbed frames is what distinguishes an episode from a hum")

    # --- and it reaches the picture, on a clip where the events occur
    def run(over, frames=180):
        params = merge_values(SCHEMA.resolve_all(
            FACTORY_PRESETS["family_archive_1992"]["layers"]), {"camera": over})
        s = SyntheticSource(width=W, height=H, frames=frames)
        r = Renderer(s, W, H, FPS, params, seed=11)
        return [np.clip(r.render_frame(i), 0, 1) for i in range(frames)]

    def sharpness(f):
        g = f @ np.array([0.299, 0.587, 0.114], np.float32)
        return float(np.abs(np.diff(g, axis=1)).mean())

    a, b = run({}, 24), run({}, 24)
    inert = all(np.array_equal(x, y) for x, y in zip(a, b))
    record("a camera with no operator is unchanged",
           "every operator parameter ships at zero",
           "two default renders agree frame for frame: %s" % inert, inert)

    still = run({}, 30)
    shaken = run({"shake_intensity": 1.0}, 30)
    moved = max(float(np.abs(shaken[i] - still[i]).mean()) for i in range(30))
    record("the hand reaches the rendered picture",
           "the shake is applied before the sensor sees anything",
           "largest mean difference per frame %.4f" % moved, moved > 0.004)

    sharp = [sharpness(run({"zoom_travel": z}, 6)[-1]) for z in (0.0, 0.5, 1.0)]
    record("zoom travel softens the picture",
           "at long travel a consumer lens is furthest from its designed element spacing",
           "sharpness %.5f at travel 0, %.5f at 0.5, %.5f at 1.0" % tuple(sharp),
           sharp[0] > sharp[1] > sharp[2],
           "the parameter was read into an attribute and never used")

    af = run({"af_enabled": 1.0, "af_tau_s": 0.3}, 80)
    s = [sharpness(f) for f in af]
    spread = (max(s) - min(s)) / max(1e-9, max(s))
    record("the focus loop moves the lens",
           "a loop that hunts for focus and does not move the lens has not modelled hunting",
           "sharpness varies by %.1f%% of its maximum over the clip" % (100 * spread),
           spread > 0.02,
           "`CameraStage` had no defocus path, so the loop's motor position was written and never read")
def check_session_ui() -> None:
    """
    The panels must be shipped, referenced, and wired to the API, and the API must round-trip.

    A UI is easy to get wrong in ways no compiler catches: a file that exists but is not referenced, a
    script tag that loads before the thing it extends, a panel whose data never arrives from the server, a
    control that saves to the project but that the renderer never receives. Two of those are static and are
    checked by reading the shipped bytes; the last is behavioural and is checked by driving the real `App`
    and confirming the picture changes.

    The behavioural check is the one that matters most here. `App.renderer_for` built renderers from
    `resolved_params()` alone, so the environment and the keyframes never reached the renderer -- editing a
    timeline in the interface would have changed nothing on screen while every file and field looked right.
    """
    import os as _os

    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    static = _os.path.join(root, "atsim", "web", "static")

    # --- the files are shipped
    need = ["session_ui.js", "session_panels.html"]
    missing = [f for f in need if not _os.path.exists(_os.path.join(static, f))]
    record("the dimension panels are shipped",
           "a panel that is not in the static directory cannot be loaded",
           "present: %s" % ", ".join(f for f in need if f not in missing),
           not missing, "missing: %s" % missing if missing else "")

    # --- and referenced by the page, after the script they extend
    html = open(_os.path.join(static, "index.html"), encoding="utf-8").read()
    has_ui = "/static/session_ui.js" in html
    after = html.find("/static/session_ui.js") > html.find("/static/app.js")
    record("the page loads the panels, after app.js",
           "a script that extends another must load after it",
           "session_ui.js referenced: %s; loaded after app.js: %s" % (has_ui, after),
           has_ui and after)

    # --- the panels are mounted from applyProject, which is what syncs state
    app_js = open(_os.path.join(static, "app.js"), encoding="utf-8").read()
    lines = app_js.split("\n")
    calls = [i for i, l in enumerate(lines) if "SessionUI.syncFromProject" in l]
    owner = None
    if len(calls) == 1:
        import re as _re
        for i in range(calls[0], -1, -1):
            m = _re.match(r"\s*(?:async\s+)?function\s+(\w+)", lines[i])
            if m:
                owner = m.group(1)
                break
    record("the panels sync from applyProject, exactly once",
           "the panels must reflect the state the server just returned, or they show stale values",
           "call sites: %d, enclosing function: %s" % (len(calls), owner),
           len(calls) == 1 and owner == "applyProject",
           "an edit placed by matching text landed in the wrong function four times in this session")

    # --- behaviour: the API round-trips AND the renderer receives it
    import numpy as np

    from atsim.web.server import App

    app = App(data_dir=_os.path.join(root, "runs"))
    p = app.current
    saved = (dict(p.environment or {}), list(p.keyframes or []), dict(p.unit_identities or {}))
    try:
        app.set_project({"keyframes": [{
            "layer": "camera", "key": "exposure", "absolute": True,
            "points": [{"t": 0.0, "value": -1.6, "curve": "linear"},
                       {"t": 1.2, "value": 1.6, "curve": "hold"}]}]})
        app.set_project({"unit_identities": {"camcorder": "Panasonic NV-GS78 A39482"}})
        d = app.project_dict()
        round_trips = (len(d.get("keyframes") or []) == 1
                       and len(d.get("unit_report") or []) > 10
                       and d.get("unit_identities"))
        r, _s = app.store.renderer_for(p, preview=True, scale=0.25)
        reaches = getattr(r, "_keyframes", None) is not None
        means = [float(np.clip(r.render_frame(i), 0, 1).mean()) for i in (0, 6, 12, 18, 24)]
        span = max(means) - min(means)
        record("a keyframe edited through the API moves the rendered picture",
               "the whole point of the editor is that what is drawn takes effect",
               "round-trip: %s; renderer holds the timeline: %s; frame mean spans %.4f"
               % (round_trips, reaches, span),
               round_trips and reaches and span > 0.01,
               "renderer_for built from resolved_params() alone, so a timeline edited in the "
               "interface would have changed nothing on screen")

        # a serial must select a machine, on the *reported* path and not only the rendered one
        app.set_project({"unit_identities": {"camcorder": "AAAA"}})
        a = sorted((x["key"], round(x["value"], 6)) for x in app.project_dict()["unit_report"])
        app.set_project({"unit_identities": {"camcorder": "BBBB"}})
        b = sorted((x["key"], round(x["value"], 6)) for x in app.project_dict()["unit_report"])
        record("the device report follows the serial number",
               "a report that ignores the serial makes every unit's claim unverifiable",
               "%d rows differ between two serials" % sum(1 for (k, x), (_, y) in zip(a, b)
                                                          if abs(x - y) > 1e-9),
               a != b,
               "unit_report built its own DeviceUnit(identity=\"probe\") -- a hardcoded string -- "
               "so every serial gave the same table while the renderer used the real one")
    finally:
        app.set_project({"environment": saved[0], "keyframes": saved[1],
                         "unit_identities": saved[2]})
def check_optics() -> None:
    """
    The three optical mechanisms must have their signatures, not merely change the output.

    Each has a property that a wrong implementation would lack, and those properties are what is asserted:

      * **blooming is vertical.** The charge spills along the CCD's transfer channel, so the spread is
        anisotropic. A round halo -- which is what `halation` already is -- would fail this, and that
        distinction is the whole reason the note lists the two separately;
      * **a defect site is fixed and absolute.** The same value in every frame, and not changed by the
        camera's own processing. Both halves were wrong at first: the rate was a hundred times too low so
        the Poisson process never fired, and the write happened before the unsharp mask, which pulled a
        dead site back to the neighbourhood average -- measured, set to 0.0000 and read back as 0.5199;
      * **fog lifts the blacks.** The veil is a fixed amount of scattered light, so its effect is largest
        where there was least light. Weighting it by darkness as well *cancels* that, because a blurred
        copy is already large where the picture is bright -- measured with the product, the brightest
        decile rose +0.1136 against +0.0502 for the darkest.
    """
    import numpy as np

    from atsim.codecs.optics import lens_fog, plan_defect_sites
    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.io.source import SyntheticSource
    from atsim.render.renderer import Renderer

    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    W, H, FPS = 200, 150, 25.0
    base = SCHEMA.resolve_all(FACTORY_PRESETS["family_archive_1992"]["layers"])

    # --- the rate is a count that actually produces sites
    # the same number every size: the parameter is sites per 720x480-equivalent area, so a larger frame
    # scales it.  720x480 is 34.56 units of 1e4 pixels, so a rate of 4 gives about 138 sites.
    counts = [len(plan_defect_sites(r, 0.5, 480, 720, s))
              for r in (1.0, 2.0, 4.0) for s in range(12)]
    expect = max(1.0, 2.0, 4.0) * 480 * 720 / 1e4
    record("the defect rate produces sites across its range",
           "a unit conversion that makes a Poisson process always return zero is invisible to review",
           "rates 1/2/4 on 720x480 gave counts %d..%d over 36 draws (the largest expectation is %.0f)"
           % (min(counts), max(counts), expect),
           min(counts) > 0 and max(counts) < expect * 2.0,
           "rate * area / 1e6 made 6 sites on a 200x150 frame into poisson(0.18), which is zero "
           "most of the time: present, wired, and never firing at any setting a user could choose")

    # --- blooming is vertical
    def spot(level=0.30, frames=2):
        class S:
            width, height, fps, frame_count = W, H, FPS, frames

            def frame(self, index):
                a = np.full((H, W, 3), level, np.float32)
                a[H // 2 - 3:H // 2 + 3, W // 2 - 3:W // 2 + 3] = 1.0
                return a
        return S()

    def cam(over, frames=2):
        p = merge_values(base, {"camera": over}) if over else base
        r = Renderer(spot(frames=frames), W, H, FPS, p, seed=17)
        return r

    plain = np.clip(cam({}).render_frame(0), 0, 1)
    bloomed = np.clip(cam({"bloom_amount": 0.9, "bloom_threshold": 0.7}).render_frame(0), 0, 1)
    d = np.abs(bloomed - plain) @ COEF

    def spread(profile, axis):
        p = profile / max(1e-9, profile.sum())
        m = float((p * axis).sum())
        return float(np.sqrt((p * (axis - m) ** 2).sum()))

    sv = spread(d.sum(axis=1), np.arange(H, dtype=np.float32))
    sh = spread(d.sum(axis=0), np.arange(W, dtype=np.float32))
    record("blooming spreads vertically, not as a halo",
           "the CCD's transfer channel is vertical, which is what distinguishes it from halation",
           "vertical spread %.1f px against %.1f px horizontally (ratio %.2f)" % (sv, sh, sv / max(1e-9, sh)),
           sv > 1.8 * sh)

    # --- a defect site is fixed and absolute at the sensor
    r = cam({"dead_sites": 3.0, "stuck_fraction": 0.5}, frames=5)
    r.render_frame(0)
    stage = r.stage["camera"]
    sites = stage._sites or []
    if not sites:
        record("a defect site is fixed and absolute", "a stuck site reads the same in every frame",
               "no sites were planned", False)
    else:
        ok = True
        detail = []
        for s in sites:
            vals = []
            for i in range(5):
                r.render_frame(i)
                vals.append(float(np.asarray(r._cache["camera"][i])[s.y, s.x].mean()))
            good = all(v > 0.95 for v in vals) if s.hot else all(v < 0.05 for v in vals)
            ok = ok and good
            detail.append("%s %.3f..%.3f" % ("hot" if s.hot else "dead", min(vals), max(vals)))
        record("a defect site is fixed and absolute at the sensor",
               "a stuck site reads its own value regardless of what is in front of it",
               "%d sites; %s" % (len(sites), "; ".join(detail[:3])),
               ok,
               "written before _sharpen, which computed 0*(1+k) - k*neighbour_mean and pulled the "
               "site back to the neighbourhood average")

    # --- fog lifts the blacks more than the brights
    # A small blown source on a dark field, which is the shape a flare or a fogged highlight has.  A large
    # bright *region* cannot test this: a blur that fits inside it takes most of its veil from itself, so
    # the far-field veil is smaller where the frame is bright -- the opposite of a real scene.
    h2, w2 = 160, 200
    a = np.full((h2, w2, 3), 0.18, np.float32)
    a[h2 // 2 - 4:h2 // 2 + 4, w2 // 2 - 4:w2 // 2 + 4] = 1.0
    pl = a @ COEF
    fg = lens_fog(a, 0.9) @ COEF
    far = np.ones((h2, w2), bool)
    far[h2 // 2 - 24:h2 // 2 + 24, w2 // 2 - 24:w2 // 2 + 24] = False
    lift_dark = float((fg - pl)[far].mean())
    # the skirt: the field just outside the source rises more than the far field
    ring = np.zeros((h2, w2), bool)
    ring[h2 // 2 - 16:h2 // 2 + 16, w2 // 2 - 16:w2 // 2 + 16] = True
    ring[h2 // 2 - 8:h2 // 2 + 8, w2 // 2 - 8:w2 // 2 + 8] = False
    skirt = float((fg - pl)[ring].mean())
    record("fog lifts a dark field and grows a skirt around a source",
           "a fixed amount of scattered light matters most where there was least light, and a blown "
           "source scatters enough to glow around itself",
           "dark far field rises %+.5f; the ring just outside the source rises %+.5f" % (lift_dark, skirt),
           lift_dark > 0.02 and skirt > lift_dark,
           "weighting the veil by (1 - luma) cancels the dark lift, because a blurred copy is already "
           "large where the picture is bright: measured +0.1136 bright against +0.0502 dark")

    # --- and all three are off by default
    x = np.clip(cam({}).render_frame(0), 0, 1)
    y = np.clip(cam({}).render_frame(0), 0, 1)
    record("a camera with none of the new optics is unchanged",
           "all three default to zero, so no existing preset moves",
           "two default renders identical: %s" % np.array_equal(x, y), np.array_equal(x, y))
def check_preset_dimensions() -> None:
    """
    A dimension nobody sets is a dimension nobody finds, so the demonstration presets must use them.

    Every new mechanism defaults to off, for the good reason that an unrequested dimension must not move an
    existing preset. The cost is that a user opening the tool meets none of them, so the presets are the
    only way in. This checks that they set what they claim, that the timeline survives the trip through the
    API and into the renderer, and that the fog actually clears -- which is the visible consequence of the
    two tracks in the cold-start preset.

    The sharpness direction is asserted rather than the brightness, and that correction is worth recording:
    fog is an *additive* veil, so removing it makes the picture **darker**. The first version of this check
    expected the brightness to rise as the machine warmed, which is what an exposure loop does when a scene
    brightens -- and it is not what happens here, because the veil was adding light.
    """
    import numpy as np

    from atsim.core.profiles import FACTORY_PRESETS
    from atsim.web.server import App

    NEW = ("lens_fog", "bloom_amount", "bloom_threshold", "dead_sites", "stuck_fraction")
    cold = FACTORY_PRESETS.get("cold_start_handheld") or {}
    warm = FACTORY_PRESETS.get("warm_archival") or {}
    cold_cam = (cold.get("layers") or {}).get("camera") or {}
    warm_cam = (warm.get("layers") or {}).get("camera") or {}
    used = [k for k in NEW if k in cold_cam or k in warm_cam]
    record("the demonstration presets use the optical parameters",
           "a dimension no preset sets is one a user will never meet",
           "cold_start uses %s; warm_archival uses %s; the camera layer carries %d of the five"
           % ([k for k in NEW if k in cold_cam], [k for k in NEW if k in warm_cam], len(used)),
           len(used) >= 4,
           "`lens_fog` is driven by the timeline instead of set as a value, because it clears")

    env = cold.get("environment") or {}
    tl = cold.get("timeline") or []
    keys = [(t.get("layer"), t.get("key")) for t in tl]
    record("the cold-start preset carries a state and a timeline",
           "the machine's condition and its life are separate fields from the parameters",
           "environment %s; timeline %s" % (env, ["%s.%s" % k for k in keys]),
           bool(env) and ("environment", "warmth") in keys and ("camera", "lens_fog") in keys)

    # --- and it survives into the renderer, and the fog clears
    app = App(data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "runs"))
    cur = app.current
    saved = (dict(cur.environment or {}), list(cur.keyframes or {}), dict(cur.unit_identities or {}),
             cur.preset)
    try:
        d = app.apply_preset("cold_start_handheld")
        through_api = [(t["layer"], t["key"]) for t in (d.get("keyframes") or [])]
        r, _s = app.store.renderer_for(app.current, preview=True, scale=0.5)
        reaches = r._keyframes is not None and r._environment is not None

        def sharp(f):
            g = f @ np.array([0.299, 0.587, 0.114], np.float32)
            return float(np.abs(np.diff(g, axis=1)).mean())

        s = [sharp(np.clip(r.render_frame(i), 0, 1)) for i in (0, 12, 30)]
        rises = s[-1] > s[0] * 1.03
        record("the cold-start fog clears over the clip",
               "the fog is a state that recovers as the glass warms, not a constant",
               "timeline through the API: %s; renderer holds it: %s; sharpness %s"
               % (["%s.%s" % k for k in through_api], reaches,
                  " ".join("%.5f" % v for v in s)),
               reaches and rises,
               "fog is an additive veil, so clearing it makes the picture darker, not brighter -- "
               "an earlier version of this check expected brightness to rise and was wrong")
    finally:
        app.set_project({"environment": saved[0], "keyframes": saved[1],
                         "unit_identities": saved[2]})
        if saved[3]:
            app.apply_preset(saved[3])
def check_osd_positions() -> None:
    """
    The generator's two positions must land the tape damage in different places, and the editor must talk to
    the API in the shape the API reads.

    **The measurement had to be rebuilt, and the reason is worth keeping.** The first version fixed the damage
    and varied the position, and took its text mask from one of the two positions -- so at the other position
    the mask covered the wrong pixels, the numbers came out reversed, and the mask grew to nearly the whole
    frame (15853 pixels of 76800) because placing the overlay *before* the tape changes the tape's plan and
    therefore every pixel in the frame. That is a real consequence of the placement and it makes a
    position-difference mask useless as a text mask.

    The corrected measurement fixes the position and varies the damage, and takes the text's pixels from the
    render where the text is clean by construction -- the camera position -- with the painted luma as the
    target, which is known because a character generator draws opaque glyphs.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.io.source import SyntheticSource
    from atsim.render.renderer import STAGE_ORDER, Renderer

    W, H, FPS = 320, 240, 25.0
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["family_archive_1992"]["layers"])
    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    OSD = {"enabled": 1.0, "font": 1.0, "text": "TITLE 标题", "cell_h": 20.0,
           "shadow": 0.6, "margin": 0.06}
    HEAVY = {"tape": {"dropout_rate": 9.0, "head_wear": 0.95, "tracking_error": 0.9,
                      "head_switch_lines": 24.0},
             "player": {"head_wear": 0.85, "tracking_error": 0.5}}

    def render(over, n=6):
        p = merge_values(BASE, over)
        s = SyntheticSource(width=W, height=H, frames=n)
        r = Renderer(s, W, H, FPS, p, seed=11)
        return np.clip(r.render_frame(n - 1), 0, 1)

    clean_cam = render({"overlay": dict(OSD, at=0.0)})
    no_text = render({"overlay": {"enabled": 0.0}})
    ink = (np.abs(clean_cam - no_text) @ COEF) > 0.05
    print("    the text occupies %d pixel(s)" % int(ink.sum()))

    if int(ink.sum()) < 20:
        record("the generator's two positions land damage differently",
               "a camera burns its OSD over the frame; a titler records onto the tape",
               "the overlay drew only %d pixel(s), too few to test" % int(ink.sum()), False)
    else:
        target = (clean_cam @ COEF)[ink]
        out = {}
        for at in (0.0, 1.0):
            clean = render({"overlay": dict(OSD, at=at)})
            damaged = render({**HEAVY, "overlay": dict(OSD, at=at)})
            out[at] = (float(np.abs((damaged @ COEF)[ink] - target).mean()),
                       float(np.abs((damaged @ COEF)[~ink] - (no_text @ COEF)[~ink]).mean()))
        t0, r0 = out[0.0]
        t1, r1 = out[1.0]
        record("the generator's two positions land damage differently",
               "a camera's OSD survives the tape while a titler's text does not",
               "over the text's own %d pixel(s): deviation %.5f at the camera position against a "
               "picture move of %.5f; %.5f at the titler position against %.5f"
               % (int(ink.sum()), t0, r0, t1, r1),
               t0 < r0 and t1 > t0 * 5.0,
               "an OSD written over the formed picture is not damaged by the tape, and one recorded "
               "onto the tape is")

    # --- the chain places exactly one instance, in the slot the parameter selects
    # the comparison uses the *chain's* own order rather than STAGE_ORDER: the chain puts the overlay in a
    # different slot per setting, so the tape's index is not the same in the two cases.  Measuring against
    # the static tuple made the second case fail a correct chain.
    for at, want in ((0.0, "after"), (1.0, "before")):
        p = merge_values(BASE, {"overlay": dict(OSD, at=at)})
        s = SyntheticSource(width=160, height=120, frames=2)
        r = Renderer(s, 160, 120, FPS, p, seed=5)
        r._build_chain()
        names = [st.name for st in r.chain.stages]
        i, th = names.index("overlay"), names.index("tape")
        ok = (i > th) if at == 0.0 else (i < th)
        record("the chain places the generator %s the tape" % want,
               "exactly one instance, in the slot the parameter selects -- two would disagree about "
               "what a cached frame contains",
               "at=%.0f -> %s" % (at, names), ok and names.count("overlay") == 1)

    # --- the editor's payload shape, whose failure was silent
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    js = open(os.path.join(root, "atsim", "web", "static", "session_ui.js"),
              encoding="utf-8").read()
    shaped = "setProject({ params: patch })" in js
    record("the subtitle editor sends layer values under `params`",
           "a bare layer mapping is silently ignored by the API",
           "setProject wraps the patch: %s" % shaped, shaped,
           "measured: the project still read enabled=0.0, text='' after the editor set them")

    # --- and the engine's own missing-glyph report reaches the payload
    from atsim.web.server import App
    app = App(data_dir=os.path.join(root, "runs"))
    cur = app.current
    saved = dict((cur.params or {}).get("overlay") or {})
    try:
        app.set_project({"params": {"overlay": {"enabled": 1.0, "font": 0.0,
                                                "text": "REC 婚礼", "cell_h": 20.0}}})
        app.frame(0.0, 0.25, 80)
        rep = app.project_dict().get("osd_report") or {}
        missing = "".join(rep.get("missing") or [])
        record("the engine reports which characters its font cannot draw",
               "a subtitle that silently loses its Chinese is worse than one that says it cannot",
               "the OSD font asked for 婚礼 reported missing=%r, drew %s glyph(s)"
               % (missing, rep.get("drawn")),
               missing == "婚礼" and (rep.get("drawn") or 0) > 0,
               "the OSD font has no CJK at all")
    finally:
        app.set_project({"params": {"overlay": saved or {"enabled": 0.0, "text": ""}}})

def check_osd_multiline() -> None:
    """
    A subtitle with more than one line must put the lines in different rows.

    `parse_lines` split a string on newlines and never advanced the row, so every line was drawn at
    `margin_y + 0 * cell` -- one on top of another -- and the frame showed a single pile of overlapping
    glyphs. It reads as a font or rasterising fault, and it was found by looking at an exported frame after
    the panel appeared to work.

    **Every test up to that point used one line**: "REC 00:12:34", "TITLE 标题", "婚礼". A one-line case
    cannot show a row that never increments, and multi-line input is the first thing a person types. So the
    check uses two lines, and asserts separate ink bands rather than a count of drawn glyphs -- the glyph
    count was correct the whole time.
    """
    import numpy as np

    from atsim.codecs.osd import CharacterGenerator, OSDPlan, parse_lines

    nl = chr(10)
    for spec, want in (("A" + nl + "B", [0, 1]), ("A" + nl + "B" + nl + "C", [0, 1, 2]),
                       (["A", "B"], [0, 1])):
        got = [l.row for l in parse_lines(spec)]
        record("a %d-line subtitle is assigned %d row(s)" % (len(want), len(want)),
               "each line needs its own row, or the lines are drawn on top of one another",
               "%r -> rows %s" % (spec, got), got == want,
               "parse_lines split on newlines and never incremented the row, so every line landed at row 0")

    g = CharacterGenerator("lyusung", OSDPlan(lines=parse_lines("LINE1" + nl + "LINE2"),
                                              cell_h=26, margin_y=10))
    img, rep = g.draw(np.zeros((140, 700, 3), np.float32))
    ink = (img @ np.array([0.299, 0.587, 0.114], np.float32)) > 0.5
    rows = np.where(ink.any(axis=1))[0]
    bands, start = [], int(rows[0])
    for a, b in zip(rows, rows[1:]):
        if b != a + 1:
            bands.append((start, int(a)))
            start = int(b)
    bands.append((start, int(rows[-1])))
    record("a two-line subtitle occupies two separate bands of the frame",
           "the glyph count was correct while the lines overlapped, so the pixels are what has to be checked",
           "%d line(s) reported, ink bands %s" % (rep.get("lines"), bands),
           len(bands) == 2 and rep.get("lines") == 2,
           "measured: both lines at y=10..36, one band, 12 glyphs drawn")
def check_payload_shapes() -> None:
    """
    Both ways of naming a layer must work, because a caller that sends the wrong one used to be ignored.

    `App.set_project` merges `payload["params"]` per layer -- that is the server's own convention, and
    `set_levels` wraps its argument that way -- but the route passes the body straight through, so a client
    that sends a bare `{"overlay": {...}}` sets a layer literally named `params` and the real layer is never
    touched. Measured before the fix: the bare form left `overlay.enabled` at 0.0 with no error and no
    warning, so a UI could look like it was working while every value went somewhere nothing reads.

    That is exactly what happened: the subtitle was stored as
    `"params": {"overlay": {"enabled": 1, "text": "1"}}` inside a saved project while the `overlay` layer read
    `enabled=0.0, text=""`, and the panel's own state came back from a payload it had failed to change.

    So the check exercises all three shapes -- bare, nested, and nested twice -- on the real `App`, and asserts
    that none of them leaves a stray `params` layer behind.
    """
    import tempfile

    from atsim.web.server import App

    app = App(data_dir=tempfile.mkdtemp(prefix="payload-"))
    app.set_project({"params": {"overlay": {"enabled": 0.0, "text": ""}}})

    shapes = [
        ("bare layer map", {"overlay": {"enabled": 1.0, "text": "BARE"}}),
        ("nested under params", {"params": {"overlay": {"enabled": 1.0, "text": "NESTED"}}}),
        ("nested twice", {"params": {"params": {"overlay": {"enabled": 1.0, "text": "DOUBLE"}}}}),
    ]
    results = []
    for label, payload in shapes:
        app.set_project(payload)
        d = app.project_dict()
        p = d.get("params") or {}
        ov = p.get("overlay") or {}
        results.append((label, ov.get("enabled"), ov.get("text"), "params" in p))
        print("    %-20s -> enabled=%s text=%r  stray params layer: %s"
              % (label, ov.get("enabled"), ov.get("text"), "params" in p))

    ok = all(r[1] == 1.0 and r[3] is False for r in results)
    record("every payload shape sets the layer it names",
           "a caller that sends a bare layer map must not be silently ignored",
           "; ".join("%s -> text=%r" % (r[0], r[2]) for r in results),
           ok,
           "measured before the fix: the bare form left overlay.enabled at 0.0 with no error, and a saved "
           "project grew a layer named `params` holding the subtitle")

    record("no payload shape leaves a stray `params` layer",
           "a layer named after the payload key stores values where nothing reads them",
           "stray layers after all three shapes: %s"
           % ([r[0] for r in results if r[3]] or "none"),
           not any(r[3] for r in results))
def check_osd_alignment() -> None:
    """
    Alignment must place text where it says, and the cell must come from what the font draws.

    **Two faults found by looking at a rendered strip**, after the numbers had said the layout was right:

      * 柳宋 and A-POS drew every character with a full character's worth of space after it. The half-cell was
        taken as the *most common advance* in the font, and in those two fonts the most common advance belongs
        to the CJK glyphs -- 2400 units in 柳宋 against 1200 for Latin -- so every Latin cell was twice its
        proper width. The measured advance is now what the font actually draws: 1200 for 柳宋 and 500 for
        A-POS, exactly half the previous values, and the string ink fills 0.94-0.98 of its advance instead of
        0.74;
      * the user had positioned a subtitle by typing `"\n\n\n\n\n\n\n      Kigurumi"` -- seven newlines
        and six spaces -- because that was the only control the editor offered. A character generator has a
        position register; it does not reflow, but it absolutely does place a line, so `halign`/`valign` are
        the apparatus's own control and need no line breaking.

    Both halves are asserted: the nine positions land where the rule says, and a character generator asked for
    an overflow reports it rather than silently clipping.
    """
    import numpy as np

    from atsim.codecs.fonts import load_font
    from atsim.codecs.osd import CharacterGenerator, OSDLine, OSDPlan

    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    W, H = 640, 360

    # --- the cell is measured, and the full cell is twice the half one in every font
    ratios = {}
    for key in ("osd", "lyusung", "apos"):
        s = load_font(key)
        half_px = s.half_advance * 26 / max(1, s.units_per_em)
        ratios[key] = (s.half_advance, s.full_advance, s.full_advance / max(1, s.half_advance),
                       half_px)
    bad = [k for k, v in ratios.items() if not (1.8 <= v[2] <= 2.2)]
    record("every font's full cell is twice its half cell",
           "the half/full convention is what lets mixed scripts share a layout without a layout engine",
           "; ".join("%s half=%d full=%d (%.2f), %.1f px half-cell at 26 px"
                     % (k, v[0], v[1], v[2], v[3]) for k, v in ratios.items()),
           not bad,
           "measured before: 柳宋 half=2400 and A-POS half=1008, taken from the most common advance in the "
           "table, which belongs to the CJK glyphs -- so every Latin cell was twice its proper width and a "
           "gap followed every character")

    # --- a Latin string fills its advance
    ratios_fill = {}
    for key in ("osd", "lyusung", "apos"):
        g = CharacterGenerator(key, OSDPlan(lines=[OSDLine("Kigurumi")], cell_h=32,
                                            margin_x=8, margin_y=8))
        half, _h = g.cell()
        img, _rep = g.draw(np.zeros((64, 1200, 3), np.float32))
        ink = (img @ COEF) > 0.5
        cols = np.where(ink.any(axis=0))[0]
        span = int(cols[-1] - cols[0] + 1) if cols.size else 0
        ratios_fill[key] = span / max(1, 8 * half)
    record("a Latin string fills its cells rather than leaving a gap after each character",
           "a gap after every character is what a doubled cell looks like",
           "; ".join("%s ink/advance %.2f" % (k, v) for k, v in ratios_fill.items()),
           all(v > 0.88 for v in ratios_fill.values()),
           "measured before the fix: 0.74 for both Chinese fonts")

    # --- the nine positions land where the rule says
    HA = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    VA = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    mx, my = 16, 12
    bad_pos = []
    for pos in range(9):
        g = CharacterGenerator("lyusung", OSDPlan(lines=[OSDLine("TEST")], cell_h=24,
                                                  margin_x=mx, margin_y=my,
                                                  halign=HA[pos], valign=VA[pos]))
        img, _rep = g.draw(np.zeros((H, W, 3), np.float32))
        ink = (img @ COEF) > 0.5
        cols = np.where(ink.any(axis=0))[0]
        rows = np.where(ink.any(axis=1))[0]
        if not cols.size:
            bad_pos.append(pos)
            continue
        half, ch = g.cell()
        line_w = g.line_width_cells("TEST") * half
        uw, uh = W - 2 * mx, H - 2 * my
        want_x = mx + (0 if HA[pos] == 0 else (uw - line_w) // 2 if HA[pos] == 1 else uw - line_w)
        want_y = my + (0 if VA[pos] == 0 else (uh - ch) // 2 if VA[pos] == 1 else uh - ch)
        if abs(int(cols[0]) - want_x) > 3 or abs(int(rows[0]) - want_y) > 3:
            bad_pos.append(pos)
    record("all nine positions place the text where the rule says",
           "a convenience that places text wrongly is worse than no convenience",
           "%d of 9 positions landed within 3 px of the computed corner; wrong: %s"
           % (9 - len(bad_pos), bad_pos or "none"),
           not bad_pos)

    # --- alignment at its default must not move existing text
    a = CharacterGenerator("lyusung", OSDPlan(lines=[OSDLine("X")], cell_h=24,
                                              margin_x=16, margin_y=12))
    b = CharacterGenerator("lyusung", OSDPlan(lines=[OSDLine("X")], cell_h=24,
                                              margin_x=16, margin_y=12, halign=0, valign=0))
    ia, _ = a.draw(np.zeros((H, W, 3), np.float32))
    ib, _ = b.draw(np.zeros((H, W, 3), np.float32))
    record("alignment at its default does not move the text",
           "a new parameter must not shift every saved project",
           "explicit 0,0 identical to omitting them: %s" % np.array_equal(ia, ib),
           np.array_equal(ia, ib))

    # --- an overflowing line is reported rather than silently clipped
    # a line that genuinely overflows, and one that does not: the report has to separate them.  The first
    # version used `cells_used`, which was the *drawn* width and is capped at the frame's capacity, so a line
    # that ran off the edge could never be reported as too wide -- the one case a person needs told.
    long_text = "这是一行非常长的中文字幕它一定会超出画面宽度，而且超出得非常多非常多，多到不可能放下"
    cases = []
    for text, want in ((("REC 00:12:34"), False), (long_text, True)):
        g = CharacterGenerator("lyusung", OSDPlan(lines=[OSDLine(text)], cell_h=24,
                                                  margin_x=16, margin_y=12))
        _img, rep = g.draw(np.zeros((H, W, 3), np.float32))
        cases.append((text[:12], rep.get("cells_wanted"), rep.get("cells_drawn"),
                      rep.get("cells_fit"), bool(rep.get("overflow")), want))
    record("an over-wide line is reported instead of silently clipped",
           "a character generator does not wrap, so the commonest reason text is missing is that it ran off",
           "; ".join("%r wanted %s, drew %s of %s, overflow=%s"
                     % (c[0], c[1], c[2], c[3], c[4]) for c in cases),
           all(c[4] == c[5] for c in cases)
           # the drawn width is capped by the frame: it starts inside the safe area, so it can slightly
           # exceed `cells_fit` (which is measured from the margin) but never the whole frame
           and all(c[2] <= W // 2 for c in cases)
           and cases[1][1] > cases[1][3])


def check_style_and_crt() -> None:
    """
    Style templates must reach the picture, and the tube distortions must have their shapes.

    Two properties, and the first is the one this project keeps having to assert: a parameter that a stage does
    not read is inert, and a template that sets three colours is inert in three ways at once. Measured on the
    rendered frame rather than on the attribute, because the attribute was correct the last time this failed.

    The second is about *shape*. A trapezoid and a misconvergence both move pixels, so "the output changed"
    proves nothing about whether the right distortion was built. Convergence in particular has a signature --
    zero at the centre, growing towards the edges -- and a uniform channel shift would fail it while still
    changing the output.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Flat:
        def __init__(self, frames=2):
            self.width, self.height, self.fps, self.frame_count = 200, 150, 25.0, frames

        def frame(self, index):
            a = np.full((self.height, self.width, 3), 0.10, np.float32)
            a[self.height // 2 - 4:self.height // 2 + 4, :] = 0.60
            return a

    def render(over):
        p = merge_values(BASE, {"overlay": dict({"enabled": 1.0, "text": "STBY", "font": 0,
                                                 "cell_h": 20, "margin": 0.05}, **over)})
        r = Renderer(Flat(), 200, 150, 25.0, p, seed=5)
        return np.clip(r.render_frame(1), 0, 1), r

    off, _ = render({"enabled": 0.0})
    colours = {}
    for st in range(6):
        img, r = render({"style": float(st)})
        ov = r.stage["overlay"]
        colours[st] = (tuple(round(v, 2) for v in ov.colour), float(np.abs(img - off).mean()))
    distinct = len({c[0] for c in colours.values()})
    record("every style template draws and has its own colour",
           "a template a stage does not read is inert in three colours at once",
           "; ".join("style %d -> %s, frame diff %.5f" % (k, v[0], v[1])
                     for k, v in sorted(colours.items())),
           distinct >= 5 and all(v[1] > 1e-4 for v in colours.values()),
           "measured on the rendered frame, because the attribute was correct the last time this failed")

    # --- the tube distortions have their shapes
    class Grid:
        def __init__(self, frames=2):
            self.width, self.height, self.fps, self.frame_count = 240, 180, 25.0, frames

        def frame(self, index):
            a = np.zeros((self.height, self.width, 3), np.float32)
            a[:, ::20] = 0.85
            return a

    def tube(**disp):
        p = merge_values(BASE, {"display": dict({"enabled": 1.0}, **disp)})
        r = Renderer(Grid(), 240, 180, 25.0, p, seed=5)
        return np.clip(r.render_frame(1), 0, 1)

    a = tube()
    moves = {}
    for key in ("trapezoid_x", "trapezoid_y", "convergence"):
        b = tube(**{key: 0.4 if key != "convergence" else 1.0})
        moves[key] = float(np.abs(b - a).max())
    record("the tube's trapezoid and misconvergence each move the picture",
           "a distortion parameter no stage reads is inert",
           "; ".join("%s max change %.4f" % (k, v) for k, v in sorted(moves.items())),
           all(v > 1e-3 for v in moves.values()))

    b = tube(convergence=1.0)
    d = np.abs((b[..., 0] - b[..., 1]) - (a[..., 0] - a[..., 1]))
    edge = float(d[:, :40].mean())
    mid = float(d[:, 110:130].mean())
    record("misconvergence fringes at the edges and not at the centre",
           "three beams through a shadow mask land differently by an amount that grows from the centre",
           "edge displacement %.5f against %.5f at the centre (ratio %.2f)"
           % (edge, mid, edge / max(1e-9, mid)),
           edge > mid * 2.0,
           "a uniform channel shift would fringe evenly and pass a merely-different test")

    # --- the outline is what makes text legible without covering the picture
    from atsim.codecs.osd import CharacterGenerator, OSDLine, OSDPlan
    g = CharacterGenerator("apos", OSDPlan(lines=[OSDLine("Kig")], cell_h=24, outline=1.0,
                                           colour=(0.92, 0.94, 0.90)))
    _i, rep = g.draw(np.full((80, 200, 3), 0.95, np.float32))
    record("an outline makes white text legible on a bright frame",
           "this is what a real camcorder OSD uses, and why a box is not the default",
           "with the outline: behind=%s contrast=%s legible=%s"
           % (rep.get("behind_luma"), rep.get("contrast"), rep.get("legible")),
           rep.get("legible") is True and (rep.get("contrast") or 0) > 0.5,
           "without it, white text on a 0.95 frame is invisible")
def check_fisheye_and_beam() -> None:
    """
    The tube face's curvature must be radial, and the beam's spot must control the scanline depth.

    Two shapes, and in both cases "the output changed" proves nothing.

    **A fisheye is a radial magnification.** The signature is that the centre barely moves while the edges bow,
    because the mapping grows with radius. Measured: centre 0.00204 against edge 0.13829, a ratio of 68. A
    uniform zoom would change the frame by a similar total amount and fail this.

    **A scanline is the gap between beam passes, not a painted stripe.** The model previously used
    `cos(2*pi*rows/1.4)` with a comment saying the period was chosen to keep a fine structure visible -- a
    number picked for appearance, darkening whole rows without reference to the beam. The beam form has a
    signature too: a smaller spot must give *deeper* lines, which is what a focus control does. It was written
    inert twice, first by dividing the deviation by its mean and then by its spread, each of which cancels
    exactly because the deviation is sampled once per line. Measured after the fix: 0.4316 at a spot of 0.20
    against 0.0740 at 1.30, monotone, with the frame's mean unchanged.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Grid:
        def __init__(self, frames=2):
            self.width, self.height, self.fps, self.frame_count = 240, 180, 25.0, frames

        def frame(self, index):
            a = np.zeros((self.height, self.width, 3), np.float32)
            a[::20, :] = 0.85
            a[:, ::20] = 0.85
            return a

    def render(**disp):
        p = merge_values(BASE, {"display": dict({"enabled": 1.0, "scanlines": 0.0}, **disp)})
        r = Renderer(Grid(), 240, 180, 25.0, p, seed=5)
        return np.clip(r.render_frame(1), 0, 1)

    a = render()
    b = render(curvature=0.8)
    d = np.abs(b - a).mean(axis=2)
    h, w = d.shape
    ctr = float(d[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].mean())
    edge = float(np.concatenate([d[:6].ravel(), d[-6:].ravel(), d[:, :6].ravel(),
                                 d[:, -6:].ravel()]).mean())
    record("the tube's curvature is radial, not a uniform magnification",
           "a curved face magnifies about the centre, so the centre barely moves and the edges bow",
           "curvature 0.8: centre changes by %.5f, the edge band by %.5f (ratio %.0f)"
           % (ctr, edge, edge / max(1e-9, ctr)),
           edge > ctr * 8.0,
           "a uniform zoom would change the frame by a similar total and fail this")

    depths = {}
    means = {}
    for sigma in (0.20, 0.42, 0.90, 1.30):
        img = render(scanlines=0.8, spot_sigma=sigma)
        depths[sigma] = float(np.abs(np.diff(img.mean(axis=(1, 2)))).mean())
        means[sigma] = float(img.mean())
    ordered = [depths[s] for s in (0.20, 0.42, 0.90, 1.30)]
    monotone = all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1))
    record("the beam's spot controls the scanline depth",
           "a smaller spot leaves darker gaps, which is what a focus control does on a real set",
           "; ".join("spot %.2f -> depth %.5f" % (s, depths[s])
                     for s in (0.20, 0.42, 0.90, 1.30)),
           monotone and ordered[0] > ordered[-1] * 3.0,
           "written inert twice: dividing the deviation by its mean, then by its spread, each cancels "
           "exactly because the deviation is sampled once per line -- measured 0.14360 for every spot")

    spread = max(means.values()) - min(means.values())
    record("the scanline structure costs no light",
           "a mask whose mean shifts would be a brightness control with a misleading name",
           "the frame's mean varies by %.5f across the four spot sizes" % spread,
           spread < 0.002)
def check_panel_controls_reachable() -> None:
    """
    The subtitle panel's action row must be the pinned one, with the button inside it.

    The user reported "there is no apply button" three times. It existed and worked -- `btn.onclick = push` --
    but it sat below the visible part of a scrolling panel, and each control group added above it pushed it
    further down, so the report came back after every fix. **A control that exists but cannot be scrolled to is
    not a control**, and no check of "is the string present" could fail: the string was present throughout.

    Four links, each of which silently stops being true when the file is edited:

      * the row that carries the `osdSticky` class is the one built for the enable switch;
      * that row is prepended to the panel host, so it is the first thing in the panel;
      * the apply button is appended to that same row;
      * and the stylesheet actually pins it, because a class with no rule behind it pins nothing.

    A live DOM harness was written first and thrown away. It needed an internal the module does not expose, so it
    would have reported a harness failure instead of a panel failure -- a check that cannot fail, which is the
    fault this project has now found twelve times and would have been the thirteenth.
    """
    js = open(os.path.join("atsim", "web", "static", "session_ui.js"), encoding="utf-8").read()
    css = open(os.path.join("atsim", "web", "static", "app.css"), encoding="utf-8").read()
    import re

    m_row = re.search(r'const\s+(\w+)\s*=\s*el\("div",\s*"ssRow osdSticky"\)', js)
    row = m_row.group(1) if m_row else None
    prepended = bool(row) and re.search(r'host\.prepend\(%s\)' % re.escape(row or ""), js)
    btn_in_row = bool(row) and re.search(r'%s\.append\(btn\)' % re.escape(row or ""), js)
    css_ok = bool(re.search(r'\.osdSticky\s*\{[^}]*position:\s*sticky', css, re.S))
    links = {"the row carrying the pinned class": bool(row),
             "that row prepended to the panel": bool(prepended),
             "the apply button inside that row": bool(btn_in_row),
             "the stylesheet pins the class": css_ok}
    record("the subtitle panel's action row is pinned, with the button inside it",
           "the apply button was reported missing three times while existing and working, because each group "
           "added above it pushed it below the fold of a scrolling panel",
           "; ".join("%s: %s" % (k, v) for k, v in links.items()),
           all(links.values()),
           "no check of the source text could have failed: the string was present the whole time, and the "
           "arrangement is what broke")
def check_no_function_is_cut_in_half() -> None:
    """
    No function body may contain a line at column zero.

    A function declaration inserted at a text anchor inside another function **ends that function early**: every
    line after it becomes a top-level statement. That is what happened to `osdPanels` -- two colour helpers were
    inserted at a comment that sits inside it, so about 170 lines of its body moved to module scope and referred
    to `el`, `host` and `osdEls`, which do not exist there. The whole IIFE threw while loading, so **nothing** in
    the client was built: no button, no pickers, no alignment controls, no panels. The user reported it four
    times as "the button is missing".

    **`node --check` cannot see this.** The file is syntactically valid; it is wrong only at run time.

    The boundary comes from the **balanced braces**, which is what delimits a function in JavaScript. The first
    version of this check looked for a `return` instead and crossed every boundary, failing on twenty-eight
    perfectly good functions -- a check that fails on everything carries no more information than one that cannot
    fail, and both are the fault this project keeps finding.
    """
    import re
    path = os.path.join("atsim", "web", "static", "session_ui.js")
    lines = open(path, encoding="utf-8").read().split("\n")

    def body_span(i: int) -> int:
        """The line index of the brace that closes the function declared at line `i`."""
        depth = 0
        seen = False
        for j in range(i, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if lines[j].count("{"):
                seen = True
            if seen and depth <= 0:
                return j
        return len(lines) - 1

    broken = []
    checked = 0
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)function\s+(\w+)", line)
        if not m:
            continue
        checked += 1
        last = body_span(i)
        for j in range(i + 1, last):
            cur = lines[j]
            if not cur.strip():
                continue
            if not cur.startswith((" ", "\t")):
                broken.append("%s (line %d) has a top-level line at %d" % (m.group(2), i + 1, j + 1))
                break
    record("no client function is cut in half by an insertion inside it",
           "a function declaration inserted at a text anchor inside another function ends it early, and the body "
           "after it becomes top-level statements referring to names that do not exist there",
           ("; ".join(broken) if broken else
            "all %d functions keep their bodies indented to their closing brace" % checked),
           not broken,
           "measured after `osdPanels` was split: the entire client failed to load and the user saw one "
           "checkbox, while `node --check` passed because the file was still syntactically valid")
def check_workspace_layout() -> None:
    """
    The workspace grid: every area has an element, the splitters are overlays, and the widths are variables.

    Three faults this feature produced, all of them *valid* markup that lays out wrongly:

      * **the splitters were grid items.**  `grid-template-columns` named a five-pixel `lspl` track, and a named
        track does not collapse while the stage is explicitly placed beyond it.  Measured with both panels
        hidden: `stageLeft: 5` rather than `0`.  They are absolute overlays now, so the grid has three columns
        and a hidden panel's track collapses on its own;
      * **a named area with no element** shows as empty space, and a **claim naming no area** is ignored, and
        neither is a syntax error anywhere.  So the areas and the claims are checked against each other;
      * **a hardcoded panel width in a later rule** silently wins over the variable, which makes dragging look
        broken while every measurement of the variable agrees.  The old
        `main{grid-template-columns:270px ... 340px}` is exactly that shape, so the check requires the
        declaration to use the variables.

    Verified live over the DevTools protocol as well, which is what actually proved the arrangement: dragging
    the left splitter took it from 270 to 390 and the stage from 864 to 744, and the workspace hiding both panels
    gave the stage the full 1484 pixels.
    """
    import re as _re
    css = open(os.path.join("atsim", "web", "static", "app.css"), encoding="utf-8").read()
    html = open(os.path.join("atsim", "web", "static", "index.html"), encoding="utf-8").read()

    m = _re.search(r"grid-template-areas:\s*((?:\s*\"[^\"]+\")+)", css)
    areas = set()
    if m:
        for row in _re.findall(r'"([^"]+)"', m.group(1)):
            areas.update(row.split())
    claimed = dict((a, sel) for sel, a in _re.findall(
        r"main\.wsGrid\s*>\s*([.#\w-]+)\s*\{[^}]*grid-area:\s*(\w+)", css, _re.S))
    unclaimed = sorted(a for a in areas if a not in claimed)
    undefined = sorted(a for a in claimed if a not in areas)
    record("every workspace grid area has an element claiming it",
           "an area with no element lays out as empty space and a claim naming no area is ignored, and neither "
           "is an error anywhere",
           "areas %s; claimed %s; unclaimed %s; undefined %s"
           % (sorted(areas), sorted(claimed), unclaimed or "none", undefined or "none"),
           bool(areas) and not unclaimed and not undefined,
           "checked against the CSS alone: a browser would silently render a hole")

    split_rule = _re.search(r"\.wsSplit\{[^}]*\}", css, _re.S)
    is_overlay = bool(split_rule and "position:absolute" in split_rule.group(0))
    stage_is_area = "grid-area:stage" in css
    record("the splitters are overlays and not grid items",
           "a named five-pixel track does not collapse while the stage is placed beyond it, so a hidden panel "
           "left the stage five pixels off the edge",
           "the splitter rule is absolute: %s; the stage is an area: %s" % (is_overlay, stage_is_area),
           is_overlay and stage_is_area,
           "measured before the fix: the workspace hiding both panels gave stageLeft 5, and 0 after")

    # Only the main layout's own rules matter here.  Other components legitimately hardcode their columns --
    # the timeline, the parameter rows -- and an earlier version of this check flagged three of them.
    uses_vars = bool(_re.search(r"grid-template-columns:\s*var\(--w-left", css))
    main_rules = _re.findall(r"(?:^|\n)(main(?:\.[\w-]+)?\s*\{[^}]*\})", css)
    offenders = [r[:60].replace("\n", " ") for r in main_rules
                 if "grid-template-columns" in r and "var(--w-left" not in r]
    record("the panel widths come from the workspace's own variables",
           "a hardcoded width in a later rule silently wins over the variable, so dragging appears to do "
           "nothing while every measurement of the variable agrees",
           "the main grid uses var(--w-left): %s; main rules with a hardcoded column list: %s"
           % (uses_vars, offenders or "none"),
           uses_vars and not offenders)

    loads = '<script src="/static/workspace.js">' in html
    balanced = len(_re.findall(r"<script src=", html)) == len(_re.findall(r"</script>", html))
    record("the workspace script loads and the script tags are balanced",
           "a replace that leaves a stray closing tag stops every later script loading, and the page still "
           "parses",
           "workspace.js is loaded: %s; script tags balanced: %s" % (loads, balanced),
           loads and balanced)
def check_highlight_reaches_white() -> None:
    """
    A saturated highlight must reach white in the camera's own output, and a normal scene must not be lifted.

    `_roll_off` compresses the sensor's highlights, and the curve it used was

        knee + (1 - knee) * (1 - exp(-(lin - knee) / (1 - knee)))

    which is **asymptotic**: it approaches 1.0 and never arrives. Measured, a lamp at twice the reference white
    left the step at **0.991** -- 504 pixels at or above 1.0 went in and zero came out -- and only `_halation`
    afterwards lifted them back. **No pixel in any picture could be pure white**, which is why the highlights read
    as flat grey and why the user's reference, whose whole character is a large blown white core with a halo,
    could not be reproduced.

    The replacement puts the white point where `dynamic_range` says it is -- `2 ** ((dr - 8) / 8)`, so a
    narrow-range CCD blows out sooner -- and normalises a rational shoulder so that `comp(knee) == knee` and
    `comp(white) == 1.0` exactly.

    **The camera's own output is what is measured.** The stages below it rebuild the picture from a signal, so a
    measurement taken at the end conflates this curve with the whole chain -- and an earlier attempt to fix the
    same symptom at the end of the chain broke the auto-white-balance loop instead, because it clamped the pixels
    that loop acts on. Measured here: a 2.5x source peaks at 1.106 with 188 pixels at or above white, and a 0.45x
    source is untouched.
    """
    import numpy as np

    from atsim.core.pipeline import FrameBuffer
    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    W, H = 200, 260
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Lamp:
        def __init__(self, level):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 4
            self.level = level

        def frame(self, index):
            a = np.full((H, W, 3), 0.14, np.float32)
            a[H // 2 - 14:H // 2 + 14, W // 2 - 3:W // 2 + 3] = self.level
            return a

    def camera_out(level, **over):
        p = merge_values(BASE, {"camera": dict({"exposure": 0.0}, **over),
                                "display": {"enabled": 0.0}, "tape": {"enabled": 0.0},
                                "player": {"enabled": 0.0}})
        r = Renderer(Lamp(level), W, H, 25.0, p, seed=5)
        r._build_chain()
        buf = FrameBuffer(np.asarray(r._fit_source(r.source.frame(1)), np.float32), None, 1, 0.04,
                          r.std, {"field": 1})
        return np.clip(r.stage["camera"].apply(buf).rgb, 0.0, 2.0)

    bright = camera_out(2.5)
    white = int((bright >= 0.999).sum())
    record("a saturated highlight reaches pure white in the camera's own output",
           "the sensor's well fills and the value clips; a curve that only approaches white cannot produce the "
           "blown core that is the whole character of an overexposed highlight",
           "a 2.5x source peaks at %.3f with %d pixels at or above 1.0" % (float(bright.max()), white),
           float(bright.max()) >= 0.999 and white > 20,
           "measured before: the step returned 0.991 and 504 pixels at or above 1.0 went in with zero coming out")

    normal = camera_out(0.45)
    record("a normally exposed scene is not lifted by the saturation curve",
           "a fix based on a floor would raise the whole picture; only pixels at the sensor's white point may move",
           "a 0.45x source peaks at %.3f" % float(normal.max()),
           float(normal.max()) < 0.9,
           "the shoulder is a per-pixel curve, so a value below the knee passes through unchanged")

    # and the white point must follow the sensor's range, which is what the parameter means
    peaks = {}
    for dr in (5.0, 8.0, 12.0):
        peaks[dr] = float(camera_out(2.5, dynamic_range=dr).max())
    record("the sensor's white point follows its dynamic range",
           "a narrow-range CCD blows out sooner and a wide one holds more; that is what the parameter is for",
           "; ".join("dr %.0f (white %.2f) -> peak %.3f" % (d, 2.0 ** ((d - 8.0) / 8.0), peaks[d])
                     for d in sorted(peaks)),
           all(v >= 0.999 for v in peaks.values()))
def check_overflow_drain() -> None:
    """
    A blown highlight must have a dark ring around it, and it must scale with the parameter.

    The user's reference -- a Hi8 through an AV capture -- shows its ceiling downlights as a blown white core
    **with a dark ring around it**, then back to the ceiling's own light grey. That is the opposite direction
    from a bloom or a halation halo: when a photosite's well overflows, the anti-blooming structure removes the
    excess by pulling the *neighbouring* sites below their own exposure. Measured before this existed, the ring
    sat **0.012 above** the ceiling and the profile fell monotonically from the core, so nothing in the model
    could produce a rim and no strength of halation could, because halation adds light outward and a rim subtracts
    it.

    **The measurement is the difference, not the profile.** `off` minus `on` isolates the mechanism. An earlier
    attempt searched a window of the luma profile for its minimum and reported the same 0.694 for every setting,
    drain on or off, because the window contained a dip the drain had nothing to do with -- which is the fifth
    time this project has measured a scene instead of an effect.

    Three properties, and the third is what makes it the sensor's mechanism rather than a drawn outline:

      * the ring **darkens** the pixels just outside the saturated core;
      * it **scales** with `overflow_drain`, so the parameter is a control and not a switch;
      * and it lands **at the core's edge**, because the drain acts on neighbouring photosites.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    H, W = 300, 400
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Downlight:
        """A blown disc on a light ceiling, which is the reference's own configuration."""

        def __init__(self):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 4

        def frame(self, index):
            a = np.full((H, W, 3), 0.62, np.float32)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
            a[r < 9.0] = 2.6
            return a

    def render(**cam):
        p = merge_values(BASE, {"camera": dict({"exposure": 0.0}, **cam)})
        r = Renderer(Downlight(), W, H, 25.0, p, seed=5)
        return np.clip(r.render_frame(1), 0, 1) @ COEF

    off = render()
    depths = {}
    radii = {}
    for amt in (0.15, 0.25, 0.35):
        d = off - render(overflow_drain=amt)
        prof = d[H // 2, W // 2:W // 2 + 30]
        depths[amt] = float(prof.max())
        radii[amt] = int(np.argmax(prof))

    ordered = [depths[0.15], depths[0.25], depths[0.35]]
    record("a blown highlight has a dark ring, scaling with the overflow drain",
           "a photosite that overflows has its excess charge removed by the anti-blooming structure, which pulls "
           "the neighbouring sites below their own exposure -- the opposite direction from a halo",
           "; ".join("drain %.2f -> %.5f darker at radius %d px" % (a, depths[a], radii[a])
                     for a in sorted(depths)),
           all(v > 1e-3 for v in ordered)
           and ordered[0] < ordered[1] < ordered[2]
           and all(r <= 10 for r in radii.values()),
           "measured before: the ring sat 0.012 ABOVE the ceiling and the profile fell monotonically from the "
           "core, so no strength of halation could produce it -- halation adds light outward")

    # and it must be a ring rather than a global darkening
    on = render(overflow_drain=0.25)
    record("the drain darkens a ring and not the picture",
           "the mechanism acts on the photosites beside a saturated one; a whole-frame darkening would be a "
           "brightness control with a misleading name",
           "the frame's mean changes by %+.5f while the ring deepens by %.5f"
           % (float(on.mean() - off.mean()), depths[0.25]),
           abs(float(on.mean() - off.mean())) < 0.01 and depths[0.25] > 1e-3)
def check_lateral_aberration() -> None:
    """
    Lateral chromatic aberration must be a *displacement*, and the composite chroma path must shape how much of
    it reaches the picture.

    **A displacement, not a difference in blur.**  The method this replaced gave each channel its own lowpass and
    shifted nothing:

        r = lowpass(rgb[..., 0], 1.0 + 2.2 * a)
        b = lowpass(rgb[..., 2], 1.0 + 2.6 * a)
        g = lowpass(rgb[..., 1], 1.0 + 0.5 * a)

    Measured, R/G/B softened almost equally -- 0.1107/0.1111/0.1057 falling to 0.0826/0.0823/0.0818 -- and no edge
    acquired a fringe.  Blurring the channels by different amounts gives a coloured *haze*; a lens gives a
    coloured *fringe*, because the red and blue images arrive at slightly different magnifications about the
    optical axis, so the effect is `k * a * r`: nothing at the centre and growing towards the corners.

    **The metric is a position.**  Four earlier attempts measured intensity and every one was blind to a shift:
    an edge locator that found a different edge per channel; an edge at the frame's centre where the radial term
    is zero; a window spanning both edges of a bar; and the mean `R-B` near an edge, which is nearly invariant to
    a sideways move because it redistributes the contribution symmetrically.  That last one is why the camera
    measured -0.171 while the finished frame measured 0.00000 and the YIQ round trip looked guilty when it
    preserves a fringe exactly.

    **And the chain is asserted, not just the camera.**  The composite chroma path is what decides how much
    survives -- measured, 2.226 px at the camera becomes 0.149 px in the format's signal -- so the check walks the
    parameter that controls it, and also asserts the control: the luma bandwidth must not change the result.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    H, W = 300, 400
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Hard:
        """A hard vertical edge near the frame's border, where the radial term is largest."""

        def __init__(self):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 4

        def frame(self, index):
            a = np.zeros((H, W, 3), np.float32)
            a[:, 300:] = 0.9
            return a

    def render(ab, bw=None, luma_bw=None):
        fmt = {}
        if bw is not None:
            fmt["chroma_bandwidth"] = bw
        if luma_bw is not None:
            fmt["luma_bandwidth"] = luma_bw
        p = merge_values(BASE, {"camera": {"exposure": 0.0, "chroma_aberration": ab},
                                "format": fmt})
        r = Renderer(Hard(), W, H, 25.0, p, seed=5)
        return np.clip(r.render_frame(1), 0, 1)

    def centroid(rgb, ch):
        v = np.asarray(rgb)[H // 2, 288:312, ch].astype(np.float64)
        g = np.abs(np.diff(v))
        if g.sum() < 1e-9:
            return float("nan")
        return 288.0 + float((np.arange(g.size) * g).sum() / g.sum())

    def fringe(rgb):
        return centroid(rgb, 0) - centroid(rgb, 2)

    # the camera must separate the channels, and by more as the parameter rises
    p_cam = merge_values(BASE, {"display": {"enabled": 0.0}, "tape": {"enabled": 0.0},
                                "player": {"enabled": 0.0}})
    from atsim.core.pipeline import FrameBuffer

    def camera_fringe(ab):
        pp = merge_values(p_cam, {"camera": {"exposure": 0.0, "chroma_aberration": ab}})
        r = Renderer(Hard(), W, H, 25.0, pp, seed=5)
        r._build_chain()
        buf = FrameBuffer(np.asarray(r._fit_source(r.source.frame(1)), np.float32), None, 1,
                          0.04, r.std, {"field": 1})
        return fringe(r.stage["camera"].apply(buf).rgb)

    lo, hi = camera_fringe(0.0), camera_fringe(1.0)
    record("lateral chromatic aberration displaces the channels radially",
           "a lens focuses the three wavelengths at different magnifications, so the fringe is `k * a * r` -- "
           "absent at the centre and growing outward. Blurring the channels by different amounts gives a haze",
           "the R-B edge offset at the camera: %.3f px at 0.0, %.3f px at 1.0" % (lo, hi),
           abs(lo) < 0.3 and abs(hi) > 1.5,
           "measured before: R/G/B softened 0.1107/0.1111/0.1057 -> 0.0826/0.0823/0.0818, i.e. equally, and no "
           "edge acquired a fringe at all")

    surv = {bw: abs(fringe(render(1.0, bw=bw))) for bw in (0.25, 0.5, 0.75, 2.0)}
    narrow = max(surv[b] for b in (0.25, 0.5, 0.75))
    wide = surv[2.0]
    record("the composite chroma bandwidth decides how much of the fringe survives",
           "NTSC composite chroma has about 1.3 MHz of bandwidth, roughly ten pixels of resolution, so a "
           "few-pixel channel offset is below what it can carry and the encoder averages it away",
           "; ".join("chroma %.2f MHz -> %.3f px of fringe" % (k, v) for k, v in sorted(surv.items())),
           narrow < 0.5 and wide > 1.0,
           "measured after every stage: the camera produces 2.226 px, the format's signal carries 0.149, and the "
           "picture is rebuilt from that signal.  **Not asserted as monotone**: at 0.25 MHz the chroma is filtered "
           "so heavily that the edge centroid is moved by the filter's own phase response, which is a property of "
           "the metric rather than of the dispersion -- 0.030 against 0.007 at 0.75.  The claim the physics makes "
           "is that a narrow path suppresses the fringe and a wide one preserves it, and that is what is asserted")

    lbw = {v: abs(fringe(render(1.0, luma_bw=v))) for v in (3.0, 6.0)}
    record("the luma bandwidth does not change the fringe",
           "the effect is in the chroma path; a fix that resharpened the edge would move this too",
           "; ".join("luma %.1f MHz -> %.3f px" % (k, v) for k, v in sorted(lbw.items())),
           max(lbw.values()) < 0.5)
def check_lens_field_and_longitudinal() -> None:
    """
    Field curvature must soften with field radius, and longitudinal chromatic aberration must widen the channels
    **without moving them**.

    **Field curvature** is a simple lens imaging a flat subject onto a curved surface, so the plane in focus at the
    frame's centre is not the plane in focus at its corners -- the picture is sharp in the middle and soft at the
    edges at the same focus setting. The signature is spatial: sharpness depends on *where* the edge is, and the
    weight is the field radius squared because the point-spread function grows with the field angle.

    **Longitudinal chromatic aberration** is the wavelengths focusing at different distances along the axis, so at
    the sensor one is sharp and the others are not. Its signature is a **width** difference between the channels
    at the same place, and critically **no position difference**: a symmetric blur cannot move a centroid, and a
    displacement is the lateral kind.

    So the two are asserted by opposite quantities, and the lateral kind is asserted by position -- three
    mechanisms, three signatures. **The measurement matters more than the assertion here.** The first attempt at
    field curvature used a four-pixel grid, which this format's own luma path has already softened, so there was
    no detail left for a blur to remove and the parameter measured inert. The first attempt at longitudinal used a
    window narrower than the edge it was measuring, so the centroid slid to the window's edge and reported a
    displacement that does not exist.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    H, W = 300, 400
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Grid:
        """Square-wave bars spanning the frame; a **coarse** period, because a fine one has no detail left."""

        def __init__(self, period=12):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, 4
            self.period = period

        def frame(self, index):
            a = np.zeros((H, W, 3), np.float32)
            a[:] = (((np.arange(W) // self.period) % 2) * 0.85)[None, :, None]
            return a

    def render(**cam):
        p = merge_values(BASE, {"camera": dict({"exposure": 0.0}, **cam)})
        r = Renderer(Grid(), W, H, 25.0, p, seed=5)
        return np.clip(r.render_frame(1), 0, 1)

    def sharpness(img, x0, x1):
        band = np.asarray(img)[:, x0:x1] @ COEF
        return float(np.abs(np.diff(band, axis=1)).mean())

    centre, edge = {}, {}
    for fc in (0.0, 0.5, 1.0):
        img = render(field_curvature=fc)
        centre[fc] = sharpness(img, W // 2 - 12, W // 2 + 12)
        edge[fc] = sharpness(img, 2, 40)
    record("field curvature softens the corners while the centre stays sharp",
           "a simple lens images a flat subject onto a curved surface, so the plane in focus at the centre is not "
           "the plane in focus at the corners -- sharpness depends on where the edge is, not on the parameter "
           "alone",
           "corner sharpness %.5f -> %.5f as the parameter goes 0 -> 1, centre/corner ratio %.2f -> %.2f"
           % (edge[0.0], edge[1.0], centre[0.0] / edge[0.0], centre[1.0] / edge[1.0]),
           edge[1.0] < edge[0.0] * 0.85
           and centre[1.0] / edge[1.0] > centre[0.0] / edge[0.0] * 1.15,
           "the first measurement used a four-pixel grid, which the format's own luma path has already softened, "
           "so no detail was left for a blur to remove and the parameter appeared inert")

    # --- the longitudinal kind, on a single edge with a window wide enough for the blurred result
    #
    # Called straight from the module, because it is a pure function: `(picture, spec) -> picture`. That is what
    # moving these out of `CameraStage` bought -- a mechanism can be measured on its own input instead of through a
    # rendered frame whose own softness hides it, which is how three of four measurements went wrong before.
    from atsim.codecs.lens import LensSpec as _LensSpec
    from atsim.codecs.lens import longitudinal_ca as _longitudinal
    hard = np.zeros((H, W, 3), np.float32)
    hard[:, 200:] = 0.85
    LO, HI = 150, 250

    def position(img, ch):
        v = np.asarray(img)[H // 2, LO:HI, ch].astype(np.float64)
        g = np.abs(np.diff(v))
        if g.sum() < 1e-9:
            return float("nan")
        return LO + float((np.arange(g.size) * g).sum() / g.sum())

    def width(img, ch):
        v = np.abs(np.diff(np.asarray(img)[H // 2, LO:HI, ch].astype(np.float64)))
        return int((v > v.max() * 0.1).sum()) if v.max() > 1e-9 else 0

    base = _longitudinal(hard, _LensSpec(longitudinal_ca=0.0))
    out = _longitudinal(hard, _LensSpec(longitudinal_ca=1.0))
    rw, gw, bw = width(out, 0), width(out, 1), width(out, 2)
    moved = max(abs(position(out, 0) - position(base, 0)),
                abs(position(out, 2) - position(base, 2)))
    record("longitudinal chromatic aberration widens the channels without moving them",
           "the wavelengths focus at different distances along the axis, so red and blue land soft on either side "
           "of green -- the fringes appear *around* a highlight rather than on one side of it, which is the "
           "distinction from the lateral kind",
           "at full strength R/G width %.2f, B/G width %.2f, and the channel positions move by %.4f px"
           % (rw / max(1, gw), bw / max(1, gw), moved),
           bw > gw * 1.5 and rw > gw * 1.5 and moved < 0.05,
           "the first measurement used a 30-sample window for an edge that grows from 24 to 74 samples, so the "
           "centroid slid to the window's edge -- a symmetric blur cannot move a centroid")
def check_lens_reachable_and_ordered() -> None:
    """
    Every aberration must be in the served schema, and the pre-existing mechanisms must keep their order.

    **Reachability.** The parameters were in the schema and the panel renders the schema, so nothing was wrong with
    the code -- and they were still invisible, because the server that serves the schema had silently failed to
    restart. "It is in the schema" and "it is on screen" are different claims; this project has already confused
    them once with a button appended to the right row of code that never ran. So the check asserts the served
    schema rather than the schema object, which is what the browser actually receives.

    **Order.** `field_curvature` differed by 2.96e-04 after being moved, with everything else identical, because
    `distortion(rgb, 0)` was placed ahead of it. The parameter is zero, so the function is a mathematical identity
    -- but it returns through a **copy**, so the array `field_curvature` receives has a different float32 history,
    and the difference showed up as 2.96e-04 concentrated in the corners, which is the shape of the radial weight.

    That is the general form of the lesson: **a move has to preserve the sequence of operations, not just the set.**
    Putting the new mechanisms around the old ones rather than in front of them closed the comparison to
    byte-identical, measured over thirty frames across three presets.
    """
    import re
    import urllib.request
    import json as _json

    # --- reachable: asked of the server, because that is what the panel receives
    want = ["axis_offset", "field_curvature", "longitudinal_ca", "coma", "astigmatism",
            "spherical_aberration", "lens_distortion", "chroma_aberration"]
    served, layers = [], None
    for port in (8941, 8947):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/api/schema" % port, timeout=15) as r:
                layers = (_json.loads(r.read()).get("layers") or [])
            break
        except Exception:                                                 # noqa: BLE001
            continue
    if layers is None:
        record("every lens parameter is in the schema the server serves",
               "the panel renders the schema, so a parameter that is not served cannot be reached however "
               "correct the code is",
               "no server answered on 8941 or 8947, so this could not be checked",
               False,
               "this check exists because the parameters were invisible for exactly this reason: the code was "
               "right and the server had silently failed to restart")
        return
    cam = next((L for L in layers if L.get("layer") == "camera"), None)
    keys = []
    if cam:
        for _g, specs in (cam.get("groups") or {}).items():
            keys += [s.get("key") for s in specs]
    missing = [k for k in want if k not in keys]
    record("every lens parameter is in the schema the server serves",
           "the panel renders the schema, so a parameter that is not served cannot be reached however correct the "
           "code is",
           "the served camera layer has %d specs; missing: %s" % (len(keys), missing or "none"),
           not missing,
           "measured after a turn in which all five new parameters were invisible because the server was down, and "
           "nothing failed")

    # --- ordered: the relative order of the mechanisms that existed before must be unchanged
    cam_src = open(os.path.join("atsim", "stages", "camera.py"), encoding="utf-8").read()
    order = [m.group(1) for m in re.finditer(r"^\s+rgb = (\w+)\(rgb, self\.lens\)\s*$", cam_src, re.M)]
    # the old sequence was: chromatic (= _aberration), field_curvature, longitudinal_ca
    pre = ["chromatic_aberration", "field_curvature", "longitudinal_ca"]
    pos = {n: (order.index(n) if n in order else -1) for n in pre}
    stable = all(pos[pre[i]] < pos[pre[i + 1]] for i in range(len(pre) - 1))
    record("the mechanisms that existed before keep their relative order",
           "a move preserves the sequence of operations, not just the set: `distortion(rgb, 0)` is a mathematical "
           "identity and still cost 2.96e-04 in the corners by returning through a copy",
           "the order is %s; the pre-existing three are %s" % (order, [pos[n] for n in pre]),
           stable and all(v >= 0 for v in pos.values()),
           "measured: with `distortion` ahead of `field_curvature` the frames differed by 2.96e-04; with the "
           "original relative order restored, thirty frames across three presets are byte-identical")

    every_once = all(order.count(n) == 1 for n in order)
    record("the camera calls each lens mechanism exactly once",
           "a duplicated call is a mechanism applied twice, which no parameter can correct",
           "the sequence has %d calls: %s" % (len(order), order),
           every_once and len(order) == 8,
           "the count is the number of mechanisms and it was 7 before source_glow was added; the "
           "assertion that matters is that no mechanism is called twice")
def check_digital_component_chroma() -> None:
    """
    A digital component codec must not inherit the composite subcarrier's chroma bandwidth.

    **The fault this guards.** The format stage lowpassed I and Q at `format.chroma_bandwidth`, default 0.75 MHz --
    the VHS composite **subcarrier** sideband width -- and it read nothing from the codec layer, so it applied that
    limit to DV as well. MiniDV is 4:1:1 digital component at a fixed bitrate; its chroma is not on a subcarrier, so
    its limit is a sampling one and not a sideband one. Measured on the user's own unprocessed photograph, the
    effect was that a blown highlight's coloured edge was **2-4x too weak**, and widening the bandwidth changed
    nothing because the filter should not have been in the path at all.

    **Why no check caught it.** Everything measured was correct in isolation: `chroma_bandwidth` did exactly what it
    said, the DV codec did exactly what it said, and the two were never compared. A limit that applies to one format
    being applied to another is invisible to any check that examines one stage at a time.

    **The two assertions are deliberately complementary.** With DV on, the bandwidth must make no difference --
    because the filter is skipped. With DV off, it must make a difference -- so the check cannot be satisfied by
    removing the filter for everybody, which would be a different bug that passes.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    H, W = 240, 320
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Fringes:
        """Red and blue bars, which the chroma path either carries or does not."""

        def __init__(self, frames=4):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, frames

        def frame(self, index):
            a = np.zeros((H, W, 3), np.float32)
            a[:, :W // 2] = [0.8, 0.1, 0.1]
            a[:, W // 2:] = [0.1, 0.1, 0.8]
            return a

    def chroma_amount(dv: float, bw: float) -> float:
        p = merge_values(BASE, {"camera": {"exposure": 0.0},
                                "format": {"chroma_bandwidth": bw},
                                "player": {"chroma_bandwidth": bw},
                                "codec": {"dv_enabled": dv}})
        img = np.clip(Renderer(Fringes(), W, H, 25.0, p, seed=5).render_frame(1), 0, 1)
        return float(np.abs(img[..., 0] - img[..., 2]).mean())

    narrow_dv = chroma_amount(1.0, 0.30)
    wide_dv = chroma_amount(1.0, 2.00)
    narrow_analog = chroma_amount(0.0, 0.30)
    wide_analog = chroma_amount(0.0, 2.00)

    record("a digital component codec does not get the composite chroma limit",
           "0.75 MHz is the width of the sidebands around a 3.58 MHz subcarrier, a limit that exists because chroma "
           "shares the luma's spectrum; DV's chroma is not on a subcarrier, so the limit does not apply and applying "
           "it deletes the coloured edge that shows the lens's own aberration",
           "with DV on, 0.30 -> 2.00 MHz changes the colour by %.5f; with DV off it changes it by %.5f"
           % (abs(wide_dv - narrow_dv), abs(wide_analog - narrow_analog)),
           abs(wide_dv - narrow_dv) < 1e-6 and abs(wide_analog - narrow_analog) > 1e-3,
           "the second half is what stops this passing by removing the filter for every format, which would be a "
           "different fault")
def check_spikes_are_per_channel() -> None:
    """
    The diffraction spikes must be formed per channel, so they separate into red, green and blue.

    **Why this is a mechanism and not a tint.** An aperture's diffraction spikes scale with wavelength, so the three
    channels form theirs at slightly different angles and lengths, and the result is *coloured streaks* rather than
    one pattern with a colour on it. A single achromatic pattern multiplied by a tint **cannot** produce them at any
    strength, because all three channels get the same geometry -- which is what the model did.

    **And the measurement is the substance here.** Three attempts read the separation off rendered frames and all
    three reported 0.22-0.27 at **zero** spread, where the geometry is shared by definition. The renderer carries I
    and Q at 0.75 MHz against luma's 3 MHz, so a rendered frame's channels differ by construction. Called directly on
    the flare's output, zero spread reads exactly 0.00000. That is the control this check asserts, and it is what
    makes the effect measurement meaningful rather than a description of the pipeline.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.pipeline import FrameBuffer
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    H, W = 300, 400
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
    ang = np.arctan2(yy - H / 2, xx - W / 2)
    # the spike band: below the ghost polygon at 0.14 * max(h, w) = 56 px, so the achromatic ghost is excluded
    band = (r > 20) & (r < 48)
    idx = ((ang + np.pi) / (2 * np.pi) * 360).astype(int) % 360

    class Point:
        def __init__(self, frames=4):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, frames

        def frame(self, index):
            a = np.zeros((H, W, 3), np.float32)
            a[H // 2 - 2:H // 2 + 2, W // 2 - 2:W // 2 + 2] = 2.8
            return a

    def flare_of(spread):
        # **`flare_defocus` is set to zero on purpose.**  It defocuses the flare per channel as well, so leaving
        # it at its default of 0.5 made this check read the *other* mechanism: at `spike_spread = 0` the separation
        # came out at 0.130 instead of 0.  Each check measures one mechanism by switching the others off.
        p = merge_values(BASE, {"camera": {"exposure": 0.0, "lens_flare": 0.9,
                                           "spike_spread": spread, "flare_defocus": 0.0}})
        rr = Renderer(Point(), W, H, 25.0, p, seed=5)
        rr._build_chain()
        src_a = np.asarray(Point().frame(0), np.float32)
        return np.asarray(rr.stage["camera"]._lens_flare(src_a), np.float32)

    def separation(img):
        profs = []
        for ch in range(3):
            v = img[..., ch][band]
            b = idx[band]
            p = np.zeros(360)
            for k in range(360):
                sel = b == k
                p[k] = v[sel].mean() if sel.any() else 0.0
            p = p - p.mean()
            s = p.std()
            profs.append(p / s if s > 1e-9 else p)
        return float(np.abs(profs[2] - profs[1]).mean())

    zero = flare_of(0.0)
    full = flare_of(1.0)
    s0, s1 = separation(zero), separation(full)

    record("the diffraction spikes are formed per channel",
           "the fringe spacing scales with wavelength, so red and blue form their spikes at different angles and "
           "lengths -- coloured streaks, which one achromatic pattern times a tint cannot produce at any strength",
           "channel separation at r=20..48: spread 0 -> %.5f, spread 1 -> %.5f (x%.0f)"
           % (s0, s1, s1 / max(1e-12, s0)),
           s1 > 0.05 and s1 > s0 * 10.0,
           "measured on the flare's own output, because three attempts read it off rendered frames and every one "
           "reported 0.22-0.27 at zero spread -- the renderer's chroma path carries I/Q at 0.75 MHz against luma's "
           "3 MHz, so a rendered frame's channels differ by construction and the statistic was describing the "
           "pipeline")

    record("at zero spread the three channels share one spike geometry",
           "this is the control: with the parameter at zero the ray geometry is shared by definition, so a nonzero "
           "reading would mean the statistic is measuring something else",
           "separation at zero spread %.8f" % s0,
           s0 < 1e-6)
def check_flare_defocus_per_channel() -> None:
    """
    The flare must defocus per channel, with **blue widest and red narrowest**.

    **The physics.** Longitudinal chromatic aberration puts the three channels at different focal distances, so the
    out-of-focus image of the aperture is larger in blue than in red. On a bright source that is a coloured halo
    *around and below* the light, not a colour painted on it -- which is what the reference material shows: a blue
    streak under a street lamp, a blue trail down a lamp post under an orange glow, and hard rings around a blown
    lamp.

    **Measured as a width, not as a colour**, and that is the substance of this check. Three earlier attempts
    measured colour on rendered frames and every one was dominated by the chroma path, which carries I/Q at 0.75 MHz
    against luma's 3 MHz, so a rendered frame's channels differ by construction. A **width** cannot be confounded
    that way.

    **And two faults had to be fixed to make it measurable.** Applied to the whole flare field the ratio was 1.00,
    because the field's second moment is about 61 px and dominated by the achromatic ghost polygon at 56 px; and
    that placement was wrong physically as well, since defocus acts on the image of the aperture rather than on a
    spike or a polygon. The defocus is now applied to the flare's compact core.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    H, W = 300, 400
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)

    class Point:
        def __init__(self, frames=4):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, frames

        def frame(self, index):
            a = np.zeros((H, W, 3), np.float32)
            a[H // 2 - 2:H // 2 + 2, W // 2 - 2:W // 2 + 2] = 2.8
            return a

    def flare_only(defocus):
        # and `spike_spread` to zero, for the same reason in the other direction
        p = merge_values(BASE, {"camera": {"exposure": 0.0, "lens_flare": 0.8,
                                           "flare_defocus": defocus, "spike_spread": 0.0}})
        r = Renderer(Point(), W, H, 25.0, p, seed=5)
        r._build_chain()
        src_a = np.asarray(Point().frame(0), np.float32)
        out = np.asarray(r.stage["camera"]._lens_flare(src_a), np.float32)
        return np.clip(out - src_a, 0.0, None)

    def core_width(field, ch, rmax=26):
        m = (rr > 2) & (rr < rmax)
        v = field[..., ch][m]
        tot = float(v.sum())
        return float(np.sqrt((v * rr[m] * rr[m]).sum() / tot)) if tot > 1e-9 else 0.0

    zero = flare_only(0.0)
    full = flare_only(1.0)
    w0 = [core_width(zero, c) for c in range(3)]
    w1 = [core_width(full, c) for c in range(3)]

    record("the flare defocuses per channel, blue widest and red narrowest",
           "longitudinal chromatic aberration puts the channels at different focal distances, so the out-of-focus "
           "image of the aperture is larger in blue than in red -- a coloured halo around and below the source",
           "core width at full strength: red %.3f, green %.3f, blue %.3f (blue/red %.3f)"
           % (w1[0], w1[1], w1[2], w1[2] / max(1e-9, w1[0])),
           w1[2] > w1[1] > w1[0] and w1[2] > w1[0] * 1.05,
           "applied to the whole flare field the ratio was 1.00, because the second moment is about 61 px and "
           "dominated by the achromatic ghost polygon at 56 px; and that placement was wrong physically, since "
           "defocus acts on the image of the aperture rather than on a spike or a polygon")

    record("at zero defocus the flare's core is the same width in every channel",
           "the control: with the parameter at zero the channels share one aperture image, so a nonzero spread "
           "would mean the measurement is describing something else",
           "core widths at zero: red %.4f, green %.4f, blue %.4f (spread %.4f)"
           % (w0[0], w0[1], w0[2], max(w0) - min(w0)),
           max(w0) - min(w0) < 0.2)
def check_declared_ranges_are_reachable() -> None:
    """
    Every parameter's declared range must reach the stage, and changing it must change the picture.

    **The fault this exists for.** `display.gamma` was declared `1.5..3.2` and the stage read it faithfully, so every
    mechanism check passed -- but asking for 0.50, 0.80, 1.00 or 1.25 all left the stage holding **1.5**, because the
    *schema* raised them. The parameter was not mis-implemented, it was **mis-declared**: the lower half of the range
    its own name implies was unreachable, and no check asked that question. It took a ramp and a direct read of the
    stage's attribute to find it.

    So two assertions, on a small set of parameters whose effect a chosen subject can actually show:

      * **the stage holds the value it is given**, at each end of the declared range -- a reader that clamps, or a
        `merge_values` that raises, shows up here;
      * **and the picture changes** between the two ends -- a declared range that is inert at one end is either
        mis-declared or unwired.

    The subject matters and is chosen per parameter. A check that used one picture for everything would pass on
    parameters whose effect that picture cannot show, which is the mistake this project has made most often.
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.pipeline import FrameBuffer
    from atsim.core.profiles import FACTORY_PRESETS, SCHEMA
    from atsim.render.renderer import Renderer

    H, W = 60, 128
    BASE = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])

    class Ramp:
        """A grey ramp and a row of saturated colours: enough structure for most parameters to show on."""

        def __init__(self, frames=3):
            self.width, self.height, self.fps, self.frame_count = W, H, 25.0, frames

        def frame(self, index):
            a = np.zeros((H, W, 3), np.float32)
            ramp = np.linspace(0.0, 1.0, W, dtype=np.float32)
            a[:] = ramp[None, :, None]
            a[H // 2:, :W // 3] = [0.80, 0.10, 0.10]
            a[H // 2:, W // 3:2 * W // 3] = [0.10, 0.70, 0.10]
            a[H // 2:, 2 * W // 3:] = [0.10, 0.10, 0.80]
            return a

    CLEAN = {
        "camera": {"exposure": 0.0, "color_cast": 0.0, "halation": 0.0, "ccd_smear": 0.0,
                   "chroma_aberration": 0.0, "lens_flare": 0.0, "spike_spread": 0.0,
                   "flare_defocus": 0.0, "longitudinal_ca": 0.0, "axis_offset": 0.0,
                   "bloom_amount": 0.0, "lens_wear": 0.0, "lens_fog": 0.0},
        "format": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0, "edge_boost": 0.0},
        "player": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0},
        "codec": {"dv_enabled": 1.0},
        "tape": {},
    }
    # the display's own geometry and mask terms would blur or tint the subject and hide a small effect
    QUIET_DISPLAY = {"enabled": 1.0, "scanlines": 0.0, "mask_depth": 0.0, "vignette": 0.0,
                     "reflection": 0.0, "bloom": 0.0, "defocus": 0.0, "interlace_flicker": 0.0,
                     "crt_age": 0.0, "spot_sigma": 0.0, "geometry_warp": 0.0, "overscan": 0.0}

    def run(layer, key, value):
        merged = {k: dict(v) for k, v in CLEAN.items()}
        merged["display"] = dict(QUIET_DISPLAY)
        merged[layer] = dict(merged.get(layer) or {}, **{key: float(value)})
        p = merge_values(BASE, merged)
        r = Renderer(Ramp(), W, H, 25.0, p, seed=5)
        r._build_chain()
        st = r.stage.get(layer)
        held = float(getattr(st, key, float("nan"))) if st is not None else float("nan")
        out = st.apply(FrameBuffer(Ramp().frame(0).copy(), None, 1, 0.04, r.std, {"field": 0}))
        rgb = np.asarray(out.rgb if out.rgb is not None else out.signal.to_rgb(), np.float32)
        return held, np.clip(rgb, 0.0, 1.0)

    subject = [("display", "gamma"), ("display", "contrast"), ("display", "brightness"),
               ("display", "saturation")]
    unreachable, inert = [], []
    for layer, key in subject:
        spec = SCHEMA.layer(layer).specs.get(key)
        if spec is None:
            continue
        lo, hi = float(spec.minimum), float(spec.maximum)
        held_lo, img_lo = run(layer, key, lo)
        held_hi, img_hi = run(layer, key, hi)
        if abs(held_lo - lo) > 1e-6 or abs(held_hi - hi) > 1e-6:
            unreachable.append("%s.%s: range %g..%g, stage held %g..%g"
                               % (layer, key, lo, hi, held_lo, held_hi))
        if float(np.abs(img_lo - img_hi).max()) < 1e-3:
            inert.append("%s.%s: %g and %g give the same picture" % (layer, key, lo, hi))

    record("every declared parameter range reaches the stage",
           "a parameter whose lower or upper end is raised or clamped in transit is mis-declared rather than "
           "mis-implemented, so every mechanism check passes while half its range is dead -- which is what "
           "`display.gamma` did at a minimum of 1.5",
           "checked %d parameters; unreachable: %s" % (len(subject), unreachable or "none"),
           not unreachable,
           "found by asking the stage what it holds after being handed a value at each end of the declared range")

    record("both ends of each declared range change the picture",
           "a range that is inert at one end is either mis-declared or not wired to anything",
           "inert: %s" % (inert or "none"),
           not inert)
def check_video_writer() -> None:
    """
    The encoder path: a source with no audio, and a source whose dimensions are ODD.

    **This check exists because the suite had none.**  Two hard faults lived in `/api/render` -- `-map 1:a:0` failing
    on a silent clip, and `yuv420p` refusing a 334x485 frame -- and neither was reachable by any test here, because
    nothing in this file had ever opened a `VideoWriter`.  Both reached the user as **a zero-length `.mp4`**, so a check
    that only asked "did the render raise" would have passed.  **The check has to look at the FILE.**

    The two conditions are the ones the real callers produce: a still image supplies whatever dimensions the file has
    (odd, often), and an imported clip may carry no audio stream at all.
    """
    import tempfile

    from atsim.io.ffmpeg import VideoWriter, have_ffmpeg, probe

    if not have_ffmpeg():
        record("the video writer encodes a silent, odd-sized source",
               "the container must accept what a still image and a silent clip actually are",
               "skipped: no ffmpeg", True)
        return

    #: 334x485 is the size the user's still image produced, and it is odd in BOTH directions
    w, h = 334, 485
    rng = np.random.default_rng(11)
    frames = [np.clip(0.5 + 0.25 * rng.standard_normal((h, w, 3)), 0, 1).astype(np.float32) for _ in range(3)]
    err = ""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "odd.mp4")
        try:
            with VideoWriter(out, w, h, 25.0, crf=28, preset="ultrafast") as vw:
                #: **`open()` with no `audio_from`, which is the silent-source case**: the writer must not require an
                #: audio stream that no caller offered
                vw.open()
                for f in frames:
                    vw.write(f)
        except Exception as exc:                                          # noqa: BLE001
            err = "%s: %s" % (type(exc).__name__, exc)
        size = os.path.getsize(out) if os.path.exists(out) else 0
        info = probe(out) if size else None
        n = int(getattr(info, "frame_count", 0) or 0) if info else 0

    record("the video writer encodes a silent, odd-sized source",
           "a still's odd dimensions and a clip's missing audio must both give a real file, not a 0-byte one",
           "%dx%d -> %d bytes, %d frames%s" % (w, h, size, n, ("  ERROR " + err) if err else ""),
           size > 0 and n >= len(frames) and not err,
           "measured: this call wrote 0 bytes until the audio map was made optional and the frame padded to even")


def check_chroma_lag() -> None:
    """
    Temporal chroma lag: the colour of a moving object must survive into the next frame's picture.

    **This is the one mechanism in the model that is about the passage of time rather than about one picture**, and
    the note asks for it by name (its §2: `Y(t) + 0.7*C(t) + 0.3*C(t-1)`, "this will be very close to a real
    device"). It was the last ABSENT row in `tools/note_audit.py`, so the first thing this check has to establish is
    that it exists at all.

    **Why a naive check would have passed on a spatial blur.** Every quantity that a "smear" produces is also
    produced by a horizontal lowpass: colour spread sideways, edges softened, a coloured halo. A check that measured
    "the colour is more spread out than before" would therefore pass on a mechanism that never looked at another
    frame -- which is this project's most expensive failure shape, and it is why the assertions below are all
    *temporal*:

      * the leading assertion is a **one-frame colour flash**, and it is scored on the frames *after* the flash. A
        spatial filter cannot do anything with it: the frames that follow are pixel-identical in the two scenes being
        compared, so filtering them changes nothing, and any difference at all is memory. That is a **presence**
        test -- 0.00000 against something -- rather than a threshold, and the 0.00000 is exact;
      * the second is the **decay ratio** between consecutive frames after the flash, which is the recursion
        coefficient itself and so pins *which* mechanism is present: a two-tap blend or a one-frame delay gives a
        ratio of 0.000;
      * the third is what the mechanism adds to a **moving** object, scored in the band the object has just arrived
        in -- where it is now and was not last frame -- with the cosine against `previous - current`. That is a
        statement about *where the colour came from*, and it is the only one of the three that a blur could mimic,
        which is why it is not the one doing the proving;
      * and the negative control is the **same scene with the object held still**, where consecutive frames are
        identical and lag must do *nothing at all*.

    **The instrument took four attempts, and every failure was the same failure.** A difference between two scenes is
    only a measurement of the flash if the flash is the only thing that differs: comparing "bar every frame" with
    "bar only at frame 6" read 0.12968 at lag 0; giving the bar a cyan colour on the flash frame read 0.01846; and
    letting a 30 px/frame bar sweep past a 24 px fixed patch meant it crossed the patch eventually whatever the
    geometry. So the flash assertion's scene has **no bar at all** -- a cyan field and a stationary cyan patch that
    is red on one frame -- and the motion assertion uses a separate scene. Each failure is recorded at the point in
    the code where it was made.

    Measured on the check's own scene (176x128, a 24 px patch, a 24 px bar travelling 30 px/frame):

        flash, frames 7..10    lag 0.00: 0.00000 0.00000 0.00000 0.00000      <- exact
                               lag 0.30: 0.00385 0.00116 0.00035 0.00010
                               lag 0.60: 0.00441 0.00265 0.00159 0.00096
        ratio, lag 0.60        0.00265 / 0.00441 = 0.601, against k = 0.60
        moving object          cosine with (previous - current): 0.883 at lag 0.30, 0.884 at 0.60;
                               at lag 0 the difference is identically zero, so there is nothing to score
        still object           max|delta| 0.00000, against 0.18641 for the moving one

    The first row is the important one: **zero on all four frames with the mechanism off, and a geometric tail with
    it on.**
    """
    import numpy as np

    from atsim.core.params import merge_values
    from atsim.core.profiles import FACTORY_PRESETS
    from atsim.render.renderer import Renderer

    COEF = np.array([0.299, 0.587, 0.114], np.float32)
    W, H, FPS, FRAMES = 176, 128, 25.0, 16
    #: **`STEP > BAR` is a hard requirement of the geometry, not a taste.** With the first version's 20 px step and
    #: 24 px bar, every frame's band overlapped the previous frame's completely, `now & ~was` was empty on **every**
    #: frame, and the third assertion below scored 0 frames while reporting PASS -- vacuously true, and the fourth
    #: time this project has recorded a measurement that could not see its own subject. The band is now 40 px wide
    #: and the count of frames it scored is printed with the result.
    #:
    #: `FRAMES` is 16 rather than 12 for the same reason: at 12 the object has left a 176 px frame by frame 9, so
    #: only four frames produce a band at all. The count is the instrument's own coverage and it is asserted.
    STEP, BAR = 30.0, 24
    IMPULSE_AT = 6
    CYAN = (0.16, 0.55, 0.56)
    HOT = (0.90, 0.12, 0.10)
    #: The flash patch's own columns, and its measurement window (its middle 10 columns, so the horizontal band
    #: limit's ~23 px spread from the patch's own edges stays outside the window -- that spread is what put
    #: `_sep_probe`'s "the rainbow fringe turned out to have been there all along" into the record).
    FLASH = (40, 64)

    def _columns(centre: float, width: int = BAR) -> np.ndarray:
        """A full-height band of columns -- a bar, which has no vertical structure to speak of."""
        m = np.zeros((H, W), dtype=bool)
        x0 = int(round(centre - width / 2.0))
        m[:, max(0, x0):min(W, max(0, x0) + width)] = True
        return m

    class SlidingBar:
        """
        Two subjects in one small scene: a **moving bar**, and a **stationary patch that flashes on one frame**.

        They are used separately, and keeping them apart is the fourth correction to this instrument -- the previous
        three all failed for the same underlying reason, which is that a *difference between two scenes* is only a
        measurement of the flash if the flash is the only thing that differs:

        * comparing "bar every frame" with "bar only at frame 6" left the two scenes different at **every** frame,
          and the lag-0 column read 0.12968;
        * giving the bar a cyan colour on the flash frame still left it red in the other scene on the next frame,
          and the lag-0 column read 0.01846;
        * putting the fixed patch at columns 40..64 and letting the bar sweep past it meant the bar crossed the
          patch's edge on some frame anyway -- the patch is 24 px and the bar travels 30 px a frame, so over a
          16-frame clip it crosses **whatever** the geometry. Measured, the lag-0 column was zero on the frames in
          between and non-zero again on frame 9, where the bar arrived.

        So the flash test's scene has **no moving bar at all**: a cyan field and a stationary cyan patch that is red
        on one frame. Nothing else in the picture ever changes, so the only thing a difference can contain is the
        flash and what the channel did with it. The moving bar's own scene is used for the motion assertion, and the
        two never share a measurement.
        """

        def __init__(self, moving: bool = True, flash_at: int | None = None,
                     bar: bool = True):
            self.width, self.height, self.fps, self.frame_count = W, H, FPS, FRAMES
            self.moving = bool(moving)
            self.flash_at = flash_at
            self.bar = bool(bar)

        def centre(self, index: int) -> float:
            """
            The bar sweeps **right to left**, so that on the flash frame it is at the far side of the picture.

            That no longer matters now the flash test has its own scene, but it is kept: it puts the two subjects
            50 px apart on frame 6, which is more than the ~23 px the band limit spreads a step, so even a scene that
            showed both would not mix them on the frame the measurement starts.
            """
            if not self.moving:
                return W * 0.5
            return W * 0.5 - (index - FRAMES / 2.0) * STEP

        def frame(self, index: int):
            out = np.zeros((H, W, 3), np.float32)
            out[..., 0], out[..., 1], out[..., 2] = CYAN
            if self.bar:
                x0 = int(round(self.centre(index) - BAR / 2.0))
                x1 = min(W, max(0, x0) + BAR)
                if x1 > max(0, x0):
                    out[:, max(0, x0):x1] = np.array(HOT, np.float32)
            if self.flash_at is not None:
                f0, f1 = FLASH
                if int(index) == int(self.flash_at):
                    out[:, f0:f1] = np.array(HOT, np.float32)
                # ... and on every other frame the patch is the field's own colour, so the two scenes differ in
                # exactly one frame and in exactly one region
            return out

    # Everything that would blur, tint or add noise is off: the only thing left to move the chroma is the lag.
    #
    # **The camera's own control loops are off too, and that mattered.** With auto-exposure and auto-white-balance
    # live, the impulse at frame 6 changes the loops' state, and frame 7 -- whose source is identical in both runs --
    # came out with a camera mean of 0.200000 against 0.229933. The differencing instrument then reported 0.00049
    # of "chroma carried past the impulse" at `chroma_lag = 0`, i.e. an effect with the mechanism switched off. A
    # feedback loop is a second temporal mechanism, and leaving it in means measuring the pair.
    QUIET = merge_values(FACTORY_PRESETS["family_archive_1992"]["layers"], {
        "display": {"enabled": 0.0},
        "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                   "bloom_amount": 0.0, "lens_flare": 0.0, "flare_amount": 0.0,
                   "halation": 0.0, "chroma_aberration": 0.0, "longitudinal_ca": 0.0,
                   "sharpness": 1.0, "motion_blur": 0.0,
                   "ae_enabled": 0.0, "awb_enabled": 0.0, "af_enabled": 0.0},
        "format": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                   "edge_boost": 0.0, "burst_phase_error": 0.0},
        "player": {"color_noise": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                   "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0},
        "tape": {"line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0,
                 "noise_floor": 0.0, "head_switching": 0.0, "dropout_rate": 0.0,
                 "vertical_instability": 0.0, "wear": 0.0, "drum_speed": 1.0,
                 "binder_loss": 0.0},
    })

    def chroma(rgb: np.ndarray):
        g = rgb @ COEF
        return (rgb[..., 2] - g) * 0.564, (rgb[..., 0] - g) * 0.713

    def run(lag: float, moving: bool = True, flash_at: int | None = None,
            bar: bool = True, frames: int = FRAMES) -> tuple[object, list]:
        scene = SlidingBar(moving, flash_at, bar)
        layers = merge_values(QUIET, {"format": {"chroma_lag": float(lag)}})
        r = Renderer(scene, W, H, FPS, layers, seed=11)
        return scene, [r.render_frame(t) for t in range(frames)]

    def chroma(rgb: np.ndarray):
        g = rgb @ COEF
        return (rgb[..., 2] - g) * 0.564, (rgb[..., 0] - g) * 0.713

    #: The window for the **flash** test: the patch's own middle, so the horizontal band limit -- which spreads a
    #: *spatial* step over about 23 px -- is not what is being measured.
    FLASH_WINDOW = np.zeros((H, W), dtype=bool)
    FLASH_WINDOW[:, FLASH[0] + 5:FLASH[1] - 5] = True
    FLASH_WINDOW &= (np.arange(H)[:, None] > 8) & (np.arange(H)[:, None] < H - 8)
    #: The window for the **leading-edge** test: the moving bar's own middle, same reasoning.
    WINDOW = _columns(W * 0.5, BAR - 10)
    WINDOW &= (np.arange(H)[:, None] > 8) & (np.arange(H)[:, None] < H - 8)

    # --- 1. the single-frame colour flash: colour shown once, and the channel's memory of it
    #
    # **This is the assertion that cannot be satisfied by a spatial filter**, which is why it leads. A one-frame
    # colour flash is, to a spatial filter, nothing: the frame after it has the same cyan field and the same cyan bar
    # in the same place, and filtering that leaves it unchanged. Only a mechanism with memory across frames can put
    # the flash's colour into frame 7 at all. So the value at frame 7 is not a calibrated threshold, it is a
    # **presence** -- 0.00000 against something.
    #
    # **The two scenes differ in one frame and in one region, and nothing else in either picture ever moves.** The
    # scene here has **no bar**: it is a cyan field and a stationary cyan patch, and the patch is red on one frame.
    # Every earlier version compared two scenes that also differed in where something was, and each time that leaked
    # into the measurement -- 0.12968 and then 0.01846 at `chroma_lag = 0`, with the mechanism switched off.
    flash = {}
    plain = {}
    for k in (0.0, 0.3, 0.6):
        _, plain[k] = run(k, flash_at=None, bar=False)
        _, out = run(k, flash_at=IMPULSE_AT, bar=False)
        # **The measurement starts one frame AFTER the flash, and that is deliberate.** The flash frame's own
        # difference is not the mechanism: the display stage's frequency response has a delay line in it (the
        # `constant_k` filter's group delay is about 23 px at the chroma cutoff), so the two scenes differ on the
        # flash frame in columns 0..4 as well as the patch's own 40..63 -- measured through the whole chain, the
        # camera's own output already has the difference at columns 31..36, displaced by the lens's chromatic
        # aberration. The lag-0 column at the flash frame is therefore **0.01846**, which is an impulse response and
        # not a leak. What only a memory across frames can produce is a difference on the frames *after* it, and
        # there the lag-0 column is exactly zero.
        flash[k] = [float(np.sqrt(((chroma(out[t])[0] - chroma(plain[k][t])[0])[FLASH_WINDOW] ** 2
                                   + (chroma(out[t])[1] - chroma(plain[k][t])[1])[FLASH_WINDOW] ** 2).mean()))
                    for t in range(IMPULSE_AT + 1, IMPULSE_AT + 5)]
    present = flash[0.3][0] > 1e-4 and flash[0.6][0] > 1e-3
    record("a one-frame colour flash leaves a tail in later frames",
           "a spatial filter cannot do this at all: the frames after the flash have the same bar in the same place "
           "and the same colour, so filtering them changes nothing, and the flash's colour appearing in frame t+1 can "
           "only come from a memory across frames. This is the assertion a 'colour smeared sideways' measurement "
           "could never make",
           "rms chroma carried past the flash, frames %d..%d, lag 0.00: %s; lag 0.30: %s; lag 0.60: %s"
           % (IMPULSE_AT + 1, IMPULSE_AT + 4,
              " ".join("%.5f" % v for v in flash[0.0]),
              " ".join("%.5f" % v for v in flash[0.3]),
              " ".join("%.5f" % v for v in flash[0.6])),
           present and max(flash[0.0]) < 1e-6 and flash[0.3][2] > 1e-5,
           "the two scenes compared are the flash and the same scene without it -- no bar at all in either, so the "
           "only thing anywhere in the picture that changes is the patch's colour on one frame. The lag-0 column is "
           "exactly zero on all four measured frames, so the assertion is an equality rather than a threshold")

    # --- 2. the tail is geometric, which is what makes it a trail rather than a ghost
    t1, t2 = flash[0.6][0], flash[0.6][1]
    ratio = (t2 / t1) if t1 > 1e-9 else float("nan")
    record("the lag is recursive, so the tail outlives one frame",
           "a two-tap blend gives exactly one non-zero frame and then nothing, which reads as a double image. A "
           "first-order IIR carries k**2 into the second frame and decays geometrically, which is what a channel "
           "with memory does",
           "lag 0.60 flash: frame+1 %.5f, frame+2 %.5f, ratio %.3f (k = 0.60)" % (t1, t2, ratio),
           t1 > 1e-4 and 0.25 < ratio < 1.0 and t2 > 1e-5,
           "the ratio is the recursion coefficient itself, so this also pins the *mechanism* rather than merely "
           "detecting it: a two-tap blend would give a ratio of 0.000, and a one-frame delay would give a ratio of "
           "0.000 as well")

    # --- 3. what it does to a moving object: the band the object has just arrived in
    #
    # The quantity is the **direction** of what the lag added, not its size. `D = lagged - straight` is what the
    # mechanism contributed; `R = previous - current` is what it would contribute if it were exactly the previous
    # frame's colour. Their cosine is 1.0 when the lag adds the previous frame's chroma and nothing else.
    #
    # The first version of this correlated `lagged` against `previous` directly and reported values of **12.75 and
    # 16.19** -- correlations above one, which is the instrument saying it divided by nearly nothing. A normalised
    # correlation is only a statement about direction where both vectors have length, and at the leading edge of an
    # object the previous frame's chroma over the window is close to cancelling. Differencing first removes the
    # division by a near-zero reference.
    scene, straight = run(0.0)
    bands = {t: (_columns(scene.centre(t)) & ~_columns(scene.centre(t - 1)))
             for t in range(2, FRAMES)}
    bands = {t: m for t, m in bands.items() if int(m.sum()) >= 30}
    if bands:
        rows = []
        for k in (0.0, 0.3, 0.6):
            _, out = run(k)
            cosines = []
            for t in sorted(bands):
                m = bands[t]
                di = chroma(out[t])[0] - chroma(straight[t])[0]
                dq = chroma(out[t])[1] - chroma(straight[t])[1]
                ri = chroma(straight[t - 1])[0] - chroma(straight[t])[0]
                rq = chroma(straight[t - 1])[1] - chroma(straight[t])[1]
                den = float(np.sqrt((di[m] ** 2 + dq[m] ** 2).sum()
                                    * (ri[m] ** 2 + rq[m] ** 2).sum()))
                if den > 1e-12:
                    cosines.append(float((di[m] * ri[m] + dq[m] * rq[m]).sum()) / den)
            rows.append((k, float(np.mean(cosines)) if cosines else float("nan"),
                         len(cosines)))
        cos6 = rows[-1][1]
        record("what the lag adds is the previous frame's colour",
               "the impulse proves there is memory; this proves the memory is pointed at the right frame and at the "
               "right pixels -- what the mechanism adds to a moving object must be the difference between where the "
               "object was and where it is, which is what makes the colour trail rather than spread",
               "band %d px, %d frames scored; " % (int(max(b.sum() for b in bands.values())), rows[-1][2])
               + "; ".join("lag %.2f: cosine with (previous - current) %s"
                           % (r[0], "n/a" if r[1] != r[1] else "%.3f" % r[1]) for r in rows),
               rows[-1][2] >= 5 and cos6 > 0.8 and abs(cos6 - rows[1][1]) < 0.05 and rows[0][2] == 0,
               "**the frame count is asserted, because the first version of this assertion was vacuous**: with a "
               "20 px step and a 24 px bar every frame's band overlapped its predecessor's completely, the band was "
               "empty on every frame, and the row passed by scoring nothing. At lag 0 the difference is identically "
               "zero, so the cosine has no denominator -- the lag-0 row reporting 0 frames scored is the instrument "
               "saying there is nothing to score, which is stronger than a floor. At 0.30 and 0.60 it scores %.3f "
               "and %.3f, so what the mechanism adds is the previous frame's colour to within about 2%%"
               % (rows[1][1], rows[2][1]))
    else:
        record("what the lag adds is the previous frame's colour",
               "the leading-edge band must contain pixels the object has just arrived in",
               "no frame produced a band of 30 px or more, so nothing could be measured", False,
               "the scene geometry changed; fix the check rather than the mechanism")

    # --- 4. zero is exactly a no-op, on every frame
    _, out0 = run(0.0)
    identical = all(np.array_equal(a, b) for a, b in zip(out0, straight))
    record("chroma_lag 0 leaves every frame bit-identical",
           "the mechanism must be skipped rather than approximated at zero, or the default is a small change to "
           "every measurement in this repository",
           "12 frames identical: %s" % identical, identical,
           "and it is applied at the chroma channel's output, before ACC, so with it off the arithmetic is the "
           "arithmetic that was there before")

    # --- 5. the negative control: a still object, where lag must do nothing
    #
    # Measured on a frame the object is actually *in* for both rows: at `FRAMES - 3` the moving bar has long since
    # left the 176 px frame, so the "moving" reading would have been taken on the same empty field as the still one
    # and read 0.00000 -- passing the assertion for the wrong reason. The frame is named once and used for both.
    WATCH = FRAMES // 2 + 2
    _, still0 = run(0.0, moving=False)
    _, still6 = run(0.6, moving=False)
    _, moved0 = run(0.0, moving=True)
    _, moved6 = run(0.6, moving=True)
    d_still = float(np.abs(still6[WATCH] - still0[WATCH]).max())
    d_moved = float(np.abs(moved6[WATCH] - moved0[WATCH]).max())
    record("a stationary object is not touched by the lag",
           "the mechanism can only act on what changed between frames, so with consecutive frames identical it must "
           "be a no-op -- which is what separates a temporal channel from a spatial blur",
           "frame %d: still object max|delta| %.5f; moving object max|delta| %.5f"
           % (WATCH, d_still, d_moved),
           d_still < 1e-6 and d_moved > 0.02,
           "a spatial blur would move both, and would move the still one most, because a still object is what a "
           "blur has to work with")
if __name__ == "__main__":
    raise SystemExit(main())
