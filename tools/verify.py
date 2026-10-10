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
    want = {"camera", "format", "codec", "rf", "tape", "player", "display"}
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

    check_consistency()
    check_engine()
    check_presets()
    check_shuttle()
    if not args.quick:
        check_drum_speed()
        check_acc_gain()
        check_binder_loss()
        check_rf()
        check_deck_osd()
        check_graph_wiring()
        check_preset_independence()
        check_broadcast_noise()
        check_head_switch()

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


if __name__ == "__main__":
    raise SystemExit(main())
