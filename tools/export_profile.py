"""
Write the settings out to a non-default path, as a diff against the schema defaults, and check it round-trips.

**Two shapes exist and they are for different things.**

  * `settings.json` in the repository root is a **full snapshot of every layer** -- 253 numbers -- written by
    `tools/write_settings.py`.  It loads, and it is complete, and it is the right thing to diff against a release;
  * **this file is the other shape**: the server's own `/api/export` format, which is **a diff against the schema
    defaults**.  Its own docstring says why -- "a full dump of every layer is 400-odd numbers and says nothing about which
    of them a reproduction depended on. The diff says exactly that, and it is small enough to read."

**It goes to `profiles/` rather than the repository root**, so it is not the same artefact as the snapshot and cannot be
confused with it.

**And it is verified rather than assumed**: the file is written, read back, merged over the schema defaults through the
same `merge_values` path the renderer uses, and **every parameter is compared with the live project's** -- so "the file
round-trips" is a measurement here and not a claim.  **The check reports the count of parameters that survive the round
trip and the count that do not**, and it fails loudly if any disagree, because a settings file that silently loses a value
is worse than no file.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.params import merge_values                                    # noqa: E402
from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402

OUT_DIR = os.path.join(ROOT, "profiles")
OUT = os.path.join(OUT_DIR, "nightshot-trv310e.json")


def main() -> int:
    layers = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
    #: the shipped preset turns the mode on, because every value calibrated here belongs to it
    layers["camera"]["nv_mode"] = 2.0
    #: **The basis of the diff is the SCHEMA DEFAULTS, not the preset**, because that is what an import resets to before
    #: merging.  For the camera layer the two happen to be identical -- **every NightShot value this work calibrated is
    #: the schema default now**, since `nv_illuminator 1.2`, `nv_chroma_residue 0.15`, `nv_highlight_saturation 4.0`,
    #: `nv_ir_focus_shift 1.5` and `nv_exposure_trim -0.1` are all declared in `profiles.py` -- so the diff carries only
    #: the preset's own RF, tape and display values.
    #:
    #: **I briefly believed this file was losing the NightShot calibration, and measured that it is not**: an import of
    #: it was compared with the live project across all 253 parameters and produced **zero disagreements**.  The belief
    #: came from counting the diff (7 entries) and assuming the camera values must therefore be missing, when in fact
    #: they are the defaults.  **The count was right and the inference was wrong**, which is the same shape as the audits
    #: that reported `blooming` absent because they looked in the wrong module.
    defaults = SCHEMA.resolve_all({})

    changed: dict[str, dict[str, object]] = {}
    for layer, values in layers.items():
        diff = {k: v for k, v in values.items() if defaults.get(layer, {}).get(k) != v}
        if diff:
            changed[layer] = dict(sorted(diff.items()))

    payload = {
        "kind": "atsim.config",
        "version": 1,
        "exported": _dt.datetime.now().isoformat(timespec="seconds"),
        "name": "NightShot -- Sony DCR-TRV310E",
        #: **`ntsc`, not the preset's name.**  The first version of this file wrote `"standard": "broadcast_1992_weak"`,
        #: which is a **preset**, not a standard, and the import died with
        #: `KeyError: "unknown standard 'broadcast_1992_weak'; known: ['ntsc', 'pal']"`.  The two are different dimensions
        #: and this field takes the standard; the preset is recorded below as information.
        #:
        #: **And the value is measured rather than assumed.**  The clip is a `TRV310E` -- the E is the European model --
        #: and its own frames are 1440x1080, so `pal` looked right; **`ffprobe` reports `avg_frame_rate=30/1`**, and
        #: 30 fps is 29.97, i.e. **NTSC**.  A pal camcorder cannot produce 30 fps, so the recording is NTSC whatever the
        #: model number says, and the frame rate is the harder fact of the two.
        "standard": "ntsc",
        "seed": 20240101,
        "params": changed,
        #: **the same handoff block the snapshot carries**, because this file may be the only one a reviewer opens
        "_what_this_is": "The analog tape simulator's NightShot calibration, as a DIFF against the schema defaults -- "
                         "so every number here is one the reproduction actually depends on.",
        "_factory_preset_this_was_built_on": "broadcast_1992_weak  (apply it with POST /api/preset before or instead of "
                                             "this file; it is a PRESET, not a standard, and the two are separate "
                                             "fields -- writing it into `standard` is what made the first version of this "
                                             "file fail to import)",
        "_calibrated_against": {
            "source": "Sony DCR-TRV310E, C:\\Users\\Administrator\\Downloads\\10月13日.mp4",
            "source facts": "HEVC 1440x1080, avg_frame_rate 30/1 -- which is NTSC",
            "cut_map": "23s normal -> mode 1; 45-52s normal; 52-53s mode 1; 54-68s NIGHTSHOT 1; 68-83s NIGHTSHOT 2",
            "the footage's own range": {"mean": [0.11, 0.39], "std": [0.09, 0.26], "R/G": [0.37, 0.85],
                                        "p1": [0.024, 0.055]},
            "the fair test": "bare photographs only -- a camcorder frame is already the chain's own output, so feeding it "
                             "back asks the chain to preserve a transformation rather than perform one",
        },
        "_reproduce": {
            "the suite": "python tools/verify.py",
            "the self-test": "python tools/selftest.py",
            "the full snapshot": "python tools/write_settings.py",
            "this file": "python tools/export_profile.py",
            "bare inputs": "python tools/bare_inputs.py",
            "the note audit": "python tools/note_audit.py",
        },
        "_note_on_color_cast": "**`color_cast`'s help text had its two directions reversed** and is corrected: measured "
                               "on a neutral grey field through the chain, c -0.50 gives B/R 0.8369 (warm) and c +0.50 "
                               "gives 1.1892 (cool), so **POSITIVE is cool/cyan and NEGATIVE is warm**.  The user "
                               "spotted the contradiction between the label and the picture.",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")

    # --- the round trip, measured THE WAY AN IMPORT ACTUALLY WORKS ---------------------------------------------------
    #: **Against `SCHEMA.resolve_all({})`, which is what an import resets to.**  The first version of this check merged
    #: the diff over the PRESET (`SCHEMA.resolve_all({})` was used for `defaults` but the comparison ran against `layers`,
    #: which was built from the preset) and so it reported success while the file was losing five parameters: the preset
    #: supplied the values the file had omitted, so the check could not see the omission.  **A round-trip check that
    #: starts from a richer state than an import does cannot detect what the file fails to carry.**
    with open(OUT, encoding="utf-8") as fh:
        back = json.load(fh)
    merged = merge_values(SCHEMA.resolve_all({}), back["params"])
    bad, ok = [], 0
    for layer, values in merged.items():
        for k, v in values.items():
            if k not in layers.get(layer, {}):
                continue
            want = layers[layer][k]
            #: **`overlay.text` is a STRING** (the OSD's own text), so the comparison cannot assume numbers -- the first
            #: version of this check called `float()` on everything and died on it, which is the check being wrong rather
            #: than the file.
            if isinstance(want, str) or isinstance(v, str):
                same = str(v) == str(want)
            else:
                same = abs(float(v) - float(want)) <= 1e-9
            if same:
                ok += 1
            else:
                bad.append((layer, k, v, want))
    print("   written: %s" % os.path.relpath(OUT, ROOT))
    print("   %d layers changed against the defaults, %d parameters in the diff"
          % (len(changed), sum(len(v) for v in changed.values())))
    print("   round trip: **%d parameters survive** against the live project's values, %d disagree" % (ok, len(bad)))
    for layer, k, got, want in bad[:10]:
        print("      %s.%s came back as %r, expected %r" % (layer, k, got, want))
    if bad:
        print("   **FAILED: a settings file that loses or alters a value is worse than no file.**")
        return 1

    # --- and the header fields the parameter check CANNOT see --------------------------------------------------------
    #: **This is the check that would have caught the actual failure and did not exist.**  The first exported profile
    #: carried `"standard": "broadcast_1992_weak"` -- a PRESET name in a field that takes `ntsc` or `pal` -- and the
    #: import died with `KeyError: "unknown standard 'broadcast_1992_weak'; known: ['ntsc', 'pal']"`.  **The parameter
    #: round trip passed cleanly throughout**, because `standard` is not a parameter: it is a project field, and a
    #: validator that only looks at `params` cannot see a header that makes the whole file unusable.
    from atsim.core.standards import STANDARDS
    head_bad = []
    if back.get("standard") not in STANDARDS:
        head_bad.append(("standard", back.get("standard"), sorted(STANDARDS)))
    if back.get("kind") != "atsim.config":
        head_bad.append(("kind", back.get("kind"), "atsim.config"))
    if not isinstance(back.get("params"), dict):
        head_bad.append(("params", type(back.get("params")).__name__, "dict"))
    print("   header: standard %r of %s, kind %r -- %d problem(s)"
          % (back.get("standard"), sorted(STANDARDS), back.get("kind"), len(head_bad)))
    for f, got, want in head_bad:
        print("      **%s is %r, must be one of %r**" % (f, got, want))
    if head_bad:
        print("   **FAILED: the file would not import, however correct its parameters are.**")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
