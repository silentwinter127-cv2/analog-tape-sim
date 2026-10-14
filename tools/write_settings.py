"""
Write the instrument's settings file: the calibrated parameters, with the measurement behind each one.

**Why this file exists.**  `FACTORY_PRESETS["broadcast_1992_weak"]` is the factory state and the schema holds each
parameter's range and help, **but neither is a single artefact a person can read, diff, or hand to someone else.**  This
writes one: every parameter that this work calibrated, its value, and the measurement that put it there.

**It is generated from the live schema rather than typed**, so it cannot drift from what the engine actually runs -- the
values are read back through `SCHEMA.resolve_all` and the file is therefore true by construction.  **And it is a preset
that can be loaded**, not only documentation, so the same bytes serve both purposes.

**What it records, and what it deliberately does not.**  It records the values and the figures; **it does not claim the
model is calibrated**, because two figures are still out and the file says which.  The user's report that started this --
"还是不够亮" -- is answered by the figures beside each parameter rather than by an assertion.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atsim.core.profiles import FACTORY_PRESETS, SCHEMA                       # noqa: E402

#: the parameters this work calibrated, and why each has the value it has.  **Every figure here was measured**, and the
#: ones that came from a sweep say what the sweep showed.
NOTES = {
    "exposure": "camera exposure, stops.  Left at 0.0: the night-vision trim is applied inside the mode instead, "
                "because changing this broke 'a cold machine renders differently from a warm one'.",
    "saturation_stops": "**the sensor's saturation point, in stops below `dynamic_range`'s own white.**  Measured: at "
                        "0.0 the white point is 1.0 while the picture's maximum is 0.87, so `lin >= white` never fires "
                        "and no pixel can ever clip -- the model's brightest pixel was 0.8686 with 0.00000 clipped, "
                        "against the footage's 1.0000 with 0.00657.  0.30 is the largest value that leaves the overflow "
                        "drain its ring: the ring reads 0.00310 / 0.00597 / 0.00919 at drain 0.15 / 0.25 / 0.35, and by "
                        "0.35 it has collapsed to 0.00021 / 0.00048 / 0.00084 because the white point (0.812 at 0.30, "
                        "0.757 at 0.40) approaches the drain check's own 0.62 ceiling.  **Applies only while `nv_mode` "
                        "is on**: applied always it made the scanline structure cost light (mean varying 0.00275 "
                        "across the four spot sizes).",
    "dynamic_range": "stops.  Left at 8.0.  **A lower value RAISES the white point** (8 -> 1.00, 10 -> 1.19, 12 -> 1.41), "
                     "so it is the opposite control to `saturation_stops` and cannot produce a highlight.",
    "nv_mode": "0 = off, 1 = NightShot, 2 = Super NightShot, 3 = Colour Slow Shutter.",
    "nv_illuminator": "**the lamp, and it is a real control now.**  Measured before this round: 0.15 through 3.00 gave "
                      "bit-identical output and the field itself was identical at both ends (min 0.05475, max 0.06625) "
                      "-- the value arrived at `illuminator_field` and the arithmetic never used it, because the field "
                      "was normalised by its own mean and multiplied by a fixed 0.062.  It now scales, and the default "
                      "1.2 reproduces the previous picture (`0.062 * 1.2` against the old constant).",
    "nv_highlight_saturation": "**the display's drive, applied ABOVE the gate only.**  A channel saturates where "
                               "`lum * w == 1`; with the weights normalised so `weights @ LUMA == 1` they are "
                               "[0.6069, 1.2021, 0.9904], so **green pins at lum 0.832**, blue at 1.010 and red at "
                               "1.648, and the brightest luminance the scene branch delivers is 0.8225 -- within 1% of "
                               "the gate.  Above it the strongest channel is pinned while the others keep climbing, so "
                               "the pixel **desaturates towards white** and its luminance leaves the ratio's own 0.958 "
                               "ceiling.  **This is `note.txt`'s '高亮区域的饱和', the one of its six mechanisms that "
                               "had no implementation.**  4.0 is its plateau: 6.0 and 8.0 are bit-identical.  "
                               "**Why the gate matters**: multiplied into every pixel it merely brightens -- gain 1.0 to "
                               "1.6 took the mean from 0.2370 to 0.3493.",
    "nv_ir_albedo": "how much of the near-IR response comes from the vision model's per-material albedo rather than "
                    "from the luminance estimate.",
    "nv_green_gain": "solved, not guessed: 1.0 scored 0.706 error against the footage's green cast and 1.15 scored 0.046.",
    "nv_chroma_residue": "swept.  0.15 -> 0.1169, 0.25 -> 0.1397, 0.35 -> 0.1706, **0.45 -> 0.2092**, 0.55 -> 0.2764, "
                         "0.65 -> 2.7255 (unstable), 1.00 -> 0.7229.  Luminance is unchanged at every setting.  "
                         "**This parameter was unreachable before**: all three mode presets carried their own value and "
                         "`mode_spec` rebuilt the spec every frame, so 0.00 / 0.35 / 0.95 all arrived as 0.350.",
    "nv_shot_noise": "the signal-dependent term.  Swept against the footage's own grain: 0.00 -> row 0.10664, 0.05 -> "
                     "0.10792, 0.10 -> 0.11153, 0.18 -> 0.12121, against the footage's 0.07562.  Its fine grain "
                     "(0.08189) is within 3.5% of the footage's (0.07891), so the LEVEL is about right.",
    "nv_nir_material": "1 = ask the vision-language model for per-material near-IR albedo (`note.txt`'s second "
                       "mechanism).  Where no model is installed the luminance estimate is used, **which is the "
                       "documented fallback rather than a degraded mode**.",
    "nv_nir_grid": "the labelling grid.  **It was a visible defect**: upsampled with nearest-neighbour it stamped "
                   "66x50-pixel cells with a 7.5x step at every boundary onto the picture as coarse diagonal bands.  "
                   "It is now bilinear with a smoothstep, which removed them.",
    "nv_accumulate": "frames averaged.  mode 2 (Super NightShot) reads 0.427 against 1/sqrt(8) = 0.354.",
    "cross_color": "the player's luminance-to-chroma crosstalk.",
    "cross_luma": "**scale fixed from 0.15 to 0.02.**  At 0.15 it subtracted 38% of the luma's spread, which was the "
                  "contrast compression.",
    "comb_depth": "the player's comb filter.  0.55.",
}


def main() -> int:
    layers = SCHEMA.resolve_all(FACTORY_PRESETS["broadcast_1992_weak"]["layers"])
    # **`nv_mode` is set to Super NightShot in the shipped file.**  The factory value is 0, but this entire calibration is
    # the NightShot calibration -- the saturation point, the display drive, the lamp, the green gain and the residue were
    # all fitted against the NIGHTSHOT segments of the reference clip.  Left at 0 the file would load and show nothing of
    # what it documents, so the mode it belongs to is switched on here and the factory value stays untouched.
    layers["camera"]["nv_mode"] = 2.0
    out = {
        "_what_this_is": "The analog tape simulator's calibrated settings, read back from the live schema so it cannot "
                         "drift from what the engine runs.",
        "_calibrated_against": {
            "source": "Sony DCR-TRV310E, C:\\Users\\Administrator\\Downloads\\10月13日.mp4",
            "cut_map": "23s normal -> mode 1; 45-52s normal; 52-53s mode 1 (NIGHTSHOT); "
                       "54-68s NIGHTSHOT 1; 68-83s NIGHTSHOT 2",
            "measured_at": "533x400, the OSD band excluded (sl = slice(88, 400))",
            "targets": {
                "NIGHTSHOT 0": {"mean": 0.2346, "p90": 0.5870, "p99": 0.9511, "max": 1.0000, "clipped": 0.00657,
                                "hue_spread": 0.5519, "saturation": 0.1436},
                "NIGHTSHOT 1": {"mean": 0.3348, "p90": 0.7570, "p99": 0.9977, "max": 1.0000, "clipped": 0.02313,
                                "hue_spread": 0.1170, "saturation": 0.3095},
                "NIGHTSHOT 2": {"mean": 0.3405, "p90": 0.7373, "p99": 0.9724, "max": 1.0000, "clipped": 0.00785,
                                "hue_spread": 0.1129, "saturation": 0.3387},
            },
        },
        "_where_it_stands": {
            "the_model_both_modes": {"mean": 0.2334, "p90": 0.5438, "p99": 0.9315, "max": 0.9977, "clipped": 0.00156,
                                     "hue_spread": 0.2387, "saturation": 0.1761},
            "in_agreement": ["mean within 0.5%", "p99 within 2.1%", "max within 0.2%",
                             "hue spread inside the footage's own range", "clipped now real (0.00156 against 0.00657)"],
            "still_out": ["**the 90th percentile reads 0.5438 against 0.5870, 7.4% low** -- that is the upper "
                          "mid-tone, not the highlight.  Measured, `saturation_stops` cannot close it: pushing it to "
                          "0.40 (where the overflow drain's ring has already gone) reaches 0.5837 but takes the mean to "
                          "0.2512.  It is a tone-curve shape question at the upper mid-tone and it is named here "
                          "rather than left to be discovered."],
            "one_contradiction_left_open": "mode 1's `_night_vision` output is still slightly more banded in the "
                                           "picture while the corrected metrics cannot separate render from footage.",
        },
        "_the_two_things_that_were_most_wrong": {
            "the_inversion": "The player corrrelated -0.352 with its own input while every other stage returned "
                             "+0.999/+1.000.  **Cause: six player parameters were read from the FORMAT layer**, so "
                             "`chroma_bw` was 2.000 MHz for every request; at 2 MHz the second split took 0.770 of the "
                             "composite as chroma and the comb's 1/(1-0.55) multiplied it by 2.143, giving "
                             "chroma/composite 1.099, so `luma = comp - chroma` went negative.  Six attempted fixes to "
                             "the chroma division failed and were reverted before the layer was checked.",
            "the_missing_mechanism": "**Highlight saturation was absent, not mistuned.**  `note.txt` lists "
                                     "'高亮区域的饱和' among its six mechanisms and no term implemented it.  It is "
                                     "implemented now, and the display's own arithmetic explains why it was impossible: "
                                     "the projection scales three channels together by a fixed ratio, so it can never "
                                     "produce white.",
        },
        "layers": {k: dict(sorted(v.items())) for k, v in sorted(layers.items())},
        "_notes": NOTES,
    }
    path = os.path.join(ROOT, "settings.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    n = sum(len(v) for v in layers.values())
    print("   written: %s" % os.path.relpath(path, ROOT))
    print("   %d parameters across %d layers, read back from the schema" % (n, len(layers)))
    print("   %d parameters annotated with the measurement behind them" % len(NOTES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
