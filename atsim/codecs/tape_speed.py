"""
Tape speed: SP / LP / EP physics.

This is not "turn the quality down".  On a VHS machine the *drum* keeps spinning
at 1500 rpm, so the head-to-tape speed during recording -- and therefore the FM
deviation the video heads can lay down -- is essentially unchanged between
speeds.  What changes is the **linear tape speed**:

    SP   1x         track width nominal (e.g. 58 um on a 120-min cassette)
    LP   1/2x       track width ~1/2, recording wavelength halved
    EP   1/3x       track width ~1/3, recording wavelength thirded

Everything that visibly degrades in LP/EP follows from the narrower track and the
longer wavelength, not from a lumped "worse" knob:

1. **Reproduced output falls.**  The replay head's output is proportional to the
   rate of change of flux, i.e. to the recorded frequency; halving the tape speed
   roughly halves it.  The luminance FM carrier also shifts down (about
   3.8-4.8 MHz at SP, 2.3-2.9 at LP, 1.6-2.0 at EP), so the signal sits lower on
   the head-response curve.  Net effect at LP is around -6 dB, at EP around
   -10 dB of SNR.

2. **Dropouts get more likely.**  A narrower track is proportionally more
   affected by the same particle or scratch, so the *rate* of visible dropouts
   rises roughly with the speed ratio.

3. **Tracking becomes far more sensitive.**  The guard bands between tracks are
   correspondingly narrow, so a mistracked capstan cuts into neighbouring tracks
   instead of empty guard band.  This is why LP/EP material shows field-rate
   horizontal banding and colour corruption that SP hides.

4. **Time-base stability worsens**, because the control-track signal the capstan
   servo follows is recorded at the lower speed too.

Note what does NOT change: the colour-under recording scheme.  Chroma is still
heterodyned to ~629 kHz and the colour bandwidth is set by that carrier's
deviation, not by the tape speed.  That is why the reference "abnormal" capture
keeps its luminance level while the chroma bands and stability degrade -- the
model reproduces that asymmetry rather than lowering every parameter together.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedProfile:
    """The physical consequences of one recording speed."""
    key: str
    label: str
    ratio: float              # linear tape speed, relative to SP
    track_ratio: float        # track width relative to SP
    snr_loss_db: float        # reproduced-output loss
    dropout_scale: float      # relative probability of a visible dropout
    tracking_scale: float     # sensitivity to capstan/tracking error
    stability_scale: float    # time-base instability multiplier
    fm_carrier_mhz: tuple[float, float]   # luminance FM deviation range

    def note(self) -> str:
        return (f"走带速度 {self.ratio:.2f}×，磁迹宽度约 {self.track_ratio:.2f}×；"
                f"重放输出约 −{self.snr_loss_db:.0f} dB，失落概率约 "
                f"{self.dropout_scale:.1f}×，跟踪敏感度约 {self.tracking_scale:.1f}×；"
                f"亮度 FM 载波 {self.fm_carrier_mhz[0]:.1f}–{self.fm_carrier_mhz[1]:.1f} MHz")


# Track widths and SNR figures follow the VHS specification's speed ratios.
# SNR loss is expressed relative to SP at the same tape, not as an absolute grade.
SPEEDS: dict[str, SpeedProfile] = {
    "sp": SpeedProfile(
        key="sp", label="SP 标准播放", ratio=1.0, track_ratio=1.0,
        snr_loss_db=0.0, dropout_scale=1.0, tracking_scale=1.0, stability_scale=1.0,
        fm_carrier_mhz=(3.8, 4.8),
    ),
    "lp": SpeedProfile(
        key="lp", label="LP 长时播放", ratio=0.5, track_ratio=0.5,
        snr_loss_db=6.0, dropout_scale=2.0, tracking_scale=2.2, stability_scale=1.9,
        fm_carrier_mhz=(2.3, 2.9),
    ),
    "ep": SpeedProfile(
        key="ep", label="EP 超长播放（SLP）", ratio=1.0 / 3.0, track_ratio=1.0 / 3.0,
        snr_loss_db=10.0, dropout_scale=3.0, tracking_scale=3.4, stability_scale=2.8,
        fm_carrier_mhz=(1.6, 2.0),
    ),
}

ORDER: tuple[str, ...] = ("sp", "lp", "ep")


def by_index(value: float) -> SpeedProfile:
    """
    Map a 0/1/2 control onto a speed profile.

    The schema expresses switches as numbers so they clamp and serialise like
    every other parameter; the mapping lives here so the UI, the executor and the
    tests all agree on it.
    """
    idx = int(round(float(value)))
    idx = max(0, min(len(ORDER) - 1, idx))
    return SPEEDS[ORDER[idx]]


def speed_from_key(key: str) -> SpeedProfile:
    return SPEEDS.get(str(key).lower(), SPEEDS["sp"])


@dataclass
class SpeedEffects:
    """Concrete multipliers a speed imposes on the medium model."""
    profile: SpeedProfile
    snr_multiplier: float          # multiplies noise_floor / dropout visibility
    dropout_multiplier: float
    tracking_multiplier: float
    stability_multiplier: float

    def summary(self) -> str:
        return self.profile.label
