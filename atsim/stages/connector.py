"""
Stage 02c -- the connectors the signal passes through on its way to the recorder.

Placed after the codec and before the medium, because that is the physical order: the
source produces a signal, cables and connectors carry it, and the recorder stores whatever
arrives. Damage here is therefore *recorded* -- it becomes part of the tape, and playing
that tape back later reproduces it. That distinction is the whole reason this is a separate
stage rather than more tape parameters: tape damage is the medium decaying, connector damage
is the infrastructure decaying, and the second is written into the first.

Up to three connectors in series, applied in the order the signal meets them:

    HDMI (digital, to the AV converter)
      -> AV (composite, to the modulator)
        -> RF (modulated carrier, to the recorder)

Each is skipped unless it is switched on, and each reports its own electrical state, so a
preset can say *where* a fault is rather than only how bad the picture looks.
"""
from __future__ import annotations

import numpy as np

from ..codecs.connectors import (CONNECTORS, degrade_composite, degrade_digital,
                                 degrade_rf, degrade_svideo, impedance,
                                 reflection_coefficient)
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame


class ConnectorStage(Stage):
    """Twelve paragraphs of physics, one stage."""

    name = "connector"
    layers = ("connector",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        c = self.cfg_of("connector")

        def f(key: str, default: float = 0.0) -> float:
            return float(c.get(key, default))

        # which links are present, and their condition
        self.hdmi_on = f("hdmi_present", 0.0) > 0.5
        self.av_on = f("av_present", 0.0) > 0.5
        self.rf_on = f("rf_present", 0.0) > 0.5
        self.svideo_on = f("svideo_present", 0.0) > 0.5

        # moisture and corrosion are per-connector, because in practice one joint in a
        # chain is wet and the others are not, and that is what makes a fault locatable
        self.hdmi_moist, self.hdmi_corr = f("hdmi_moisture"), f("hdmi_corrosion")
        self.av_moist, self.av_corr = f("av_moisture"), f("av_corrosion")
        self.rf_moist, self.rf_corr = f("rf_moisture"), f("rf_corrosion")
        self.sv_moist, self.sv_corr = f("svideo_moisture"), f("svideo_corrosion")

        self.ground_loop = f("av_ground_loop")
        self.mains_hz = f("mains_hz", 50.0)
        self.shield = float(np.clip(f("rf_shielding", 0.0), 0.0, 1.0))
        self.leakage = f("svideo_leakage")
        self.state: dict = {}

    # ------------------------------------------------------------------
    def _stats(self, spec, moisture: float, corrosion: float) -> dict:
        z = impedance(spec, moisture, corrosion)
        return {"impedance_ohm": round(z, 1),
                "reflection": round(abs(reflection_coefficient(spec.nominal_z, z)), 4)}

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        active = self.hdmi_on or self.av_on or self.svideo_on or self.rf_on
        if not active:
            return buf

        meta = dict(buf.meta)
        report: dict = {}
        rgb = buf.rgb
        comp = buf.composite
        sig: SignalFrame | None = meta.get("signal")

        # ---- 1. HDMI: digital, and the only one that can drop the link entirely
        if self.hdmi_on:
            spec = CONNECTORS["hdmi"]
            rgb, info = degrade_digital(rgb, spec, self.hdmi_moist, self.hdmi_corr,
                                       buf.index, self.ctx.seed + 11)
            report["hdmi"] = info
            if info.get("link") == "no signal":
                # the link failed: nothing downstream receives anything, which is the
                # honest consequence rather than a degraded picture
                meta["signal"] = None
                meta["format"] = "no_signal"
                meta["connector"] = report
                return FrameBuffer(rgb, None, buf.index, buf.time, buf.std, meta)
            # a digital link that survives delivers exact data, so the picture is
            # *reconstructed* from what arrived rather than damaged further
            if sig is not None:
                sig = SignalFrame.from_rgb(np.clip(rgb, 0, 1), buf.std)

        # ---- 2. S-Video: separate luma and chroma, leakage between the pins
        if self.svideo_on and sig is not None:
            spec = CONNECTORS["svideo"]
            y, i, q, info = degrade_svideo(sig.y, sig.i, sig.q, spec, self.sv_moist,
                                           self.sv_corr, self.leakage)
            sig = SignalFrame(y, i, q)
            report["svideo"] = info

        # ---- 3. AV: baseband composite
        if self.av_on:
            spec = CONNECTORS["av"]
            if sig is not None:
                # the signal is composite by this point in a real AV chain; encode it so the
                # connector's mechanisms act where they physically act
                from ..core.signal import composite_encode
                comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            if comp is not None:
                comp, info = degrade_composite(comp, spec, self.av_moist, self.av_corr,
                                               self.ground_loop, self.mains_hz, buf.std,
                                               buf.index)
                report["av"] = info

        # ---- 4. RF: the modulated carrier
        if self.rf_on and comp is not None:
            spec = CONNECTORS["rf"]
            comp, info = degrade_rf(comp, spec, self.rf_moist, self.rf_corr, self.shield,
                                    buf.index, self.ctx.seed + 23)
            report["rf"] = info

        # Everything the connectors did has to be written into *one* representation that the
        # downstream stages actually read, or some of it is silently discarded.
        #
        # Measured: a change to the rgb buffer survived the tape stage and was gone after the
        # player, whose output is rebuilt from the composite plane. And a change to the
        # composite was invisible to the digital path, which reads `meta["signal"]`. So the
        # component view is rebuilt from whatever the connectors produced, and the composite
        # is re-encoded from *that* -- one signal, two views of it, rather than two signals
        # that can disagree.
        #
        # This is the same failure as the deck's blue screen being rebuilt from the tape
        # signal it had rejected: a stage acting on one representation while a later stage
        # reads the other.
        from ..core.signal import composite_encode, demod_chroma, split_composite

        if comp is not None and (self.av_on or self.rf_on):
            # the analogue connectors act on the composite, so it is authoritative for them
            y, chroma = split_composite(comp, buf.std, comb_depth=0.0,
                                        chroma_bandwidth_hz=0.9e6)
            i, q = demod_chroma(chroma / 0.63, buf.std, i_bw_hz=1.2e6, q_bw_hz=1.2e6)
            sig = SignalFrame(y, i, q)
        elif self.hdmi_on or self.svideo_on:
            sig = SignalFrame.from_rgb(np.clip(rgb, 0.0, 1.0), buf.std)

        if sig is not None:
            # keep the rgb view consistent with the component view, and rebuild the
            # composite so the player's path carries the same damage
            rgb = np.clip(sig.to_rgb(), 0.0, 1.0).astype(np.float32)
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)

        meta["signal"] = sig
        meta["connector"] = report
        return FrameBuffer(rgb, comp, buf.index, buf.time, buf.std, meta)

    def describe(self) -> dict:
        return dict(self.state)
