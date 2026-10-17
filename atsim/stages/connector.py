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
                                 degrade_rf, degrade_rf_signal, degrade_svideo,
                                 impedance, reflection_coefficient)
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame


class ConnectorStage(Stage):
    """Twelve paragraphs of physics, one stage."""

    name = "connector"
    layers = ("connector",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self.state: dict = {}
        self._read_params()

    def _read_params(self) -> None:
        c = self.cfg_of("connector")

        def f(key: str, default: float = 0.0) -> float:
            return float(c.get(key, default))

        # which links are present, and their condition
        self.hdmi_on = f("hdmi_present", 0.0) > 0.5
        self.av_on = f("av_present", 0.0) > 0.5
        self.rf_on = f("rf_present", 0.0) > 0.5
        self.svideo_on = f("svideo_present", 0.0) > 0.5
        self.aspect_squeeze = float(np.clip(f("aspect_squeeze", 0.0), 0.0, 1.5))

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

    def on_params_changed(self) -> None:
        self._read_params()

    # ------------------------------------------------------------------
    def _stats(self, spec, moisture: float, corrosion: float) -> dict:
        z = impedance(spec, moisture, corrosion)
        return {"impedance_ohm": round(z, 1),
                "reflection": round(abs(reflection_coefficient(spec.nominal_z, z)), 4)}

    @staticmethod
    def _apply_aspect_squeeze(rgb: np.ndarray, squeeze: float) -> tuple[np.ndarray, dict]:
        """
        Double horizontal aspect-ratio compression (`16:9 STB -> 4:3 VCR`: `x_scale = (3/4)^squeeze`).
        """
        s = float(np.clip(squeeze, 0.0, 1.5))
        if s <= 1e-4:
            return rgb, {}
        h, w = rgb.shape[:2]
        ratio = float(0.75 ** s)
        active_w = max(4.0, w * ratio)
        pad_left = 0.5 * (w - active_w)
        cols = np.arange(w, dtype=np.float32)
        src_x = (cols - pad_left) / max(1e-6, ratio)
        src_c = np.clip(src_x, 0.0, w - 1.001)
        x0 = np.floor(src_c).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        fx = (src_c - x0)[None, :, None].astype(np.float32)
        squeezed = rgb[:, x0] * (1.0 - fx) + rgb[:, x1] * fx

        # Smooth 1.8-pixel analog blanking transition at the left and right pillarbox boundaries
        scale_w = w / 720.0
        edge_px = max(0.8, 1.8 * scale_w)
        gate_l = np.clip((cols - pad_left) / edge_px, 0.0, 1.0)
        gate_r = np.clip(((w - pad_left) - cols) / edge_px, 0.0, 1.0)
        gate = (gate_l * gate_r)[None, :, None].astype(np.float32)
        pedestal = np.full_like(squeezed, 0.018)
        out = squeezed * gate + pedestal * (1.0 - gate)
        return out.astype(np.float32), {
            "aspect_squeeze": round(s, 3),
            "width_ratio": round(ratio, 4),
            "pillarbox_px": int(round(pad_left)),
        }

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        active = self.hdmi_on or self.av_on or self.svideo_on or self.rf_on or (self.aspect_squeeze > 1e-4)
        if not active:
            return buf

        meta = dict(buf.meta)
        report: dict = {}
        rgb = buf.rgb
        comp = buf.composite
        sig: SignalFrame | None = meta.get("signal")

        # ---- 0. STB 16:9 -> HDMI-AV -> 4:3 VCR horizontal aspect-ratio compression
        if self.aspect_squeeze > 1e-4:
            from ..core.signal import composite_encode
            base_in = sig.to_rgb() if sig is not None else np.clip(rgb, 0.0, 1.0).astype(np.float32)
            rgb, sq_info = self._apply_aspect_squeeze(base_in, self.aspect_squeeze)
            report["aspect"] = sq_info
            sig = SignalFrame.from_rgb(rgb, buf.std)
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)

        # ---- 1. HDMI: digital, and the only one that can drop the link entirely
        if self.hdmi_on:
            spec = CONNECTORS["hdmi"]
            rgb, info = degrade_digital(rgb, spec, self.hdmi_moist, self.hdmi_corr,
                                       buf.index, self.ctx.seed + 11)
            report["hdmi"] = info
            if info.get("link") == "no signal":
                meta["signal"] = None
                meta["format"] = "no_signal"
                meta["connector"] = report
                return FrameBuffer(rgb, None, buf.index, buf.time, buf.std, meta)
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
                from ..core.signal import composite_encode
                comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            if comp is not None:
                comp, info = degrade_composite(comp, spec, self.av_moist, self.av_corr,
                                               self.ground_loop, self.mains_hz, buf.std,
                                               buf.index)
                report["av"] = info

        from ..core.signal import composite_encode, demod_chroma, split_composite

        if comp is not None and self.av_on and not self.rf_on:
            y, chroma = split_composite(comp, buf.std, comb_depth=0.0,
                                        chroma_bandwidth_hz=0.9e6)
            i, q = demod_chroma(chroma / 0.63, buf.std, i_bw_hz=1.2e6, q_bw_hz=1.2e6)
            sig = SignalFrame(y, i, q)

        # ---- 4. RF: the modulated carrier (HDMI -> AV -> RF -> HDMI loop or VHF CATV -> VCR)
        if self.rf_on:
            spec = CONNECTORS["rf"]
            base_rgb = sig.to_rgb() if sig is not None else np.clip(rgb, 0.0, 1.0)
            rgb_rf, info = degrade_rf_signal(base_rgb, spec, self.rf_moist, self.rf_corr,
                                             self.shield, buf.index, self.ctx.seed + 23)
            report["rf"] = info
            sig = SignalFrame.from_rgb(rgb_rf, buf.std)
            rgb = rgb_rf
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            if self.rf_moist > 0.08 or self.rf_corr > 0.08:
                sig._rf_direct = True
        elif self.hdmi_on or self.svideo_on:
            sig = SignalFrame.from_rgb(np.clip(rgb, 0.0, 1.0), buf.std)

        if sig is not None:
            rgb = np.clip(sig.to_rgb(), 0.0, 1.0).astype(np.float32)
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)

        meta["signal"] = sig
        meta["connector"] = report
        return FrameBuffer(rgb, comp, buf.index, buf.time, buf.std, meta)

    def describe(self) -> dict:
        return dict(self.state)
