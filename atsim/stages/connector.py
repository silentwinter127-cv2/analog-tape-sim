"""
Stage 02c -- the connectors and RF / TV-DX broadcast link on the way to the recorder.

Placed after the codec and before the medium, because that is the physical order: the
source produces a signal, cables/connectors or an RF broadcast channel (such as a
TV-DX atmospheric duct or Es ionospheric skywave link captured on a cheap Taobao TV
tuner card) carry it, and the recorder/capture chain stores whatever arrives.
"""
from __future__ import annotations

import numpy as np

from ..codecs.connectors import (CONNECTORS, degrade_composite, degrade_digital,
                                 degrade_rf, degrade_rf_signal, degrade_svideo,
                                 impedance, reflection_coefficient)
from ..codecs.rf import RFLink, RFParams
from ..codecs.switcher import AnalogSwitcherSpec, apply_analog_switcher
from ..codecs.patchbay_da import PatchbayDASpec, apply_patchbay_da, DistAmpStage, PatchRouterStage
from ..codecs.satellite_ird import SatelliteIRDSpec, apply_satellite_ird
from ..codecs.tv_transmitter_chain import TVTransmitterSpec, apply_tv_transmitter_chain
from ..codecs.spatial_environment_field import SpatialEnvFieldSpec, apply_spatial_environment_field
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.signal import SignalFrame


class ConnectorStage(Stage):
    """Physical connectors + pluggable RF / TV-DX broadcast signal chain."""

    name = "connector"
    layers = ("connector", "rf")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._rflink: RFLink | None = None
        self._read_params()
        self.state: dict = {}

    def _read_params(self) -> None:
        c = self.cfg_of("connector")
        r = self.cfg_of("rf")

        def f(key: str, default: float = 0.0) -> float:
            return float(c.get(key, default))

        def rf_f(key: str, default: float = 0.0) -> float:
            return float(r.get(key, default))

        # which physical connector links are present, and their condition
        self.hdmi_on = f("hdmi_present", 0.0) > 0.5
        self.av_on = f("av_present", 0.0) > 0.5
        self.rf_on = f("rf_present", 0.0) > 0.5
        self.svideo_on = f("svideo_present", 0.0) > 0.5

        self.hdmi_moist, self.hdmi_corr = f("hdmi_moisture"), f("hdmi_corrosion")
        self.av_moist, self.av_corr = f("av_moisture"), f("av_corrosion")
        self.rf_moist, self.rf_corr = f("rf_moisture"), f("rf_corrosion")
        self.sv_moist, self.sv_corr = f("svideo_moisture"), f("svideo_corrosion")

        self.ground_loop = f("av_ground_loop")
        self.mains_hz = f("mains_hz", 50.0)
        self.shield = float(np.clip(f("rf_shielding", 0.0), 0.0, 1.0))
        self.leakage = f("svideo_leakage")
        self.switcher_spec = AnalogSwitcherSpec.from_params(c)
        self.da_spec = PatchbayDASpec.from_params(c)
        self.sat_spec = SatelliteIRDSpec.from_params(r)
        self.tx_spec = TVTransmitterSpec.from_params(r)
        self.env_spec = SpatialEnvFieldSpec.from_params(c)

        # RF / TV-DX broadcast link parameters
        self.rf_link_on = rf_f("enabled", 0.0) > 0.5
        self.rf_params = RFParams(
            enabled=rf_f("enabled", 0.0),
            channel_mhz=rf_f("channel_mhz", 6.0),
            carrier_offset_mhz=rf_f("carrier_offset_mhz", 1.25),
            noise_figure_db=rf_f("noise_figure_db", 8.0),
            signal_dbm=rf_f("signal_dbm", -60.0),
            sync_loss_threshold_db=rf_f("sync_loss_threshold_db", 30.0),
            sync_loss_amount=rf_f("sync_loss_amount", 1.0),
            adjacent_channel=rf_f("adjacent_channel", 0.0),
            broadcast_noise=rf_f("broadcast_noise", 0.0),
            tv_standard=rf_f("tv_standard", 0.0),
            propagation_mode=rf_f("propagation_mode", 0.0),
            distance_km=rf_f("distance_km", 45.0),
            dx_duct_gain_db=rf_f("dx_duct_gain_db", 24.0),
            qsb_depth_db=rf_f("qsb_depth_db", 0.0),
            qsb_rate_hz=rf_f("qsb_rate_hz", 0.25),
            multipath_level=rf_f("multipath_level", 0.0),
            multipath_delay_us=rf_f("multipath_delay_us", 0.85),
            multipath_phase_deg=rf_f("multipath_phase_deg", 35.0),
            cochannel_interference=rf_f("cochannel_interference", 0.0),
            cochannel_offset_khz=rf_f("cochannel_offset_khz", 10.05),
            tuner_type=rf_f("tuner_type", 0.0),
            tuning_offset_khz=rf_f("tuning_offset_khz", 0.0),
            front_end_overload=rf_f("front_end_overload", 0.0),
            color_killer_threshold_db=rf_f("color_killer_threshold_db", 18.0),
            channel_switch_rate=rf_f("channel_switch_rate", 0.0),
            channel_switch_dur_s=rf_f("channel_switch_dur_s", 0.38),
            lo_sweep_mhz=rf_f("lo_sweep_mhz", 0.0),
            vsync_roll=rf_f("vsync_roll", 0.0),
            vbi_bar_vis=rf_f("vbi_bar_vis", 0.75),
            flagging=rf_f("flagging", 0.0),
            hanover_bars=rf_f("hanover_bars", 0.0),
        )
        if self._rflink is None:
            self._rflink = RFLink(self.ctx.std, self.rf_params, seed=self.ctx.seed)
        else:
            self._rflink.p = self.rf_params
            self._rflink.tx_node.p = self.rf_params
            self._rflink.channel_node.p = self.rf_params
            self._rflink.frontend_node.p = self.rf_params
            self._rflink.demod_node.p = self.rf_params

    def on_params_changed(self) -> None:
        self._read_params()

    def prepare(self) -> None:
        if self._rflink is not None:
            self._rflink.reset()

    # ------------------------------------------------------------------
    def _stats(self, spec, moisture: float, corrosion: float) -> dict:
        z = impedance(spec, moisture, corrosion)
        return {"impedance_ohm": round(z, 1),
                "reflection": round(abs(reflection_coefficient(spec.nominal_z, z)), 4)}

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        from ..codecs.power_bus import PowerBusSpec, evaluate_power_bus
        pbus_spec = PowerBusSpec.from_params(self.ctx.params)
        pbus_on = pbus_spec.is_active()
        sw_on = self.switcher_spec.is_active()
        da_on = self.da_spec.is_active()
        sat_on = self.sat_spec.is_active()
        tx_on = self.tx_spec.is_active()
        env_on = self.env_spec.is_active()
        active = self.hdmi_on or self.av_on or self.svideo_on or self.rf_on or self.rf_link_on or pbus_on or sw_on or da_on or sat_on or tx_on or env_on
        if not active:
            return buf

        meta = dict(buf.meta)
        report: dict = {}
        rgb = buf.rgb
        comp = buf.composite
        sig: SignalFrame | None = meta.get("signal")

        from ..core.signal import composite_encode, demod_chroma, split_composite

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
                comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            if comp is not None:
                comp, info = degrade_composite(comp, spec, self.av_moist, self.av_corr,
                                               self.ground_loop, self.mains_hz, buf.std,
                                               buf.index)
                report["av"] = info

        if comp is not None and self.av_on and not self.rf_on and not self.rf_link_on:
            y, chroma = split_composite(comp, buf.std, comb_depth=0.0,
                                        chroma_bandwidth_hz=0.9e6)
            i, q = demod_chroma(chroma / 0.63, buf.std, i_bw_hz=1.2e6, q_bw_hz=1.2e6)
            sig = SignalFrame(y, i, q)

        # ---- 4. Radio / TV-DX Broadcast RF Link (Modulator -> Channel -> Tuner -> Demod)
        if self.rf_link_on and self._rflink is not None:
            if sig is None:
                sig = SignalFrame.from_rgb(np.clip(rgb, 0.0, 1.0), buf.std)
            if comp is None or self.hdmi_on or self.svideo_on:
                comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)

            rng_rf = np.random.default_rng((int(self.ctx.seed) * 8191 + int(buf.index) * 131) & 0xFFFFFFFF)
            base_comp, rf_info = self._rflink.apply(comp, int(buf.index), rng_rf)

            window = float(np.clip(
                float(self.ctx.layer("format").get("chroma_bandwidth", 0.75)) * 1.15e6,
                0.35e6, 1.2e6))
            y_rf, chroma_rf = split_composite(base_comp, buf.std, comb_depth=0.0,
                                              chroma_bandwidth_hz=window)
            i_rf, q_rf = demod_chroma(chroma_rf / 0.63, buf.std, i_bw_hz=1.5e6, q_bw_hz=1.5e6)

            ph = self._rflink.chroma_phase_error()
            if ph is not None:
                ph_arr = np.asarray(ph, dtype=np.float32).reshape(-1, 1)
                c_p, s_p = np.cos(ph_arr), np.sin(ph_arr)
                i_rf, q_rf = i_rf * c_p - q_rf * s_p, i_rf * s_p + q_rf * c_p

            ck_gain = self._rflink.color_killer_gain()
            if ck_gain < 0.999:
                i_rf = i_rf * ck_gain
                q_rf = q_rf * ck_gain

            y_rf, i_rf, q_rf, glitch_info = self._rflink.apply_post_demod_glitches(
                y_rf, i_rf, q_rf, buf.index, self.ctx.fps,
                np.random.default_rng((int(self.ctx.seed) * 4099 + buf.index * 97) & 0xFFFFFFFF))
            rf_info.update(glitch_info)
            sig = SignalFrame(y_rf.astype(np.float32),
                              i_rf.astype(np.float32),
                              q_rf.astype(np.float32))
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            rgb = np.clip(sig.to_rgb(), 0.0, 1.0).astype(np.float32)
            report["rf_link"] = rf_info
            meta["rf"] = rf_info
            self.state["rf_link"] = rf_info

        # ---- 5. Physical RF Coaxial Connector Degradation (damp/corroded joint)
        if self.rf_on:
            spec = CONNECTORS["rf"]
            base_rgb = sig.to_rgb() if sig is not None else np.clip(rgb, 0.0, 1.0)
            rgb_rf, info = degrade_rf_signal(base_rgb, spec, self.rf_moist, self.rf_corr,
                                             self.shield, buf.index, self.ctx.seed + 23)
            report["rf"] = info
            sig = SignalFrame.from_rgb(rgb_rf, buf.std)
            rgb = rgb_rf
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            sig._rf_direct = True
        elif (self.hdmi_on or self.svideo_on) and not self.rf_link_on:
            sig = SignalFrame.from_rgb(np.clip(rgb, 0.0, 1.0), buf.std)

        if sig is not None:
            rgb = np.clip(sig.to_rgb(), 0.0, 1.0).astype(np.float32)
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)

        sq = float(self.cfg_of("connector").get("aspect_squeeze", 0.0))
        if sq > 1e-4 and sig is not None:
            h_s, w_s = sig.y.shape
            ratio = float(np.clip(1.0 - 0.25 * sq, 0.5, 1.0))
            act_w = max(8, int(round(w_s * ratio)))
            pad_l = (w_s - act_w) // 2
            xs = np.linspace(0.0, w_s - 1.0, act_w, dtype=np.float32)
            x0 = np.floor(xs).astype(np.int32)
            x1 = np.minimum(x0 + 1, w_s - 1)
            fx = (xs - x0)[None, :]
            def _sq_plane(pl, fill=0.0):
                out_p = np.full((h_s, w_s), fill, dtype=np.float32)
                out_p[:, pad_l:pad_l + act_w] = pl[:, x0] * (1.0 - fx) + pl[:, x1] * fx
                return out_p
            sig = SignalFrame(_sq_plane(sig.y, 0.02), _sq_plane(sig.i, 0.0), _sq_plane(sig.q, 0.0))
            rgb = np.clip(sig.to_rgb(), 0.0, 1.0).astype(np.float32)
            from ..core.signal import composite_encode
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)

        # ---- 6. Analog Video Switcher / Mixer & Special Effects Generator (Module 6 / Candidate C)
        if sw_on:
            if sig is None:
                sig = SignalFrame.from_rgb(np.clip(rgb, 0.0, 1.0), buf.std)
            bus_b_yiq = meta["bus_b_yiq"] if "bus_b_yiq" in meta else None
            y_sw, i_sw, q_sw, sw_info = apply_analog_switcher(
                sig.y, sig.i, sig.q, self.switcher_spec,
                frame_index=int(buf.index), fps=self.ctx.fps,
                std=buf.std, seed=self.ctx.seed, bus_b_yiq=bus_b_yiq,
            )
            sig = SignalFrame(y_sw, i_sw, q_sw)
            rgb = np.clip(sig.to_rgb(), 0.0, 1.0).astype(np.float32)
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            report["switcher"] = sw_info
            meta["switcher"] = sw_info
            self.state["switcher"] = sw_info

        # ---- 6b. Broadcast Patchbay, Distribution Amplifier & 75-Ohm Impedance Load (Candidate E)
        if da_on:
            if sig is None:
                sig = SignalFrame.from_rgb(np.clip(rgb, 0.0, 1.0), buf.std)
            bus_b_yiq = meta["bus_b_yiq"] if "bus_b_yiq" in meta else None
            y_da, i_da, q_da, da_info = apply_patchbay_da(
                sig.y, sig.i, sig.q, self.da_spec,
                frame_index=int(buf.index), fps=self.ctx.fps,
                std=buf.std, seed=self.ctx.seed, bus_b_yiq=bus_b_yiq,
            )
            sig = SignalFrame(y_da, i_da, q_da)
            rgb = np.clip(sig.to_rgb(), 0.0, 1.0).astype(np.float32)
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            report["patchbay_da"] = da_info
            meta["patchbay_da"] = da_info
            self.state["patchbay_da"] = da_info

        if pbus_on:
            pstate = meta["power_bus"] if "power_bus" in meta else None
            if pstate is None:
                h_c, w_c = rgb.shape[:2]
                row_l = (np.clip(rgb, 0.0, 1.0) @ np.array([0.299, 0.587, 0.114], np.float32)).mean(axis=1)
                pstate = evaluate_power_bus(self.ctx.params, int(buf.index), self.ctx.fps, h_c, w_c, self.ctx.seed, row_luma=row_l)
                meta["power_bus"] = pstate
            if pstate is not None and pstate.active:
                report["power_bus"] = pstate.to_dict()
                self.state["power_bus"] = report["power_bus"]

        # ---- 7. Satellite IRD Receiver & Analog/Digital Cliff Effect (Candidate M)
        if sat_on:
            base_sat = sig.to_rgb() if sig is not None else np.clip(rgb, 0.0, 1.0)
            rgb_sat, sat_info = apply_satellite_ird(
                base_sat, self.sat_spec, std=buf.std,
                frame_index=int(buf.index), fps=float(self.ctx.fps), seed=int(self.ctx.seed),
            )
            sig = SignalFrame.from_rgb(rgb_sat, buf.std)
            sig._rf_direct = True
            rgb = rgb_sat
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            report["satellite_ird"] = sat_info["report"]
            meta["satellite_ird"] = sat_info
            self.state["satellite_ird"] = sat_info["report"]

        # ---- 8. TV Transmitter Exciter, High-Power Amplifier & End-to-End Broadcast Bench (Candidate N)
        if tx_on:
            base_tx = sig.to_rgb() if sig is not None else np.clip(rgb, 0.0, 1.0)
            rgb_tx, tx_info = apply_tv_transmitter_chain(
                base_tx, self.tx_spec, std=buf.std,
                frame_index=int(buf.index), fps=float(self.ctx.fps), seed=int(self.ctx.seed),
            )
            sig = SignalFrame.from_rgb(rgb_tx, buf.std)
            sig._rf_direct = True
            rgb = rgb_tx
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            report["tv_transmitter"] = tx_info["report"]
            meta["tv_transmitter"] = tx_info
            self.state["tv_transmitter"] = tx_info["report"]

        # ---- 9. 2D Spatial Physical Environment Field & Multi-Domain Coupling (Candidate R)
        if env_on:
            base_env = sig.to_rgb() if sig is not None else np.clip(rgb, 0.0, 1.0)
            rgb_env, env_info = apply_spatial_environment_field(
                base_env, self.env_spec,
                frame_index=int(buf.index), fps=float(self.ctx.fps), seed=int(self.ctx.seed) + 89,
            )
            sig = SignalFrame.from_rgb(rgb_env, buf.std)
            sig._rf_direct = True
            rgb = rgb_env
            comp = composite_encode(sig.y, sig.i, sig.q, buf.std, chroma_gain=0.63)
            report["spatial_environment"] = env_info
            meta["spatial_environment"] = env_info
            self.state["spatial_environment"] = env_info

        meta["signal"] = sig
        meta["connector"] = report
        return FrameBuffer(rgb, comp, buf.index, buf.time, buf.std, meta)

    @staticmethod
    def _apply_aspect_squeeze(rgb: np.ndarray, sq: float = 1.0) -> tuple[np.ndarray, dict]:
        arr = np.asarray(rgb, dtype=np.float32)
        h_s, w_s = arr.shape[:2]
        s_val = float(np.clip(sq, 0.0, 1.5))
        if s_val <= 1e-4:
            return arr.copy(), {"aspect_squeeze": 0.0, "active_width_px": w_s, "pillarbox_px": 0}
        ratio = float(np.clip(0.75 ** s_val, 0.45, 1.0))
        act_w = max(8, int(round(w_s * ratio)))
        pad_l = (w_s - act_w) // 2
        pad_r = w_s - act_w - pad_l
        xs = np.linspace(0.0, w_s - 1.0, act_w, dtype=np.float32)
        x0 = np.floor(xs).astype(np.int32)
        x1 = np.minimum(x0 + 1, w_s - 1)
        fx = (xs - x0)[None, :, None] if arr.ndim == 3 else (xs - x0)[None, :]
        squeezed = arr[:, x0] * (1.0 - fx) + arr[:, x1] * fx
        out = np.zeros_like(arr)
        out[:, pad_l:pad_l + act_w] = squeezed
        return out.astype(np.float32), {
            "aspect_squeeze": round(s_val, 3),
            "width_ratio": round(act_w / max(1.0, float(w_s)), 4),
            "active_width_px": int(act_w),
            "pillarbox_px": int(pad_l),
            "pillarbox_right_px": int(pad_r),
        }

    def describe(self) -> dict:
        return dict(self.state)


class VideoSwitcherStage(Stage):
    """
    Standalone Analog Video Switcher / Mixer stage for the `video_switcher` node in the node graph.
    Accepts `in_a` via `buf` and optional `in_b` via `buf.meta[bus_b_yiq]`.
    """

    name = "video_switcher"
    layers = ("connector",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()

    def _read_params(self) -> None:
        c = self.cfg_of("connector")
        self.spec = AnalogSwitcherSpec.from_params(c)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        from ..core.signal import composite_encode, demod_chroma, split_composite
        meta = dict(buf.meta)
        sig: SignalFrame | None = meta.get("signal")
        comp = buf.composite
        if sig is None:
            if comp is not None:
                y, ch = split_composite(comp, buf.std, comb_depth=0.0, chroma_bandwidth_hz=0.9e6)
                i, q = demod_chroma(ch / 0.63, buf.std, i_bw_hz=1.5e6, q_bw_hz=1.5e6)
                sig = SignalFrame(y, i, q)
            else:
                sig = SignalFrame.from_rgb(np.clip(buf.rgb, 0.0, 1.0), buf.std)

        eff_spec = self.spec
        bus_b_yiq = meta["bus_b_yiq"] if "bus_b_yiq" in meta else None
        # If placed explicitly in the graph with `in_b` wired and mode was left at 0, default to Wipe (3)
        if not eff_spec.is_active() and bus_b_yiq is not None:
            import dataclasses
            eff_spec = dataclasses.replace(eff_spec, mode=3)

        if not eff_spec.is_active():
            return buf

        y_sw, i_sw, q_sw, sw_info = apply_analog_switcher(
            sig.y, sig.i, sig.q, eff_spec,
            frame_index=int(buf.index), fps=self.ctx.fps,
            std=buf.std, seed=self.ctx.seed, bus_b_yiq=bus_b_yiq,
        )
        sig_out = SignalFrame(y_sw, i_sw, q_sw)
        rgb_out = np.clip(sig_out.to_rgb(), 0.0, 1.0).astype(np.float32)
        comp_out = composite_encode(sig_out.y, sig_out.i, sig_out.q, buf.std, chroma_gain=0.63)
        meta["signal"] = sig_out
        meta["switcher"] = sw_info
        return FrameBuffer(rgb_out, comp_out, buf.index, buf.time, buf.std, meta)
