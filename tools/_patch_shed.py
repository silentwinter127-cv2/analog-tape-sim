p = "atsim/stages/tape.py"
s = open(p, encoding="utf-8").read()
s = s.replace('        self.luma_pedestal = float(cfg.get("luma_pedestal", 0.0))',
'''        # Coating-shed grain: fine, dense, luminance-only noise.  Separate from
        # `noise_floor` because that is a general particle floor and reaches only a
        # sixth of the energy the reference shows.
        self.shed_grain = float(cfg.get("shed_grain", 0.0))
        self.luma_pedestal = float(cfg.get("luma_pedestal", 0.0))''')
s = s.replace("        y, i, q = self._apply_luma_pedestal(y, i, q)",
              "        y, i, q = self._apply_luma_pedestal(y, i, q)\n\n"
              "        # 1d. shed-coating grain: fine luminance noise, no colour\n"
              "        y, i, q = self._apply_shed_grain(y, i, q, frame_index, rng)")
open(p, "w", encoding="utf-8").write(s)
print("shed_grain wired")
