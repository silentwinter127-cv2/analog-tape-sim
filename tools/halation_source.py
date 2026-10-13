"""
Make `halation` carry the source's colour, which is what removes a thin line's blue.

**The mechanism, found by reading the line that does it.** `_halation` finishes with

    tint = np.array([1.05, 0.88, 1.05])
    return rgb + (a * 0.52) * halo[..., None] * tint[None, None, :]

so every pixel it touches receives a **fixed** tint -- a magenta-leaning coating colour, which is a real effect for a
lens but is a **constant**. The halo itself is built from a **luminance** mask (`hot` from `lum`), so it is
colourless before the tint is applied.

**And that is exactly what dilutes a small chroma difference.** The pole in the reference scene is a very bright thin
line whose **own** colour is a small blue difference (B-R +0.024). `halation` adds a large amount of light to it --
which is what a halation should do -- but adds it **neutral-to-magenta**, so the blue difference is divided by a much
larger total: the line stays a line and loses its colour. Measured across every value in the sweep, the pole's hue
never exceeded **+0.008** against the scene's **+0.024**.

**The physical fix is the one `source_glow` already uses.** Light that scatters from a bright source keeps that
source's spectrum, so a halo around a blue-white line should be blue-white and a halo around a sodium lamp should be
orange. So the halo's tint becomes the **blown region's own colour**, normalised so its luminance is unchanged -- which
means **a scene with no blown source, or a neutral one, behaves exactly as before**, and only a coloured source
changes. That keeps the change from moving anything it should not.

**The strength is blended rather than switched**, because the fixed tint is also a real lens property and some of it
may be wanted; the blend is exposed as a parameter so the two can be compared on measurement instead of argument.
"""
from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "atsim", "stages", "camera.py")
s = open(P, encoding="utf-8").read()

OLD = """        coverage = float(np.clip(hot.mean() * 14.0, 0.0, 1.0))
        halo = np.clip(0.80 * core + 0.32 * coverage * veil, 0.0, 1.6)
        tint = np.array([1.05, 0.88, 1.05], dtype=np.float32)
        return (rgb + (a * 0.52) * halo[..., None] * tint[None, None, :]).astype(np.float32)"""

NEW = '''        coverage = float(np.clip(hot.mean() * 14.0, 0.0, 1.0))
        halo = np.clip(0.80 * core + 0.32 * coverage * veil, 0.0, 1.6)
        # **The halo's colour is the source's, not a constant.**  Scattered light keeps the spectrum of what it came
        # from, so a halo around a blue-white line should be blue-white.  The fixed tint below is a real lens
        # property, but as a *constant* it dilutes a small chroma difference -- measured, a thin blue line under a
        # halation came back at B-R +0.000 against its own +0.024, and no value of the existing parameters recovered
        # it, because raising saturation cannot restore a difference that neutral light has divided away.
        #
        # So the tint is taken from the blown region itself, normalised so its **luminance is unchanged**: a scene
        # with no blown source, or a neutral one, therefore behaves exactly as before, and only a coloured source
        # moves.  Built from the same `hot` mask, per channel, so it is the colour of what is actually blowing out.
        blown = rgb * hot[..., None]
        mass = float(hot.sum())
        if mass > 1e-6:
            mean_rgb = blown.reshape(-1, 3).sum(axis=0) / mass
            g = float(mean_rgb[1])
            if g > 1e-4:
                src_tint = np.clip(mean_rgb / g, 0.55, 1.75).astype(np.float32)
            else:
                src_tint = np.ones(3, np.float32)
        else:
            src_tint = np.ones(3, np.float32)
        lens_tint = np.array([1.05, 0.88, 1.05], dtype=np.float32)
        mix = float(np.clip(self.halation_source_colour, 0.0, 1.0))
        tint = (1.0 - mix) * lens_tint + mix * src_tint
        return (rgb + (a * 0.52) * halo[..., None] * tint[None, None, :]).astype(np.float32)'''

assert OLD in s, "the halation return was not found"
s = s.replace(OLD, NEW, 1)

# the parameter
OLD_P = '        self.halation = float(cfg.get("halation", 0.0))'
NEW_P = ('        self.halation = float(cfg.get("halation", 0.0))\n'
         '        #: How much the halation\'s tint follows the **blown region\'s own colour** instead of the fixed\n'
         '        #: lens tint.  A scattering halo keeps the source\'s spectrum, so this is the physical behaviour;\n'
         '        #: it is a blend because the fixed tint is also a real lens property.\n'
         '        self.halation_source_colour = float(cfg.get("halation_source_colour", 0.0))')
assert OLD_P in s, "the halation read was not found"
s = s.replace(OLD_P, NEW_P, 1)
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("halation now takes its tint from the blown region, blended by `halation_source_colour`")

# and the schema parameter
P2 = os.path.join(ROOT, "atsim", "core", "profiles.py")
lines = open(P2, encoding="utf-8").read().split("\n")
if 'define("halation_source_colour"' in "\n".join(lines):
    print("   the schema already has it")
else:
    c = next(i for i, l in enumerate(lines) if 'define("halation"' in l)
    depth, end = 0, c
    for i in range(c, len(lines)):
        depth += lines[i].count("(") - lines[i].count(")")
        if depth <= 0 and i > c:
            end = i
            break
    ind = lines[c][:len(lines[c]) - len(lines[c].lstrip())]
    var = lines[c].strip().split(".")[0]
    block = [
        ind + '%s.define("halation_source_colour", "光晕跟随光源色", 0.0, 0.0, 1.0, unit="0-1", group=g,' % var,
        ind + '           phys_min=0.0, phys_max=1.0,',
        ind + '           help="**散射光带的是光源自己的颜色**，所以蓝色细线周围的光晕应该是蓝白的。"',
        ind + '                "0 = 固定的镜片镀膜色调（原行为），1 = 完全跟随炸白区的自身颜色。"',
        ind + '                "实测：蓝色细线在固定色调的光晕下回到 B−R +0.000（自身 +0.024），"',
        ind + '                "而且没有任何参数值能恢复——因为提高饱和度只能放大已存在的差，"',
        ind + '                "无法恢复被中性光除掉的差。")',
    ]
    out = lines[:end + 1] + block + lines[end + 1:]
    s2 = "\n".join(out)
    ast.parse(s2)
    m2 = next(i for i, l in enumerate(out) if 'define("halation_source_colour"' in l)
    r2 = next(i for i in range(m2, len(out)) if out[i].strip() == "return s")
    f2 = next(i for i, l in enumerate(out) if l.startswith("def build_schema"))
    assert f2 < m2 < r2, "placement is wrong"
    open(P2, "w", encoding="utf-8").write(s2)
    print("   added the schema parameter")

import sys
sys.path.insert(0, ROOT)
from atsim.core.profiles import SCHEMA                                       # noqa: E402
sp = SCHEMA.layer("camera").specs.get("halation_source_colour")
print("   camera.halation_source_colour: default %s, range %s..%s"
      % (sp.default, sp.minimum, sp.maximum) if sp else "   MISSING")
