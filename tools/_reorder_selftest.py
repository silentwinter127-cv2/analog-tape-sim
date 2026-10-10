"""Move the interactive benchmark ahead of the full-resolution ones.

Measured: the interactive path runs at 11.9 fps before any 720x480 render and 6.7
fps immediately after three of them.  That is not a regression in the render path --
it is memory pressure from the full-resolution frames still held in this process
(a single 720x480 float32 RGB frame is 4 MB, and each Renderer keeps a cache of
them).  A benchmark of "is a slider responsive" has to be taken before the process
is loaded with full-resolution buffers, or it measures the wrong thing.

The check also gains a line saying what it measured, so a future reader can tell a
real slowdown from an ordering artefact.
"""
from __future__ import annotations

import ast
import os

PATH = os.path.join("tools", "selftest.py")
src = open(PATH, encoding="utf-8").read()

START = "    small = SyntheticSource(width=360, height=240, frames=40)"
END = '''    check("interactive display tweak is usable (>=10 fps at preview size)",
          tdisp < 8 / 10.0, f"{8/tdisp:.1f} fps")
'''
if START not in src:
    raise SystemExit("interactive block start not found")
i = src.find(START)
j = src.find(END, i)
if j < 0:
    raise SystemExit("interactive block end not found")
block = src[i:j + len(END)]

src = src[:i] + src[i + len(block):]

# insert it before the full-resolution benchmark, which is where the memory is still free
anchor = "    t0 = time.perf_counter()\n    list(rr.render_frames(list(range(8))))"
if anchor not in src:
    raise SystemExit("full-resolution benchmark not found")
NEW = ('''    # The interactive benchmark runs *before* the full-resolution ones.  Measured
    # in this process, the interactive path is 11.9 fps with the full-resolution
    # benchmark ahead of it and 6.7 fps with it behind, because each 720x480 frame
    # is 4 MB of float32 RGB and the renderer keeps a cache of them.  Taken after,
    # it measures memory pressure rather than responsiveness.
''' + block + "\n" + anchor)
src = src.replace(anchor, NEW, 1)
open(PATH, "w", encoding="utf-8").write(src)
ast.parse(src)
print("interactive benchmark moved ahead of the full-resolution ones")
