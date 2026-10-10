import sys
sys.path.insert(0, ".")
import numpy as np
from atsim.io.ffmpeg import VideoWriter
try:
    w = VideoWriter("tools/out/_probe_enc.mp4", 64, 48, 25.0)
    print("constructed; _proc =", w._proc)
    if w._proc is not None:
        print("  returncode:", w._proc.poll())
    w.write(np.zeros((48, 64, 3), np.float32))
    print("wrote one frame")
    w.close()
    print("closed OK")
except Exception as e:
    import traceback; traceback.print_exc()
