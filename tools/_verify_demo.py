import sys, glob
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

print()
print("panel-wise |delta| vs the normal-play panel, per sampled time:")
print("%-8s %10s %10s %10s" % ("time", "shuttle12", "shuttle28", "shuttle45"))
for f in sorted(glob.glob("tools/out/bd_*.png")):
    a = np.asarray(Image.open(f).convert("RGB"), np.float32)/255.0
    w = a.shape[1]//4
    p = [a[:, i*w:(i+1)*w] for i in range(4)]
    d = [float(np.abs(p[i]-p[0]).mean()) for i in (1,2,3)]
    print("%-8s %10.4f %10.4f %10.4f" % (f.split("bd_")[-1].replace(".png",""), *d))
# temporal: consecutive frames must differ (the clip really moves)
import subprocess
print()
print("temporal motion check (consecutive decoded frames):")
prev = None
for t in (0.20, 0.24, 0.28, 0.32):
    out = f"tools/out/_tf.png"
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(t),"-i","tools/out/blocking_demo.mp4",
                    "-frames:v","1",out], check=True)
    a = np.asarray(Image.open(out).convert("RGB"), np.float32)/255.0
    if prev is not None:
        print("  t=%.2f  frame-to-frame |delta| = %.5f" % (t, float(np.abs(a-prev).mean())))
    prev = a
