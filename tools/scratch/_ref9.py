import sys, glob
sys.path.insert(0, ".")
import numpy as np
from atsim.io.imagecodec import load_image
from scipy.ndimage import laplace
print()
print("%-12s %7s %7s %9s %10s %9s" % ("frame","mean","sat","hf","row_ripple","dark%"))
for f in sorted(glob.glob("tools/ref9/*.png")):
    a = load_image(f)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = float(np.mean(np.where(mx>1e-6,(mx-mn)/np.maximum(mx,1e-6),0)))
    g = a @ np.array([0.299,0.587,0.114], np.float32)
    rr = g.mean(axis=1)
    rip = float((rr - np.convolve(rr, np.ones(3)/3, mode="same"))[5:-5].std())
    print("%-12s %7.4f %7.3f %9.5f %10.4f %8.1f%%" % (
      f.split("\\")[-1], a.mean(), sat, np.abs(laplace(g)).mean(), rip,
      100*float((g<0.12).mean())))
