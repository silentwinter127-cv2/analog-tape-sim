import sys, glob
sys.path.insert(0, ".")
import numpy as np
from PIL import Image
from scipy.ndimage import laplace
for f in sorted(glob.glob("tools/ref/*.png")):
    a = np.asarray(Image.open(f).convert("RGB"), np.float32)/255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.mean(np.where(mx>1e-6,(mx-mn)/np.maximum(mx,1e-6),0))
    luma = a @ np.array([0.299,0.587,0.114], np.float32)
    print("%-16s %4dx%-4d mean %.3f sat %.3f B-R %+.4f R-G %+.4f luma p1 %.3f p50 %.3f p99 %.3f hf %.4f" % (
      f.split("/")[-1], a.shape[1], a.shape[0], a.mean(), sat,
      a[...,2].mean()-a[...,0].mean(), a[...,0].mean()-a[...,1].mean(),
      np.percentile(luma,1), np.percentile(luma,50), np.percentile(luma,99),
      np.abs(laplace(luma)).mean()))
