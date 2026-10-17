import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_closing
from PIL import Image

im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
real = np.asarray(im_real, dtype=np.float32) / 255.0
H, W, _ = real.shape
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
lum_real = real @ LUMA

# 1. Build crisp clean_ceil.png without touching column edges and without splitting lamps
from scipy.ndimage import label
ceil_zone = np.zeros((H, W), dtype=bool)
ceil_zone[0:266, 0:235] = True
ceil_zone[0:260, 590:720] = True

# Close small dark notches before labeling so each lamp is a single connected component
raw_cores = binary_closing((lum_real > 0.72) & ceil_zone, structure=np.ones((5, 5), dtype=bool))
lbl, n_lamps = label(raw_cores)
lamps = []
lamp_footprint = np.zeros((H, W), dtype=bool)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

for idx in range(1, n_lamps + 1):
    mask_i = (lbl == idx) & (lum_real > 0.75)
    ys, xs = np.nonzero(mask_i)
    if len(ys) < 6:
        continue
    # Leading-side center of the lamp (before comatic tail extension)
    cy = float(np.percentile(ys, 42))
    cx = float(np.percentile(xs, 38))
    h_span = float(ys.max() - ys.min() + 1)
    w_span = float(xs.max() - xs.min() + 1)
    ry = float(np.clip(h_span * 0.35, 3.2, 6.2))
    rx = float(np.clip(w_span * 0.34, 3.6, 7.2))
    lamps.append((cy, cx, ry, rx))
    lamp_footprint |= ((yy - cy)**2 + (xx - cx)**2) <= (max(ry, rx) + 12.0)**2

print(f"Cleanly detected {len(lamps)} distinct ceiling lamps.")

clean = real.copy()
# Clean composite capture blanking border
clean[:5, :] = clean[5:10, :].mean(axis=0, keepdims=True)
clean[-5:, :] = clean[-10:-5, :].mean(axis=0, keepdims=True)
clean[:, :10] = clean[:, 10:15].mean(axis=1, keepdims=True)
clean[:, -9:] = clean[:, -14:-9].mean(axis=1, keepdims=True)

exit_mask = np.zeros((H, W), dtype=np.float32)
exit_mask[300:340, 635:675] = 1.0
exit_mask = gaussian_filter(exit_mask, 5.0)
clean_lum = clean @ LUMA

# Only inpaint within `lamp_footprint & ceil_zone` so column edges are 100% untouched!
inpaint_mask = lamp_footprint & ceil_zone
valid = (~inpaint_mask) & ceil_zone
ceil_bg = gaussian_filter(clean_lum * valid.astype(np.float32), 16.0) / np.maximum(gaussian_filter(valid.astype(np.float32), 16.0), 1e-4)
clean_lum = np.where(inpaint_mask, ceil_bg, clean_lum)
# Lightly smooth passive scene noise
clean_lum = gaussian_filter(clean_lum, 0.65)
clean_rgb = np.stack([clean_lum, clean_lum, clean_lum], axis=-1)
clean_rgb = clean_rgb * (1.0 - exit_mask[..., None]) + clean * exit_mask[..., None]

# Place clean, round/elliptical, super-white unclipped lamps (value = 2.4 in linear light, clipped to 1.0 when saved as 8-bit PNG)
clean_linear = clean_rgb.copy()
for cy, cx, ry, rx in lamps:
    dist = np.sqrt(((yy - cy) / ry)**2 + ((xx - cx) / rx)**2)
    disk = np.clip((1.05 - dist) / 0.22, 0.0, 1.0)
    clean_linear = clean_linear * (1.0 - disk[..., None]) + 2.2 * disk[..., None]

clean_u8 = (np.clip(clean_linear, 0.0, 1.0) * 255).astype(np.uint8)
Image.fromarray(clean_u8).save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png")
print("Updated clean_ceil.png")
