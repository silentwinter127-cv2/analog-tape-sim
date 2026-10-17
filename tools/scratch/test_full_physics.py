import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, label
from PIL import Image

im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
real = np.asarray(im_real, dtype=np.float32) / 255.0
H, W, _ = real.shape
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
lum_real = real @ LUMA

# 1. Reconstruct 100% clean ceiling-lamp scene
ceil_zone = np.zeros((H, W), dtype=bool)
ceil_zone[0:268, 0:245] = True
ceil_zone[0:265, 588:720] = True

raw_cores = binary_closing((lum_real > 0.76) & ceil_zone, structure=np.ones((3, 3), dtype=bool))
lbl, n_lamps = label(raw_cores)
raw_list = []
for idx in range(1, n_lamps + 1):
    ys, xs = np.nonzero(lbl == idx)
    if len(ys) < 5:
        continue
    raw_list.append((len(ys), float(np.percentile(ys, 40)), float(np.percentile(xs, 36)), ys, xs))

# Sort by area descending and merge any fragment within 16 px of an existing lamp
raw_list.sort(key=lambda t: -t[0])
lamps = []
lamp_footprint = np.zeros((H, W), dtype=bool)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

for area, cy, cx, ys, xs in raw_list:
    if any(np.hypot(cy - ly, cx - lx) < 16.5 for ly, lx, _ in lamps):
        continue
    h_span = float(ys.max() - ys.min() + 1)
    w_span = float(xs.max() - xs.min() + 1)
    r0 = float(np.clip(min(h_span, w_span) * 0.38, 3.3, 5.8))
    lamps.append((cy, cx, r0))
    lamp_footprint |= ((yy - cy)**2 + (xx - cx)**2) <= (max(h_span, w_span) * 0.90 + 10.0)**2

print(f"Merged into {len(lamps)} physical ceiling downlights.")

clean = real.copy()
clean[:5, :] = clean[5:10, :].mean(axis=0, keepdims=True)
clean[-5:, :] = clean[-10:-5, :].mean(axis=0, keepdims=True)
clean[:, :10] = clean[:, 10:15].mean(axis=1, keepdims=True)
clean[:, -9:] = clean[:, -14:-9].mean(axis=1, keepdims=True)

exit_mask = np.zeros((H, W), dtype=np.float32)
exit_mask[300:340, 635:675] = 1.0
exit_mask = gaussian_filter(exit_mask, 5.0)
clean_lum = clean @ LUMA

# Smooth ceiling inpainting
valid = ((~lamp_footprint) & ceil_zone & (lum_real > 0.40) & (lum_real < 0.52)).astype(np.float32)
ceil_bg = gaussian_filter(clean_lum * valid, 24.0) / np.maximum(gaussian_filter(valid, 24.0), 1e-4)
blend_w = gaussian_filter((lamp_footprint & ceil_zone).astype(np.float32), 4.0)
blend_w = np.clip(blend_w * 2.2, 0.0, 1.0) * ceil_zone.astype(np.float32)
clean_lum = clean_lum * (1.0 - blend_w) + ceil_bg * blend_w
# Restore crisp straight left edge of column 1 (around x=239..245, y=0..268) from y=275..310
col_edge_template = np.median(clean_lum[275:320, 236:250], axis=0, keepdims=True)
clean_lum[0:268, 238:250] = col_edge_template[:, 2:]
clean_lum = gaussian_filter(clean_lum, 0.55)
clean_rgb = np.stack([clean_lum, clean_lum, clean_lum], axis=-1)
clean_rgb = clean_rgb * (1.0 - exit_mask[..., None]) + clean * exit_mask[..., None]

for cy, cx, r0 in lamps:
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    disk = np.clip((r0 + 0.5 - dist) / 1.0, 0.0, 1.0)
    clean_rgb = clean_rgb * (1.0 - disk[..., None]) + 1.0 * disk[..., None]

clean_u8 = (np.clip(clean_rgb, 0.0, 1.0) * 255).astype(np.uint8)
Image.fromarray(clean_u8).save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png")
print("Saved clean_ceil.png with 27 lamps.")
