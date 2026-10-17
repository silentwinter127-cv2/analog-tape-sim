import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_closing, label
from PIL import Image

im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
real = np.asarray(im_real, dtype=np.float32) / 255.0
H, W, _ = real.shape
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
lum_real = real @ LUMA

# 1. Build seamless clean_ceil.png
ceil_zone = np.zeros((H, W), dtype=bool)
ceil_zone[0:268, 0:244] = True
ceil_zone[0:265, 588:720] = True
# Exclude column 1 (x=242..452) except where lamp (72,230) and (206,231) sit (x < 241)
raw_cores = binary_closing((lum_real > 0.72) & ceil_zone, structure=np.ones((5, 5), dtype=bool))
lbl, n_lamps = label(raw_cores)
lamps = []
lamp_footprint = np.zeros((H, W), dtype=bool)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

for idx in range(1, n_lamps + 1):
    mask_i = (lbl == idx) & (lum_real > 0.74)
    ys, xs = np.nonzero(mask_i)
    if len(ys) < 6:
        continue
    # The upstream/leading head of the lamp (before comatic flare extends it down-right)
    cy = float(np.percentile(ys, 38))
    cx = float(np.percentile(xs, 34))
    h_span = float(ys.max() - ys.min() + 1)
    w_span = float(xs.max() - xs.min() + 1)
    # Physical radius of the clean round ceiling downlight prior to comatic stretching
    r0 = float(np.clip(min(h_span, w_span) * 0.36, 3.0, 5.8))
    lamps.append((cy, cx, r0))
    lamp_footprint |= ((yy - cy)**2 + (xx - cx)**2) <= (max(h_span, w_span) * 0.85 + 11.0)**2

print(f"Detected {len(lamps)} lamps.")

clean = real.copy()
clean[:5, :] = clean[5:10, :].mean(axis=0, keepdims=True)
clean[-5:, :] = clean[-10:-5, :].mean(axis=0, keepdims=True)
clean[:, :10] = clean[:, 10:15].mean(axis=1, keepdims=True)
clean[:, -9:] = clean[:, -14:-9].mean(axis=1, keepdims=True)

exit_mask = np.zeros((H, W), dtype=np.float32)
exit_mask[300:340, 635:675] = 1.0
exit_mask = gaussian_filter(exit_mask, 5.0)
clean_lum = clean @ LUMA

# Smooth feathering for inpainting ceiling around lamps
valid = ((~lamp_footprint) & ceil_zone & (lum_real > 0.38) & (lum_real < 0.53)).astype(np.float32)
ceil_bg = gaussian_filter(clean_lum * valid, 18.0) / np.maximum(gaussian_filter(valid, 18.0), 1e-4)
blend_w = gaussian_filter((lamp_footprint & ceil_zone).astype(np.float32), 3.5)
blend_w = np.clip(blend_w * 1.8, 0.0, 1.0)
clean_lum = clean_lum * (1.0 - blend_w) + ceil_bg * blend_w
clean_lum = gaussian_filter(clean_lum, 0.6)
clean_rgb = np.stack([clean_lum, clean_lum, clean_lum], axis=-1)
clean_rgb = clean_rgb * (1.0 - exit_mask[..., None]) + clean * exit_mask[..., None]

# Draw clean round neutral-white lamps (8-bit [0..1] range, peak = 1.0)
for cy, cx, r0 in lamps:
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    disk = np.clip((r0 + 0.5 - dist) / 1.0, 0.0, 1.0)
    clean_rgb = clean_rgb * (1.0 - disk[..., None]) + 1.0 * disk[..., None]

clean_u8 = (np.clip(clean_rgb, 0.0, 1.0) * 255).astype(np.uint8)
Image.fromarray(clean_u8).save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png")
print("Saved seamless clean_ceil.png")
