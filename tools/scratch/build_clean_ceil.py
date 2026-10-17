import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_dilation
from PIL import Image

im_real = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
real = np.asarray(im_real, dtype=np.float32) / 255.0
H, W, _ = real.shape
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
lum_real = real @ LUMA

ceil_zone = np.zeros((H, W), dtype=bool)
ceil_zone[0:268, 0:245] = True
ceil_zone[0:268, 575:720] = True

core_mask = (lum_real > 0.78) & ceil_zone
lbl, n_lamps = label(core_mask)
lamps = []
for idx in range(1, n_lamps + 1):
    ys, xs = np.nonzero(lbl == idx)
    if len(ys) < 8:
        continue
    # Core leading center (before rightward/downward comatic flare)
    cy = float(np.percentile(ys, 38))
    cx = float(np.percentile(xs, 36))
    h_span = float(ys.max() - ys.min() + 1)
    w_span = float(xs.max() - xs.min() + 1)
    ry = float(np.clip(h_span * 0.36, 3.0, 6.8))
    rx = float(np.clip(w_span * 0.36, 3.5, 7.8))
    lamps.append((cy, cx, ry, rx))

print(f"Detected {len(lamps)} ceiling lamps.")

clean = real.copy()
clean[:6, :] = clean[6:12, :].mean(axis=0, keepdims=True)
clean[-6:, :] = clean[-12:-6, :].mean(axis=0, keepdims=True)
clean[:, :11] = clean[:, 11:16].mean(axis=1, keepdims=True)
clean[:, -10:] = clean[:, -15:-10].mean(axis=1, keepdims=True)

exit_mask = np.zeros((H, W), dtype=np.float32)
exit_mask[300:340, 635:675] = 1.0
exit_mask = gaussian_filter(exit_mask, 5.0)
clean_lum = clean @ LUMA

aberr_mask = ((lum_real > 0.53) | (lum_real < 0.41) | (np.abs(real[..., 0] - real[..., 2]) > 0.025)) & ceil_zone
aberr_mask = binary_dilation(aberr_mask, iterations=11)
valid_ceil = (~aberr_mask).astype(np.float32)
ceil_smooth = gaussian_filter(clean_lum * valid_ceil, 22.0) / np.maximum(gaussian_filter(valid_ceil, 22.0), 1e-4)
clean_lum = np.where(aberr_mask, ceil_smooth, clean_lum)
clean_lum_smooth = gaussian_filter(clean_lum, 0.9)
clean_rgb = np.stack([clean_lum_smooth, clean_lum_smooth, clean_lum_smooth], axis=-1)
clean_rgb = clean_rgb * (1.0 - exit_mask[..., None]) + clean * exit_mask[..., None]

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
for cy, cx, ry, rx in lamps:
    dist = np.sqrt(((yy - cy) / ry)**2 + ((xx - cx) / rx)**2)
    disk = np.clip((1.05 - dist) / 0.22, 0.0, 1.0)
    clean_rgb = clean_rgb * (1.0 - disk[..., None]) + 1.0 * disk[..., None]

out_path = "/working_dir/c_8f1524494f0b64f7/repo/tools/scratch/clean_ceil.png"
Image.fromarray((np.clip(clean_rgb, 0.0, 1.0) * 255).astype(np.uint8)).save(out_path)
print("Saved", out_path)
