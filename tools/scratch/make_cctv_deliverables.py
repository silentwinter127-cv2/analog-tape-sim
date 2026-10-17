import json
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from atsim.core.profiles import SCHEMA, FACTORY_PRESETS
from atsim.render.renderer import Renderer

def load_font(sz):
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

LUMA = np.array([0.299, 0.587, 0.114], np.float32)

# 1. Load Test Scene A (clean_ceil.png, 720x576) and Real Capture (477B4B7BED73D24392E50BA6BE1C3B65.jpg, 720x576)
clean_a_im = Image.open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/clean_ceil.png").convert("RGB")
real_a_im  = Image.open("/working_dir/c_8f1524494f0b64f7/user_uploaded_files/ttl=63d/user_uploaded/477B4B7BED73D24392E50BA6BE1C3B65.jpg").convert("RGB")
clean_a = np.asarray(clean_a_im, dtype=np.float32) / 255.0
real_a  = np.asarray(real_a_im, dtype=np.float32) / 255.0
H, W, _ = clean_a.shape

class StillSource:
    def __init__(self, arr):
        self.arr = arr.astype(np.float32)
        self.height, self.width = arr.shape[:2]
        self.fps = 25.0
        self.frame_count = 2
    def frame(self, idx):
        return self.arr

params = SCHEMA.resolve_all(FACTORY_PRESETS["sony_cctv_composite"]["layers"])
r_a = Renderer(StillSource(clean_a), W, H, 25.0, params, seed=20240101)
sim_a = np.clip(r_a.render_frame(0), 0.0, 1.0)
sim_a_im = Image.fromarray((sim_a * 255).astype(np.uint8))
sim_a_im.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/sony_cctv_simulated_hall.png")

# 2. Build Test Scene B: Photorealistic unprocessed architectural concourse/gallery with 3D perspective shading,
#    ceiling coffers, specular floor reflections, and 4 perspective rows of recessed circular downlights
Hb, Wb = 576, 720
yy, xx = np.mgrid[0:Hb, 0:Wb].astype(np.float32)
yn, xn = yy / (Hb - 1.0), xx / (Wb - 1.0)

# Perspective room geometry: ceiling (yn < 0.42), back wall & pillars (0.42 <= yn <= 0.74), polished floor (yn > 0.74)
ceil_plane = 0.45 + 0.07 * (1.0 - yn / 0.42) - 0.04 * np.abs(xn - 0.48)
# Subtle ceiling coffer seam lines
for x_seam in (0.26, 0.52, 0.76):
    seam_dist = np.abs((xn - 0.48) - (x_seam - 0.48) * (1.15 - 0.55 * (yn / 0.42)))
    ceil_plane -= 0.035 * np.exp(-0.5 * (seam_dist / 0.004) ** 2) * (yn < 0.42)

back_wall = 0.43 - 0.04 * np.abs(xn - 0.5) - 0.03 * np.abs(yn - 0.58)
floor_plane = 0.48 + 0.08 * ((yn - 0.74) / 0.26) - 0.05 * np.abs(xn - 0.5)

base_l = np.where(yn < 0.42, ceil_plane, np.where(yn > 0.74, floor_plane, back_wall))
# Add 3 perspective architectural columns with realistic 3D shading (lit front face + darker side return)
pillars = [
    # (x_front_l, x_front_r, x_side_r, y_top, y_bot, front_luma, side_luma)
    (0.06, 0.17, 0.20, 0.00, 0.90, 0.56, 0.44),
    (0.27, 0.34, 0.36, 0.14, 0.82, 0.53, 0.42),
    (0.76, 0.86, 0.89, 0.04, 0.88, 0.55, 0.43),
    (0.58, 0.64, 0.66, 0.18, 0.80, 0.51, 0.41),
]
for xfl, xfr, xsr, yt, yb, flum, slum in pillars:
    m_front = (xn >= xfl) & (xn < xfr) & (yn >= yt) & (yn <= yb)
    m_side  = (xn >= xfr) & (xn < xsr) & (yn >= yt) & (yn <= yb)
    grad_v  = 1.0 - 0.06 * ((yn - yt) / max(0.1, yb - yt))
    base_l = np.where(m_front, flum * grad_v, np.where(m_side, slum * grad_v, base_l))
    # Specular reflection of pillar on polished floor (yn > yb)
    m_refl = (xn >= xfl) & (xn < xsr) & (yn > yb)
    base_l = np.where(m_refl, base_l * 0.94 + 0.03, base_l)

# Recessed dark doorway in center-right back wall
door_m = (xn >= 0.43) & (xn <= 0.53) & (yn >= 0.48) & (yn <= 0.74)
base_l = np.where(door_m, 0.27 + 0.04 * np.cos((xn - 0.48) * 120.0), base_l)

scene_b = np.stack([base_l * 0.995, base_l * 1.000, base_l * 1.005], axis=-1).astype(np.float32)
# Illuminated green exit sign above the doorway
exit_m = np.exp(-0.5 * (((yy - 262.0) / 3.5) ** 2 + ((xx - 346.0) / 6.5) ** 2))
scene_b[..., 0] = scene_b[..., 0] * (1.0 - 0.7 * exit_m) + 0.35 * exit_m
scene_b[..., 1] = scene_b[..., 1] * (1.0 - 0.9 * exit_m) + 0.78 * exit_m
scene_b[..., 2] = scene_b[..., 2] * (1.0 - 0.7 * exit_m) + 0.52 * exit_m

scene_b = gaussian_filter(scene_b, sigma=(0.65, 0.65, 0.0))

# Place 22 clean circular recessed ceiling downlights across 4 perspective rows (radii shrinking with depth)
test_lamps = [
    # Row 1 (Left-center perspective line)
    (28, 155, 5.5), (68, 175, 4.9), (104, 192, 4.4), (135, 206, 4.0), (162, 218, 3.6), (185, 228, 3.3),
    # Row 2 (Center perspective line)
    (24, 310, 5.6), (64, 322, 5.0), (100, 332, 4.5), (132, 340, 4.0), (159, 347, 3.6), (182, 353, 3.3),
    # Row 3 (Right-center perspective line)
    (32, 465, 5.4), (72, 452, 4.8), (108, 441, 4.3), (138, 432, 3.9), (164, 424, 3.5),
    # Row 4 (Far-right ceiling bay)
    (45, 668, 5.3), (88, 652, 4.7), (124, 638, 4.2), (154, 626, 3.8), (180, 616, 3.4),
]
for cy, cx, r0 in test_lamps:
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    # Soft ambient ceiling glow around recessed fixture + sharp round white emitter disk
    amb = 0.025 * np.exp(-0.5 * (dist / (r0 * 2.8)) ** 2)
    disk = np.clip((r0 + 0.5 - dist) / 0.9, 0.0, 1.0)
    scene_b = np.clip(scene_b + amb[..., None], 0.0, 1.0)
    scene_b = scene_b * (1.0 - disk[..., None]) + 1.0 * disk[..., None]

scene_b_im = Image.fromarray((np.clip(scene_b, 0, 1) * 255).astype(np.uint8))
scene_b_im.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/unprocessed_test_gallery.png")

r_b = Renderer(StillSource(scene_b), Wb, Hb, 25.0, params, seed=20240102)
sim_b = np.clip(r_b.render_frame(0), 0.0, 1.0)
sim_b_im = Image.fromarray((np.clip(sim_b, 0, 1) * 255).astype(np.uint8))
sim_b_im.save("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/simulated_test_gallery.png")

# Save standalone JSON config for sony-cctv-composite.json
os.makedirs("profiles", exist_ok=True)
cfg_export = {
    "kind": "atsim.config",
    "version": 1,
    "name": "Sony 工业枪机 CCD（BNC 转 AV 复合视频采集卡直采）",
    "standard": "pal",
    "seed": 20240101,
    "fps": 25.0,
    "params": FACTORY_PRESETS["sony_cctv_composite"]["layers"],
}
with open("profiles/sony-cctv-composite.json", "w", encoding="utf-8") as f:
    json.dump(cfg_export, f, ensure_ascii=False, indent=2)
with open("/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/sony-cctv-composite.json", "w", encoding="utf-8") as f:
    json.dump(cfg_export, f, ensure_ascii=False, indent=2)

# 3. Build comprehensive 3-row comparison figure with clean non-overlapping typography
pw, ph = 460, 368
gap = 16
header_h = 46
row_title_h = 30
caption_h = 42

row1_h = ph + row_title_h + caption_h
crop_h = 310
row2_h = crop_h + row_title_h + caption_h
ph3 = 368
row3_h = ph3 + row_title_h + caption_h

total_w = pw * 3 + gap * 4
total_h = header_h + row1_h + row2_h + row3_h + gap * 3

canvas = Image.new("RGB", (total_w, total_h), (18, 20, 24))
draw = ImageDraw.Draw(canvas)
f_title = load_font(19)
f_sub   = load_font(14)
f_cap1  = load_font(12)
f_cap2  = load_font(11)

draw.text((gap, 12), "Sony 工业枪机 CCD + C 口镜头 + BNC/AV 复合视频采集卡：物理机制建模与无效果图片测试对比", fill=(242, 244, 248), font=f_title)

# --- Row 1: Scene A Full Frame ---
y0 = header_h
draw.text((gap, y0 + 4), "一、同场景正向物理重建验证：无像差纯白圆灯输入图（左）→ 物理管线渲染输出（中）→ 实拍参考原图（右）", fill=(130, 205, 255), font=f_sub)
y_img1 = y0 + row_title_h

def stats_str(arr):
    l = arr @ LUMA
    rmb = arr[..., 0] - arr[..., 2]
    pl = l[67:97, 102:136]
    prb = rmb[67:97, 102:136]
    return f"全图均值 {l.mean():.3f} | 暗环下限 {pl.min():.3f} | 峰值 {pl.max():.3f} | 暖边 R-B {prb.max():+.2f} | 蓝尾 B-R {(-prb.min()):+.2f}"

row1_panels = [
    ("1. 无效果处理的干净输入图 (纯白圆形顶灯 · 无像差/无暗环)", clean_a_im, clean_a),
    ("2. 物理管线仿真输出 (sony_cctv_composite 预设)", sim_a_im, sim_a),
    ("3. 实拍参考原图 (Sony 工业枪机 CCD + BNC 转 AV 采集)", real_a_im, real_a),
]
for i, (title, im_p, arr_p) in enumerate(row1_panels):
    x = gap + i * (pw + gap)
    resized = im_p.resize((pw, ph), Image.LANCZOS)
    canvas.paste(resized, (x, y_img1))
    draw.text((x + 2, y_img1 + ph + 4), title, fill=(235, 238, 242), font=f_cap1)
    draw.text((x + 2, y_img1 + ph + 21), stats_str(arr_p), fill=(172, 182, 196), font=f_cap2)

# --- Row 2: 2.2x Magnified Lamp Array Comparison ---
y0_2 = y0 + row1_h + gap
draw.text((gap, y0_2 + 4), "二、顶灯阵列 2.2× 像素级放大对比（展示赛德尔彗差三瓣扇形、自动光圈叶片衍射暗缺口、VOD 溢出漏极黑弧与红/蓝非对称色散）", fill=(130, 205, 255), font=f_sub)
y_img2 = y0_2 + row_title_h

def make_zoom_panel(im_src):
    l_c = im_src.crop((0, 0, 235, 255)).resize((282, crop_h), Image.NEAREST)
    r_c = im_src.crop((585, 0, 720, 255)).resize((172, crop_h), Image.NEAREST)
    pan = Image.new("RGB", (pw, crop_h), (28, 30, 36))
    pan.paste(l_c, (0, 0))
    pan.paste(r_c, (288, 0))
    return pan

row2_panels = [
    ("1. 干净输入图 2.2× 放大（左侧灯阵 + 右侧灯列：标准对称白圆）", make_zoom_panel(clean_a_im),
     "输入特征：对称无色差纯白圆盘 (R=G=B=1.0)，无前缘黑弧、无彗差扇尾、无指状分瓣"),
    ("2. 物理模型仿真 2.2× 放大（自动生成三瓣彗尾 + U型黑弧 + 暖顶/蓝尾）", make_zoom_panel(sim_a_im),
     "物理机制：近处大灯三射线交叠成宽帽带底凹口，远处小灯分离为三指状亮瓣 + 100 IRE 平台"),
    ("3. 实拍原图 2.2× 放大（Sony Color Video Camera 实拍细节）", make_zoom_panel(real_a_im),
     "实拍对照：左上 L/U 型防晕暗环、上缘赤橙色散、右下钴蓝彗尾及大/小灯分瓣形态吻合"),
]
for i, (title, im_p, sub_txt) in enumerate(row2_panels):
    x = gap + i * (pw + gap)
    canvas.paste(im_p, (x, y_img2))
    draw.text((x + 2, y_img2 + crop_h + 4), title, fill=(235, 238, 242), font=f_cap1)
    draw.text((x + 2, y_img2 + crop_h + 21), sub_txt, fill=(172, 182, 196), font=f_cap2)

# --- Row 3: Generalization Test on Independent Unprocessed Scene B ---
y0_3 = y0_2 + row2_h + gap
draw.text((gap, y0_3 + 4), "三、无效果处理的独立测试图泛化验证：在全新展厅长廊透视灯阵图上直接运行同一物理模型（无任何位置硬编码）", fill=(130, 205, 255), font=f_sub)
y_img3 = y0_3 + row_title_h

b_zoom = Image.new("RGB", (pw, ph3), (28, 30, 36))
bz_clean = scene_b_im.crop((130, 10, 480, 195)).resize((pw, ph3 // 2 - 3), Image.NEAREST)
bz_sim   = sim_b_im.crop((130, 10, 480, 195)).resize((pw, ph3 // 2 - 3), Image.NEAREST)
b_zoom.paste(bz_clean, (0, 0))
b_zoom.paste(bz_sim, (0, ph3 // 2 + 3))

row3_panels = [
    ("1. 无效果处理的独立测试原图（四列透视吸顶灯阵 + 立柱 + 安全出口）", scene_b_im.resize((pw, ph3), Image.LANCZOS),
     "测试输入：无像差展厅场景，包含由近及远四列不同视场位置的圆形纯白光源"),
    ("2. 同一物理模型测试输出（sony_cctv_composite 预设直接渲染）", sim_b_im.resize((pw, ph3), Image.LANCZOS),
     "泛化结果：随视场角自动生成赛德尔彗差扇形、VOD 前缘暗环、立柱孔径过冲与采集卡黑边"),
    ("3. 独立测试图局部 2.2× 放大对比（上：未处理原图 ｜ 下：物理模型输出）", b_zoom,
     "细节验证：近处大光源自动呈宽楔形帽、远处小光源自动分裂为三指状亮瓣并伴随上暖下蓝"),
]
for i, (title, im_p, sub_txt) in enumerate(row3_panels):
    x = gap + i * (pw + gap)
    canvas.paste(im_p, (x, y_img3))
    draw.text((x + 2, y_img3 + ph3 + 4), title, fill=(235, 238, 242), font=f_cap1)
    draw.text((x + 2, y_img3 + ph3 + 21), sub_txt, fill=(172, 182, 196), font=f_cap2)

out_fig = "/working_dir/c_8f1524494f0b64f7/artifacts/file_generation/ttl=63d/output/sony_cctv_physical_model_comparison.png"
canvas.save(out_fig)
print("Saved", out_fig)
