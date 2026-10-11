"""
The five layers of the virtual signal chain, as a parameter schema.

    A camera   -> "what shot it"        (before a signal exists)
    B format   -> "how the signal was made"  (bandwidth, composite coding)
    C tape     -> "what the medium did to it" (transport, dropout, wear)
    D player   -> "how it was read back"      (TBC, head, output level)
    E display  -> "how it was watched"        (CRT, gamma, overscan, glass)

Values are all nominal 0..1 unless a physically meaningful unit is given.
Bounds are UI ranges; `phys_min`/`phys_max` mark the physically plausible zone,
so the UI can warn when a value has left the calibrated envelope (the
"physical model as floor, artistic deviation allowed" decision).
"""
from __future__ import annotations

from .params import ParamSchema, merge_values

LAYER_LABELS = {
    "camera": "01 成像系统 (摄像机)",
    "format": "02 信号编码 (格式)",
    "connector": "02c 链路接口",
    "tape": "03 记录介质 (磁带)",
    "player": "04 读取解码 (播放设备)",
    "display": "05 显示设备 (CRT)",
}


def build_schema() -> ParamSchema:
    s = ParamSchema()

    # ------------------------------------------------------------------ A
    cam = s.layer("camera")
    g = "传感器"
    cam.define("sensor_noise", "传感器噪声", 0.06, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.35, source="CCD/CMOS 消费级摄像机估算",
               help="光子散粒噪声与读出噪声的合成电平。")
    cam.define("noise_grain", "噪声颗粒尺度", 1.4, 1.0, 6.0, unit="px", group=g,
               phys_min=1.0, phys_max=3.0, source="传感器像素间距估算",
               help="噪声的空间相关长度；1.0 为逐像素白噪声。")
    cam.define("fixed_pattern", "固定图案噪声", 0.05, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.25, source="CCD 列固定图案噪声估算",
               help="不随时间变化、固定在传感器位置上的噪点。")
    cam.define("dynamic_range", "动态范围", 8.0, 4.0, 14.0, unit="stops", group=g,
               phys_min=6.0, phys_max=12.0, source="1990 年代 CCD 约 8 档",
               help="高光滚降起点；越小越容易过曝成一片死白。")
    g = "曝光与白平衡"
    cam.define("exposure", "曝光", 0.0, -2.0, 2.0, unit="stops", group=g,
               phys_min=-1.0, phys_max=1.0, source="自动曝光残余误差")
    cam.define("white_balance_temp", "白平衡色温偏移", 0.0, -1.0, 1.0, unit="0-1", group=g,
               phys_min=-0.5, phys_max=0.5, source="消费级 AWB 偏差",
               help="负值偏冷，正值偏暖（室内灯光下常见的偏色）。")
    g = "镜头"
    cam.define("sharpness", "镜头锐度", 0.75, 0.0, 2.0, unit="0-2", group=g,
               phys_min=0.4, phys_max=1.2, source="消费级变焦镜头 MTF")
    cam.define("chroma_aberration", "色差", 0.05, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.3, source="低端镜头横向色差估算")
    cam.define("motion_blur", "运动模糊", 0.15, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.6, source="1/60s 快门")
    cam.define("ccd_smear", "CCD 垂直拖尾", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="隔行 CCD 读出期间的电荷泄漏",
               help="强光在垂直方向拉出的长条拖影。消费级 CCD 在夜间拍摄灯光时非常明显。")
    cam.define("halation", "高光光晕/泛光", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.9, source="CCD 过饱和溢出 + 镜头内反射",
               help="过曝区域的扩展光晕，会带出紫红色边缘。与显示阶段的辉光不同：它发生在传感器上。")
    cam.define("ccd_gain", "增益档位", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=1.0, source="AGC 在低照度下提升增益",
               help="低照度时自动增益提升，会同时放大读出噪声。这是夜间画面颗粒感的来源。")
    g = "色彩科学"
    cam.define("color_cast", "色彩倾向", 0.0, -1.0, 1.0, unit="0-1", group=g,
               phys_min=-0.5, phys_max=0.5, source="各品牌 CCD 滤色片与 DSP 色调",
               help="负值偏青蓝（Sony 消费级常见），正值偏暖。这会整体改变画面的时代感。")
    cam.define("highlight_tint", "高光染色", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="CCD 过曝时的通道非线性",
               help="过曝区域向品红/紫偏移——参考素材里灯光边缘的那圈紫边。")
    g = "镜头"
    cam.define("lens_wear", "镜头磨损/镀膜老化", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="镜片镀膜磨损与内部起雾",
               help="镀膜磨损不是模糊：它把光散射掉，因此同时造成对比度下降、黑位抬升（雾化），"
                    "并且因为抗反射层正是抑制鬼影的，鬼影会随之加重。")
    cam.define("lens_flare", "镜头光斑/眩光", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="镜头内部多次反射",
               help="按光圈叶片形状成像的几何鬼影与星芒。注意与「高光光晕」不同："
                    "后者是传感器过曝的圆形溢光，这个是光学孔径形状。")
    cam.define("flare_blades", "光圈叶片数", 6.0, 3.0, 10.0, unit="片", group=g,
               phys_min=5.0, phys_max=9.0, step=1.0, source="常见 6/7/8/9 片光圈",
               help="星芒的射线数。6 片给出六角星芒。")
    g = "传感器类型"
    cam.define("sensor_type", "3CCD 传感器", 0.0, 0.0, 1.0, unit="bool", group=g,
               phys_min=0.0, phys_max=1.0, step=1.0, source="3CCD 机型（如 Panasonic NV-GS78）",
               help="三片传感器经棱镜分光，每片独立读出原色。特征不是「更好的单 CCD」，"
                    "而是噪点更低（读出噪声表现为亮度颗粒而非彩色斑点）、色彩分离干净、"
                    "细密纹理上无伪色摩尔纹。")

    # ------------------------------------------------------------------ B
    fmt = s.layer("format")
    gf = "滤波器家族"
    fmt.define("filter_kind", "滤波器类型", 1.0, 0.0, 2.0, unit="0-2", group=gf,
               step=1.0, source="消费级电路为恒 K 型级联；测试仪器用巴特沃斯",
               help="0=高斯（零相位，理想，无色边）｜1=恒 K 型（真实电路，有群延迟）"
                    "｜2=巴特沃斯（拐点更陡）。群延迟使色度相对亮度右移，"
                    "这是硬边出现色边的成因之一——零相位滤波做不出这个效应。")
    fmt.define("filter_sections", "级联级数", 3.0, 1.0, 5.0, unit="级", group=gf,
               phys_min=2.0, phys_max=4.0, step=1.0, source="恒 K 型通常为 3 级级联")
    g = "色度通道增益"
    fmt.define("chroma_gain", "色度增益（ACC）", 1.0, 0.0, 3.0, unit="×", group=g,
               phys_min=0.7, phys_max=1.4, source="ACC/APC 自动色度控制的失控范围",
               help="色度通道的整体幅度，也是「饱和度」真正的物理来源：自动色度控制"
                    "（ACC）以副载波色同步的幅度为基准把色度拉回标准电平，"
                    "一旦基准本身出错，色度就会被系统性地放大或压死。"
                    "1.0 = 标准；2 以上是色度卡故障式的过饱和；0.1 以下近乎黑白"
                    "（色度消隐/killer 动作）。参考素材的色度故障片饱和度达 0.87，"
                    "而真实 DV 只有 0.21。")
    g = "模拟复合视频"
    fmt.define("luma_bandwidth", "亮度带宽", 3.0, 0.1, 6.0, unit="MHz", group=g,
               phys_min=2.0, phys_max=5.0, source="VHS 约 3.0 MHz / S-VHS 约 5.0 MHz",
               help="决定细节衰减。VHS 白电平区带宽约 3 MHz，色度区更低。")
    fmt.define("chroma_bandwidth", "色度带宽", 0.75, 0.25, 2.0, unit="MHz", group=g,
               phys_min=0.3, phys_max=1.5, source="VHS 色度欠载记录约 0.6-1.0 MHz",
               help="色度以欠载方式记录，带宽远低于亮度，是 VHS 颜色发糊的根源。")
    fmt.define("chroma_vertical_averaging", "色度垂直平均", 0.85, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.5, phys_max=1.0, source="色度约 40 行，接收机逐行平均",
               help="相邻行色度混合的强度，造成垂直方向的彩边。")
    fmt.define("comb_depth", "梳状滤波深度", 0.55, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.2, phys_max=0.9, source="NTSC 接收机梳状滤波器",
               help="1H 延迟梳状滤波的权重；越深串色越少，但垂直亮度分辨率下降。")
    fmt.define("cross_color", "交叉色彩 (亮度→色度)", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=1.0, source="复合信号分离残余",
               help="高频亮度细节被误判为色度，形成彩虹状彩边。")
    fmt.define("cross_luma", "交叉亮度 (色度→亮度)", 0.30, 0.0, 1.5, unit="0-1", group=g,
               phys_max=1.0, source="复合信号分离残余",
               help="色度残留进亮度，形成爬行点状花纹 (dot crawl)。")
    fmt.define("edge_boost", "轮廓增强", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="摄像机/录像机轮廓补偿电路",
               help="录像机的边缘增强，会在高对比边缘产生白边。")
    fmt.define("color_noise", "色度噪声", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="色度欠载通道信噪比",
               help="色度通道噪声远高于亮度通道，是彩点噪声的来源。")
    fmt.define("burst_phase_error", "色同步相位误差", 5.0, 0.0, 45.0, unit="deg", group=g,
               phys_max=20.0, source="色同步与色度相位误差",
               help="整场的色相旋转/漂移。")

    # ----------------------------------------------------- B2 数字编码分支
    cod = s.layer("codec")
    g = "编码分支"
    cod.define("codec_kind", "编码器选择", 0.0, 0.0, 2.0, unit="0-2", group=g,
               step=1.0, phys_min=0.0, phys_max=2.0,
               source="0=不编码（模拟）｜1=DV25｜2=MPEG-1（Video CD）",
               help="选择数字编码分支。0 为模拟通路（VHS/Hi8），不经任何数字编码；"
                    "1 为 DV25；2 为 MPEG-1，也就是 Video CD 用的编码。"
                    "三种通路的伪影结构不同，不能互相近似：DV 是 4:1:1 + 固定码率，"
                    "MPEG-1 是 4:2:0 + 粗量化 + 运动补偿预测。")
    g = "MPEG-1 / Video CD"
    cod.define("vcd_quality", "VCD 码率质量", 1.0, 0.25, 2.0, unit="x", group=g,
               phys_min=0.6, phys_max=1.4,
               source="VCD 固定 1150 kbit/s，352x240",
               help="量化步长的缩放。VCD 的码率只有 DV25 的约二十分之一，"
                    "所以量化粗得多——高频被**成结构地丢弃**，而不是平滑衰减，"
                    "这是它看起来又软又带有块感的原因。")
    cod.define("vcd_width", "VCD 宽度", 352.0, 176.0, 720.0, unit="px", group=g,
               step=1.0, phys_min=352.0, phys_max=352.0,
               source="VCD（MPEG-1）352x240，525/60",
               help="编码前的水平尺寸。VCD 标准固定 352，物理区间也只允许这一个值——"
                    "它不是为了调节画面，而是为了说明 MPEG-1 是在**四分之一分辨率**上"
                    "做运动估计与量化的，这本身就是它伪影粗的一个原因。")
    cod.define("vcd_height", "VCD 高度", 240.0, 120.0, 576.0, unit="px", group=g,
               step=1.0, phys_min=240.0, phys_max=288.0,
               source="VCD（MPEG-1）352x240，525/60；625/50 为 352x288",
               help="编码前的垂直尺寸。525/60 为 240，625/50 为 288。")
    cod.define("vcd_motion_comp", "运动补偿强度", 1.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="P 帧从上一帧的**已解码亮度**做运动预测的程度。"
                    "这是 VCD 最独特的伪影来源：预测失败的地方（遮挡、硬切、运动边缘）"
                    "残差大、误差可见，而预测成功的地方画面很干净——"
                    "所以伪影不是逐帧独立的噪声，而是**跟着预测成败走**。"
                    "取 0 退化为逐帧帧内编码，也就失去了这个特征。")
    cod.define("vcd_bitrate_kbps", "VCD 码率", 1150.0, 600.0, 2000.0, unit="kbps",
               group=g, step=50.0, phys_min=1100.0, phys_max=1200.0,
               source="VCD 标准 1150 kbit/s",
               help="参考资料值。当前实现用它设定量化尺度，并未真正做码率控制。")
    g = "DV / MiniDV 编码"
    cod.define("dv_enabled", "启用 DV 编码分支", 0.0, 0.0, 1.0, unit="bool", group=g,
               phys_min=0.0, phys_max=1.0, step=1.0, source="MiniDV / DVCAM / DVCPRO 机型",
               help="开启后画面走数字分量分支（4:1:1 子采样 + DCT 量化），不再经过复合视频载波。"
                    "MiniDV 机型如 Sony DCR-TRV310E 必须开启此项。")
    cod.define("chroma_mode", "色度子采样", 1.0, 0.0, 1.0, unit="bool", group=g,
               phys_min=0.0, phys_max=1.0, step=1.0, source="525/60 用 4:1:1，625/50 用 4:2:0",
               help="开=按制式选择（NTSC 4:1:1 / PAL 4:2:0），关=固定 4:1:1。"
                    "4:1:1 只损失横向色度分辨率，4:2:0 两个方向都损失。")
    cod.define("dv_quality", "DV 码率质量", 1.0, 0.25, 2.0, unit="x", group=g,
               phys_min=0.7, phys_max=1.2, source="DV25 固定 25 Mbit/s",
               help="1.0 为标准 DV25。调低等于码率更紧（块状伪影与蚊式噪声更重）。")
    cod.define("dv_base_step", "DV 量化基准", 0.0125, 0.002, 0.06, unit="0-1", group=g,
               source="DV 量化表量级",
               help="量化步长基准。暗部块状伪影的强度主要由它决定。")

    # ------------------------------------------------------------ RF 链路
    rf = s.layer("rf")
    g = "射频通道"
    rf.define("channel_mhz", "频道带宽", 6.0, 5.0, 8.0, unit="MHz", group=g,
              phys_min=6.0, phys_max=8.0, source="M/NTSC 6 MHz，PAL/B/G 8 MHz",
              help="单个电视频道的宽度。决定了视频信息能占用的最高频率。")
    rf.define("carrier_offset_mhz", "图像载波位置", 1.25, 0.5, 2.5, unit="MHz", group=g,
              phys_min=1.0, phys_max=1.5, source="M/NTSC 图像载波距频道下沿 1.25 MHz")
    rf.define("signal_dbm", "接收电平", -70.0, -90.0, -30.0, unit="dBm", group=g,
              phys_min=-80.0, phys_max=-40.0, source="天线输入电平",
              help="与噪声系数共同决定载噪比 C/N。本工具按 6 MHz 频道计算："
                   "C/N ≈ 接收电平 + 98 dB − 噪声系数。默认 −70 dBm 对应 C/N ≈ 28 dB，"
                   "略低于失锁阈值，因此画面底部会出现 RF 特有的撕裂与彩色错乱。"
                   "调到 −55 dBm 以上画面即干净；调到 −85 dBm 会整幅失去同步。")
    rf.define("noise_figure_db", "调谐器噪声系数", 8.0, 2.0, 20.0, unit="dB", group=g,
              phys_min=4.0, phys_max=14.0, source="消费级电视高频头",
              help="与接收电平共同决定载噪比 C/N：噪声系数每高 1 dB，C/N 就低 1 dB。")
    rf.define("sync_loss_threshold_db", "失锁阈值", 30.0, 18.0, 44.0, unit="dB", group=g,
              phys_min=24.0, phys_max=38.0, source="行 AFC 保持范围",
              help="C/N 低于此值时行同步开始打滑，出现横向撕裂与彩色错乱。")
    rf.define("sync_loss_amount", "失锁强度", 1.0, 0.0, 2.0, unit="0-2", group=g,
              phys_min=0.5, phys_max=1.5)
    rf.define("adjacent_channel", "邻频干扰", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_max=0.5, source="相邻频道载波差拍",
              help="与图像载波差拍形成的斜向人字形条纹，会缓慢移动。")

    # ------------------------------------------------------------------ C
    tpe = s.layer("tape")
    g = "录制速度"
    tpe.define("tape_speed", "录制速度", 0.0, 0.0, 2.0, unit="0-2", group=g,
               step=1.0, source="VHS 标准 SP/LP/EP（同机同一盘磁带的速度档）",
               help="0=SP 标准｜1=LP 长时｜2=EP 超长。注意这不是「把质量调低」："
                    "磁鼓转速不变，所以写入的 FM 偏离几乎不变，下降的是走带速度。"
                    "由此同时产生四种后果——重放输出下降（LP 约 −6 dB、EP 约 −10 dB）、"
                    "磁迹变窄使失落概率上升（约 2×/3×）、保护带变窄使跟踪敏感度剧增"
                    "（约 2.2×/3.4×）、控制轨迹信号变弱使时基更不稳（约 1.9×/2.8×）。"
                    "色度欠载方式不变，所以色度带宽不随速度改变——这也是为什么"
                    "LP/EP 素材里亮度电平基本不变，坏的是色度分层与稳定度。")
    g = "走带与时基误差"
    tpe.define("tape_stability", "走带稳定性", 0.55, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0, source="磁带盒与走带机构机械状态",
               help="1.0 为机械完美。这是逐行水平位移的主要来源。")
    tpe.define("line_jitter", "行抖动", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="行频相位抖动",
               help="每行独立的水平位移，造成竖线边缘的锯齿状波动。")
    tpe.define("walk_amplitude", "低频漂移", 0.30, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="走带速度缓慢波动",
               help="缓慢相关的过程，造成整幅画面缓慢的水平摆动。")
    tpe.define("micro_jitter", "微小抖动", 0.20, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.5, source="磁鼓相位噪声",
               help="亚像素级的行内不齐，由插值相位误差表现。")
    tpe.define("vertical_instability", "垂直不稳定", 0.08, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.4, source="场同步裕度不足",
               help="整场的垂直跳动，偶发出现。")
    g = "磁头切换"
    g = "走带模式（快进/倒带）"
    g = "磁粉与霉变"
    tpe.define("binder_loss", "黏合剂失效（霉变/脱粉）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="潮湿存放导致黏合剂吸水、磁粉层脱落",
               help="磁粉层脱落的面积比例。这不是「脏」而是**涂层本身缺失**，"
                    "所以有两个复合后果：受损区域固定在磁带表面、反复出现在同一位置"
                    "（不像噪声那样每帧重掷）；且**色度先死**——色度是低幅度的欠载"
                    "载波，骑在亮度上，涂层变薄时它先消失而亮度还能撑一阵。"
                    "参考素材的「发霉」样本实测饱和度崩到 0.053、高频升高 7 倍、"
                    "38% 画面接近全黑——是「高频且暗」，不是单纯变糊。")
    g = "涂层脱落"
    tpe.define("shed_grain", "脱粉颗粒", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="参考素材发霉格：高频为正常格的 3 倍",
               help="磁粉脱落使磁头读到更粗糙的表面，重放信号带有宽带颗粒。"
                    "它与「粒子噪声底」的区别有两点：**强得多且更细**"
                    "（参考发霉格高频 0.116 而正常格 0.039；噪声底拉满只能到 0.035，"
                    "差六倍，所以那种样子根本做不出来），以及**只作用于亮度**——"
                    "涂层脱落会直接毁掉微弱的色度载波，颗粒若带上彩色就错了。"
                    "它与「色度增益」「消隐电平」共同构成发霉格的三要素："
                    "去饱和 + 变暗 + 高频颗粒。")

    g = "亮度电平"
    tpe.define("luma_pedestal", "消隐电平偏移", 0.0, -0.5, 0.3, unit="0-1", group=g,
               phys_max=0.05, phys_min=-0.35,
               source="直流恢复/消隐钳位失常（S-VHS 烂碟实测约 -0.30）",
               help="亮度基底（消隐电平）的整体偏移。它与「磁粉脱落」是两回事："
                    "亮度是**基带电平**，绝对值直接承载明暗，必须钳位到参考；"
                    "色度是**围绕该参考的调制幅度**，参考错了它照样活着。"
                    "所以信号丢失让画面**又暗又灰**（亮度色度一起没），"
                    "而钳位失效让画面**又暗又极其鲜艳**——"
                    "参考素材实测饱和度升到 0.94、均值掉到 0.09。")
    tpe.define("drum_speed", "磁鼓转速", 1.0, 0.80, 1.05, unit="x", group=g,
               phys_min=0.97, phys_max=1.03,
               source="松下肩扛式 VHS / VHS-C 磁鼓轴承油变质（十台九台）",
               help="磁鼓额定转速的比值。注意它与走带误差是**不同轴**的问题："
                    "走带（主导轴）误差让画面横向抖动，磁鼓转速误差改变磁头"
                    "向下扫过磁迹的快慢——鼓慢了，相邻输出行就取自磁带上几乎"
                    "相同的位置，画面出现**微量重复的行**。垂直方向被平滑、"
                    "水平方向不变，这正是「横向颗粒」的来源，而不是随机噪声。"
                    "继续使用会导致轴承不可逆磨损。")

    g = "清洁度与污染"
    tpe.define("head_clog", "磁头堵塞", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="磁头积垢",
               help="堵塞的磁头拖出一条固定的丢失信号带，画面出现起格或条纹。")
    tpe.define("clog_bands", "堵塞条带数", 3.0, 1.0, 8.0, unit="条", group=g, step=1.0)
    tpe.define("debris", "磁粉颗粒", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="脱落的磁粉与灰尘",
               help="松散颗粒短暂撞击磁带，形成断续的短促丢失。")
    tpe.define("surface_damage", "表面划伤", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="走带划伤",
               help="划伤损坏一条窄的、缓慢漂移的行带。")
    tpe.define("chroma_loss", "色度丢失行", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.4, source="色度欠载载波弱",
               help="随机整行丢失彩色（色度载波弱于亮度）。")

    tpe.define("row_timebase", "逐行时基", 0.0, 0.0, 3.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.5,
               source="未标定。曾引用 JVC SR-S388E 素材的三个读数，该证据已撤回（见 help）",
               help="整行而非整场的水平位移，横向的「撕裂」感来源。"
                    "**量级未标定，且这不是遗漏而是结论。** 曾据 JVC SR-S388E 采集的三个读数"
                    "（t=1000s 时 std 0.19 px、1380s 时 0.46、2000s 时 1.03）给出 0.02/0.04/0.08，"
                    "但同一素材按每 5 秒重采样后发现：**相邻采样的逐行散布在约 0.1 与 2.9 px 之间"
                    "以 6x、12x、14x 的幅度翻转，且与画面内容无关**。具有这种行为的量无法用三个"
                    "读数概括，每个读数都只是一个闪烁信号的采样。所用追踪器后来也被测出非线性"
                    "（平滑场恢复 0.47x、快变场 3.11x），所以它给出的数字本来就不是绝对像素位移。"
                    "**因此本参数保持 0.0，并且在拿到状态稳定的素材之前不应标定**——"
                    "素材显示的是**间歇性**，那由 stiction_rate 单独建模。"
                    "与行抖动不同：行抖动是每行独立，这里是整行整体移动。")
    tpe.define("stripe_moire", "卷曲条纹强度", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.6,
               source="残余载波与视频通路的差频干涉（偏磁/伴音副载波泄漏、邻频、时钟不同步）",
               help="沿画面弯曲的条纹。它与「整行时基误差」是**不同现象**，这一点是量出来的："
                    "逐行位移让跨行相位变化达到完好基线的 3990 倍，"
                    "但**行内曲率仍停在地板**（0.00001 rad/px²）——"
                    "位移只是把内容横向搬走，不改变空间频率，所以直线条纹依然是直线。"
                    "要让条纹弯曲必须有**差频**：残余载波与视频通路频率略有差异，"
                    "两者相位差随位置变化，条纹因此弯曲。")
    tpe.define("stripe_beat", "条纹间距", 0.055, 0.01, 0.25, unit="cyc/px", group=g,
               phys_min=0.02, phys_max=0.12,
               help="差频决定的条纹疏密。数值越小条纹越宽。")
    tpe.define("stripe_curve", "条纹弯曲程度", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="条纹沿画面方向的弯曲量。由相位差随位置变化的快慢决定；"
                    "取 0 得到一组平行的直条纹，与「整行位移」外观不同。")

    g = "磁带仓受潮（粘带）"
    tpe.define("stiction_rate", "粘带发生率", 0.0, 0.0, 4.0, unit="次/秒", group=g,
               phys_min=0.0, phys_max=1.2,
               source="实测：相邻场逐行位移在 0.1 与 3 px 之间以数十倍幅度翻转，"
                      "且与画面内容无关",
               help="磁带黏合剂吸水，在主导轴与压带轮之间形成水膜，产生黏附。"
                    "走带机构继续拉，张力升高到黏附突然失效，磁带猛跳一下，然后再粘住。"
                    "**这是间歇性故障**，不是平滑劣化：事件之间磁带表现正常。"
                    "模型里它是一个**事件列表**（和失落、折痕一样），"
                    "而不是逐帧场——把场平滑掉恰好会抹掉让它看起来像故障的那些突变。")
    tpe.define("stiction_amount", "粘带强度", 0.6, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="每次黏附的强度。位移在抓取过程中**渐增**、释放时**回弹**，"
                    "所以是锯齿波——不是阶跃（那会像接头），也不是正弦（那会像抖动）。"
                    "空间上位移呈斜坡，所以受影响的行是渐变的，而不是一条整齐的裂口。")
    tpe.define("stiction_stretch", "沿带加重", 0.0, -0.8, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="发生率沿带长的偏移。湿气分布不均、靠近卷芯的部分更潮，"
                    "而且播放最多的位置磨损最重，所以同一盘带前后段的故障密度不同。"
                    "注意这影响的是**事件发生率**而不是每帧幅度——故障在事件里，不在信号里。")
    g = "整行色度错误"
    tpe.define("row_chroma_error", "整行错色发生率", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               source="熟悉设备的用户描述：解析出错误的颜色，而且是整行全错",
               help="整条扫描线解出**错误的颜色**——不是没有颜色。"
                    "它与「色度丢失行」是两回事：那个把色度清零，得到灰线；"
                    "这个旋转色相并改变色度增益，线仍然是饱和的，只是颜色错了。"
                    "机制上这是**参考错误**：色度副载波的相位由每行起始的色同步恢复，"
                    "所以某一行的参考错了，会把那一行上**所有**颜色旋转同一个角度，"
                    "而不是各自独立地坏掉。")
    tpe.define("row_chroma_phase", "整行色相误差", 0.0, 0.0, 90.0, unit="deg", group=g,
               phys_min=0.0, phys_max=45.0,
               help="受影响行的色相旋转幅度。因为它是旋转而不是缩放，"
                    "饱和度得以保留——这正是「错色」与「去色」在画面上的区别。")
    tpe.define("row_chroma_gain", "整行色度增益误差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="受影响行的色度幅度误差。色同步幅度经 ACC 环路决定色度增益，"
                    "所以弱的色同步会让整行过饱和或欠饱和。与色相误差相互独立。")
    g = "失落与磁头切换"
    tpe.define("dropout_rate", "失落率", 0.35, 0.0, 4.0, unit="次/秒", group=g,
               phys_max=1.5, source="磁粉脱落点的密度",
               help="Poisson 到达过程。每次失落都有确定的位置与长度，"
                    "因此相同种子会重现完全相同的缺陷时间线。")
    tpe.define("dropout_len", "失落长度", 2.5, 0.5, 30.0, unit="行", group=g,
               phys_max=8.0)
    tpe.define("dropout_color", "失落补色残留", 0.45, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="失落补偿电路残留")
    tpe.define("dropout_softness", "失落边缘柔和度", 0.35, 0.0, 1.0, unit="0-1", group=g)
    tpe.define("head_switching", "磁头切换强度", 0.85, 0.0, 1.2, unit="0-1", group=g,
               phys_max=0.8, source="两个视频磁头切换期间的信号缺失",
               help="画面底部那条细条，VHS 最有辨识度的痕迹之一。"
                    "0.5 左右得到「细而清晰」的一条，超过 0.8 会变成大块噪点。")
    tpe.define("head_switch_lines", "切换带行数", 16.0, 2.0, 40.0, unit="行/场", group=g,
               phys_min=4.0, phys_max=16.0, source="VHS 磁头切换约 5-15 行/场",
               help="每场末尾受影响的行数。因隔行交织，画面中表现为底部 "
                    "2x 该行数 的一条带：8 行/场 → 底部约 16 行 ≈ 画面的 3.3%。")
    tpe.define("head_switch_mid_line", "行中途切换", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="切换点落在行中间的情况")

    g = "介质磨损"
    tpe.define("wear", "磁带磨损", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="磁带表面磨损与走带次数",
               help="磨损使磁介质输出下降、表面变粗糙，亮度与色度信噪比一起变差。")
    tpe.define("crease_rate", "折痕发生率", 0.20, 0.0, 3.0, unit="次/秒", group=g,
               phys_max=1.0, source="磁带折痕与走带受压",
               help="折痕降低磁头与磁带的接触，所以它同时软化并削弱亮度与色度。")
    tpe.define("noise_floor", "粒子噪声底", 0.18, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.5, source="磁介质粒子噪声",
               help="磁介质本身带来的宽带噪声。注意它与「脱粉颗粒」不同：这个是一般"
                    "粒子噪声底，强度上限只有脱粉颗粒的约六分之一。")

    g = "跟踪失谐"
    tpe.define("tracking_error", "跟踪误差", 0.45, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="走带控制轨迹与磁头位置的相位差",
               help="把整条切换带横向推开的量，即录像机前面板「跟踪」旋钮的表现。")
    tpe.define("tracking_noise_rows", "失控行数", 0.0, 0.0, 240.0, unit="行", group=g,
               phys_min=0.0, phys_max=120.0, source="控制轨迹与磁头不同步的深度",
               help="底部有多少行失去同步。与磁头切换带的区别：切换带是固定硬边"
                    "细条（约 8 行），跟踪失谐是从某行开始向下渐强的斜坡。")
    tpe.define("tracking_noise_wave", "失控摆动", 0.0, 0.0, 6.0, unit="px", group=g,
               phys_min=0.0, phys_max=3.0, help="失控行在底部的最大横向位移。")
    tpe.define("tracking_noise_snow", "失控噪点", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, help="失控区的雪花与彩色丢失，越靠下越重。")

    g = "走带模式（快进/倒带）"
    tpe.define("shuttle_speed", "走带倍速", 0.0, 0.0, 8.0, unit="x", group=g,
               phys_min=0.0, phys_max=5.0, step=0.5,
               source="NV-GS 系列倒带/快进搜索速度",
               help="0 = 正常播放。非零时磁头掠过磁迹的速度与记录速率不符，"
                    "只能取到部分宏块/扫描线，由解码器或时基电路用缓冲内容补齐。")
    tpe.define("shuttle_stale", "残留比例", 0.0, 0.0, 0.9, unit="比例", group=g,
               phys_max=0.5, source="参考素材实测 15-24%（均值约 19%）")
    tpe.define("shuttle_cluster", "残留成片程度", 0.55, 0.0, 1.0, unit="0-1", group=g)
    tpe.define("shuttle_carry", "残留保持率", 0.45, 0.0, 0.95, unit="0-1", group=g)
    tpe.define("shuttle_mode", "残留粒度", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               source="VHS/Hi8/Video8 为行记录；MiniDV 为宏块",
               help="0=自动（跟随编码层）|1=强制按行|2=强制按宏块。这不是偏好而是"
                    "格式属性：模拟格式沿磁迹记录连续波形，没有可寻址的块结构。")
    tpe.define("shuttle_span", "残留段长", 8.0, 1.0, 60.0, unit="行/段", group=g,
               phys_min=2.0, phys_max=30.0, step=1.0)
    tpe.define("shuttle_wave", "残留行位移", 14.0, 0.0, 60.0, unit="px", group=g,
               phys_min=0.0, phys_max=30.0,
               help="失效行的横向位移，在一段内逐行递增——台阶状边缘的来源。")
    tpe.define("shuttle_reverse", "方向（0 快进/1 倒带）", 0.0, 0.0, 1.0, unit="0/1",
               group=g, step=1.0)
    tpe.define("shuttle_block", "宏块边长", 16.0, 8.0, 32.0, unit="px", group=g,
               step=8.0, phys_min=16.0, phys_max=16.0, source="DV 宏块为 16x16 亮度像素")

    g = "广播链路信噪比"
    rf.define("broadcast_noise", "广播噪声", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7,
               source="1992 淄博电视台弱信号录像：高频为基线 2 倍而饱和仅 0.178",
               help="接收端在解调前的宽带噪声。它与磁带噪声的关键区别是**同时淹没"
                    "亮度与色度**——实测参考素材高频升到 0.024 而饱和度只有 0.178，"
                    "即画面被噪声抹平而不是偏色。磁带劣化则相反：色度先死、"
                    "亮度还撑得住（参考 S-VHS 烂碟：sat 0.94 而 mean 0.09）。"
                    "两者用不同的参数，因为它们本来就不是同一个环节的故障。")

    # ------------------------------------------------------------------
    # Physical connectors between the source and the recorder.  Kept per-connector
    # rather than as one "damage" control, because in a real chain one joint in the run
    # is corroded and the others are not, and locating the fault is the point.
    cn = s.layer("connector")
    g = "链路中的接口"
    cn.define("hdmi_present", "HDMI 接口", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
              phys_min=0.0, phys_max=1.0, source="HDMI 转 AV 转换器的输入",
              help="高速差分数字接口。损坏的表现与模拟接口**完全不同**："
                   "数字没有「轻微雪花」这个状态，位要么对要么错，所以看到的是"
                   "马赛克、局部冻结、闪黑；再差就是握手失败直接「无信号」。")
    cn.define("hdmi_moisture", "HDMI 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="水在缝隙里把介电常数从空气的 1 拉到约 80，单位长度电容上升，"
                   "差分阻抗偏离 100Ω。差分对对此极其敏感，眼图很快闭合。")
    cn.define("hdmi_corrosion", "HDMI 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="氧化层是差的导体，接触电阻上升，结点更像开路而不是匹配负载。")
    g = "AV / RCA 接口"
    cn.define("av_present", "AV 接口", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
              phys_min=0.0, phys_max=1.0, source="RCA 复合视频（黄莲花头）")
    cn.define("av_moisture", "AV 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("av_corrosion", "AV 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("av_ground_loop", "地环路", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.5,
              help="RCA 的**外壳就是地**。外壳接触电阻大，线缆参考电位与信号源不同，"
                   "差值就是市电上的交流声——表现为一条**缓慢滚动**的亮暗条。"
                   "滚动是这类故障的全部特征；静止的条是另一种故障。")
    cn.define("mains_hz", "市电频率", 50.0, 45.0, 65.0, unit="Hz", group=g,
              phys_min=50.0, phys_max=60.0,
              help="滚动的速度取决于市电与场频之**差**，所以 50Hz 与 60Hz 地区的滚动方向"
                   "与速度不同。这解释了同一台设备在不同地区观感不同。")
    g = "S-Video 接口"
    cn.define("svideo_present", "S-Video 接口", 0.0, 0.0, 1.0, unit="bool", group=g,
              step=1.0, phys_min=0.0, phys_max=1.0)
    cn.define("svideo_moisture", "S-Video 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("svideo_corrosion", "S-Video 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("svideo_leakage", "Y/C 串扰", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.5,
              help="Y 与 C 在相隔几毫米的两个针脚上，受潮形成漏电通路。"
                   "结果是**边缘上的彩色镶边**而不是整体偏色：亮度的陡变出现在色度通道里，"
                   "反过来也一样，所以镶边正好落在边缘上——"
                   "这就是为什么它读起来像坏磁带，而源头是湿了的接口。")
    g = "RF / 天线接口"
    cn.define("rf_present", "RF 接口", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
              phys_min=0.0, phys_max=1.0, source="75Ω 同轴天线接口")
    cn.define("rf_moisture", "RF 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="水改变介电常数，阻抗偏离 75Ω，能量被反射回去。反射波与前进波干涉，"
                   "而干涉是相加还是相消取决于**电长度**，所以损伤是**频率选择性的**——"
                   "咬住某些频率而放过另一些，不是整体衰减。")
    cn.define("rf_corrosion", "RF 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="氧化层加潮气可以像一个粗糙的二极管：一个方向比另一个方向更容易导通，"
                   "正弦波被削掉一边。削顶会产生**谐波**，所以腐蚀的 RF 接点会干扰"
                   "**别的频道**——干扰是在接点上产生的，不是从外面收进来的。")
    cn.define("rf_shielding", "屏蔽损失", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.5,
              help="屏蔽层破损会让外部射频进入。与「氧化」的区别：氧化是接点**自己产生**"
                   "干扰，屏蔽损失是**外部**干扰进来。")

    # ------------------------------------------------------------------ D
    ply = s.layer("player")
    g = "时基校正与走带"
    ply.define("transport_jitter", "走带抖动", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="主导轴伺服残差")
    ply.define("tbc_enabled", "时基校正开启", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, source="TBC / 数字时基校正器")
    ply.define("tbc_strength", "时基校正强度", 1.0, 0.0, 1.0, unit="0-1", group=g)
    ply.define("tbc_residual", "校正残差", 0.0, 0.0, 1.0, unit="0-1", group=g)
    ply.define("line_delay", "行延迟", 0.0, -8.0, 8.0, unit="px", group=g,
               phys_min=-3.0, phys_max=3.0, source="色度相对亮度的群延迟")
    g = "磁头与频响"
    ply.define("head_wear", "磁头磨损", 0.30, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="磁头尖磨损",
               help="磨损的磁头损失高频并降低色度信噪比。")
    ply.define("frequency_response", "频响倾斜", 0.15, -1.0, 1.0, unit="0-1", group=g,
               phys_max=0.5)
    ply.define("ringing", "振铃强度", 0.0, 0.0, 1.0, unit="0-1", group=g, phys_max=0.5)
    ply.define("ringing_freq", "振铃频率", 0.5, 0.1, 0.95, unit="xNyquist", group=g)
    ply.define("preemphasis", "高频预加重", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.6, source="记录前的轮廓补偿（aperture correction）",
               help="记录时提升高频以抗噪，播放时本应还原——做不到完美就会在硬边上留下过冲。")
    g = "分离与输出"
    ply.define("luma_bandwidth", "亮度带宽", 3.0, 0.1, 6.0, unit="MHz", group=g,
               phys_min=2.0, phys_max=5.0)
    ply.define("chroma_bandwidth", "色度带宽", 0.75, 0.25, 2.0, unit="MHz", group=g,
               phys_min=0.4, phys_max=1.2)
    ply.define("comb_depth", "梳状滤波深度", 0.55, 0.0, 1.0, unit="0-1", group=g)
    ply.define("cross_color", "交叉色彩", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="亮度串入色度（点爬行）")
    ply.define("cross_luma", "交叉亮度", 0.30, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="色度串入亮度")
    ply.define("color_noise", "色度噪声", 0.25, 0.0, 1.0, unit="0-1", group=g)
    ply.define("burst_phase_error", "色同步相位误差", 5.0, 0.0, 45.0, unit="deg", group=g,
               phys_max=20.0)
    ply.define("output_gain", "输出增益", 0.0, -1.0, 1.0, unit="0-1", group=g,
               phys_min=-0.3, phys_max=0.3)
    ply.define("output_bias", "黑电平偏移", 0.0, -0.5, 0.5, unit="0-1", group=g,
               phys_min=-0.15, phys_max=0.15, source="直流恢复/黑电平设定")

    g = "本机输出（无信号时）"
    ply.define("blue_screen", "蓝屏/OSD 替换", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_max=1.0,
               source="JVC SR-S388E 实测：信号过差时丢弃磁带画面并输出自身蓝底白字",
               help="这不是「画面变差」，而是**机器放弃显示磁带**并输出自己的图形。"
                    "实测参考素材的蓝底帧饱和度 0.8947（纯蓝场几乎全是色度），"
                    "而本工具生成的蓝屏饱和度 0.8947，两者吻合。"
                    "与任何程度的噪声都不同：噪声仍然是在显示磁带，蓝屏则不是。")
    ply.define("blue_osd", "OSD 内容", 0.0, 0.0, 2.0, unit="0-2", group=g,
               step=1.0, source="JVC SR-S388E 机内字符发生器",
               help="0=NO SIGNAL｜1=SEARCH / CHAPTER｜2=PLAY。"
                    "真实的字符发生器只有固定的若干条信息，不接受任意文字，"
                    "所以这里也用编号选择而不是自由输入。")

    # ------------------------------------------------------------------ E
    dsp = s.layer("display")
    g = "显像管"
    dsp.define("enabled", "显示设备接入", 1.0, 0.0, 1.0, unit="bool", group=g, step=1.0)
    dsp.define("gamma", "伽马", 2.35, 1.5, 3.2, unit="x", group=g,
               phys_min=2.0, phys_max=2.8, source="消费级 CRT 典型 2.2-2.4")
    dsp.define("contrast", "对比度", 0.55, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("brightness", "亮度", 0.5, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("saturation", "饱和度", 0.5, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("scanlines", "扫描线", 0.35, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("interlace_flicker", "隔行闪烁", 0.25, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("geometry_warp", "几何形变", 0.15, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("overscan", "过扫描裁切", 0.0, 0.0, 0.15, unit="比例", group=g,
               phys_min=0.0, phys_max=0.10, source="家用电视典型 5%-10% 过扫描",
               help="画面被放大后越出显像管边框的比例。真实的过扫描会把画面底部那条"
                    "磁头切换带推出可见区（每边 4 行以上就明显削弱，12 行以上基本看不见）"
                    "——而这条带正是 VHS 最有辨识度的痕迹。默认 0 以便看得见它。")
    dsp.define("vignette", "暗角", 0.25, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("bloom", "泛光", 0.35, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("defocus", "散焦", 0.20, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("reflection", "屏幕反射", 0.10, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("color_temp", "色温", 0.5, 0.0, 1.0, unit="0-1", group=g,
               source="冷/暖白平衡")

    g = "CRT 接口受潮"
    dsp.define("crt_moisture", "显像管受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5, source="尾部接口与视放电路受潮",
               help="CRT 的信号通路极短且全模拟：接头 → 视放 → 阴极，偏转由轭驱动。"
                    "受潮影响的是**视放电路**，不是到达它的信号，"
                    "所以产生的故障和磁带损伤完全不同。")
    dsp.define("crt_corrosion", "CRT 接口氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="接触电阻升高并向地漏电，改变视放偏置，**黑电平整体移动**——"
                    "画面变亮或变暗，而不是变脏。所以受潮的机器看起来是**发白**，"
                    "不是发噪。")
    dsp.define("crt_age", "显像管老化", 0.3, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="三个阴极各有独立视放、各自老化，所以旧机器受潮时出现**偏向某色**"
                    "而不只是变暗。同一个潮湿房间里，新机器看起来发白、旧机器发绿，"
                    "差别就在这里——这是**显示器**的属性，不是录制里的偏色。")

    return s


SCHEMA = build_schema()


# ---------------------------------------------------------------------------
# device profile libraries
#
# A profile is a named set of layer values describing a real device or medium, kept
# separate from the factory presets so the two can be mixed: a preset is a whole
# scenario, a profile is one part of the chain.
#
# Values marked in `source` as estimates are exactly that -- derived from reference
# stills and the failure descriptions supplied with them, not from a bench
# measurement.  Marking them keeps the difference visible.
# ---------------------------------------------------------------------------

MEDIA_PROFILES = {
    "vhs_sp": {"label": "VHS SP 标准", "note": "磁迹最宽、重放输出最高的一档", "layers": {
        "tape": {"tape_speed": 0.0}}},
    "vhs_lp": {"label": "VHS LP 长时", "note": "走带 1/2，重放输出约 -6 dB", "layers": {
        "tape": {"tape_speed": 1.0}}},
    "vhs_ep": {"label": "VHS EP 超长", "note": "走带 1/3，重放输出约 -10 dB", "layers": {
        "tape": {"tape_speed": 2.0}}},
    "vhs_worn": {"label": "磨损 VHS 带", "note": "多次走带后的磁介质", "layers": {
        "tape": {"wear": 0.55, "dropout_rate": 0.9, "noise_floor": 0.32,
                 "crease_rate": 0.35, "chroma_loss": 0.22}}},
    "svhs": {"label": "S-VHS", "note": "亮度带宽约 5 MHz，色度仍欠载", "layers": {
        "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 0.75}}},
    "hi8": {"label": "Hi8", "note": "模拟行记录，无宏块结构", "layers": {
        "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 0.75},
        "tape": {"shuttle_mode": 1.0}}},
    "video8": {"label": "Video8", "note": "8mm 家用机，带宽低于 Hi8", "layers": {
        "format": {"luma_bandwidth": 3.0}, "tape": {"shuttle_mode": 1.0}}},
    "minidv": {"label": "MiniDV", "note": "数字分支：4:1:1 + DCT", "layers": {
        "codec": {"dv_enabled": 1.0, "dv_quality": 1.0},
        "tape": {"shuttle_mode": 2.0}}},
    "mouldy": {"label": "发霉脱粉带", "note": "黏合剂吸水、磁粉层脱落", "layers": {
        "format": {"chroma_gain": 0.12},
        "tape": {"shed_grain": 0.75, "luma_pedestal": -0.16, "binder_loss": 0.35,
                 "dropout_rate": 2.2, "surface_damage": 0.0}}},
    "shed_coating": {"label": "严重脱粉", "note": "涂层大面积缺失", "layers": {
        "format": {"chroma_gain": 0.10},
        "tape": {"binder_loss": 0.7, "shed_grain": 0.85, "luma_pedestal": -0.18,
                 "noise_floor": 0.6}}},
}

PLAYER_PROFILES = {
    "deck_consumer": {"label": "消费级录像机", "note": "无 TBC，走带残差明显", "layers": {
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.45,
                   "frequency_response": 0.15, "comb_depth": 0.55}}},
    "deck_with_tbc": {"label": "带 TBC 的专业机", "note": "时基校正接上，抖动大幅下降", "layers": {
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.9, "tbc_residual": 0.12,
                   "transport_jitter": 0.15}}},
    "deck_worn_heads": {"label": "磁头磨损的录像机", "note": "高频损失 + 色度信噪比下降", "layers": {
        "player": {"head_wear": 0.65, "frequency_response": 0.35,
                   "color_noise": 0.45}}},
    "deck_panasonic_m9000": {
        "label": "松下 M9000（磁鼓轴承油变质）",
        "note": "肩扛式 VHS 通病：磁鼓转速不足 → 行微量重复 → 横向颗粒",
        "layers": {"tape": {"drum_speed": 0.97},
                   "player": {"tbc_enabled": 0.0, "transport_jitter": 0.5}}},
    "deck_jvc_srs388e": {"label": "JVC SR-S388E（S-VHS 编辑机）",
                         "note": "信号过差时会丢弃磁带画面并输出自身蓝底 + OSD",
                         "layers": {"player": {"tbc_enabled": 1.0, "tbc_strength": 0.95}}},
}

DISPLAY_PROFILES = {
    "crt_consumer": {"label": "家用 CRT 电视", "layers": {"display": {
        "gamma": 2.35, "scanlines": 0.35, "bloom": 0.35, "overscan": 0.0,
        "vignette": 0.25, "interlace_flicker": 0.25, "geometry_warp": 0.15}}},
    "crt_trinitron": {"label": "特丽珑（较锐）", "layers": {"display": {
        "gamma": 2.2, "scanlines": 0.22, "bloom": 0.22, "overscan": 0.0,
        "vignette": 0.18, "interlace_flicker": 0.18, "geometry_warp": 0.08}}},
    "capture_card": {"label": "采集卡直入", "note": "不经显像管，几何与扫描线不参与",
                     "layers": {"display": {"enabled": 0.0}}},
}

CAMERA_PROFILES = {
    "ccd_3ccd_consumer": {"label": "3CCD 民用机", "note": "色度信噪比更好、无彩色摩尔纹",
                          "layers": {"camera": {"sensor_type": 1.0, "chroma_aberration": 0.02,
                                                "sensor_noise": 0.05}}},
    "ccd_single": {"label": "单 CCD（典型）", "layers": {"camera": {
        "sensor_type": 0.0, "sensor_noise": 0.06, "chroma_aberration": 0.05}}},
    "cmos_early": {"label": "早期 CMOS", "note": "固定图案噪声更重", "layers": {
        "camera": {"sensor_type": 2.0, "fixed_pattern": 0.12, "sensor_noise": 0.10}}},
    "lens_worn": {"label": "镜头磨损/发霉", "layers": {"camera": {
        "lens_wear": 0.55, "halation": 0.18, "lens_flare": 0.22}}},
}

PROFILE_LIBRARY = {
    "media": MEDIA_PROFILES,
    "player": PLAYER_PROFILES,
    "display": DISPLAY_PROFILES,
    "camera": CAMERA_PROFILES,
}

# NOTE on `rf.broadcast_noise` absolute level:
#   Three calibration attempts against the 1992 off-air reference all compared the
#   model against something it is not (a bright pattern against a near-black frame,
#   twice, then a dark pattern with different content).  The reference is also the
#   *sum* of the RF link, a VHS recording and a capture, and its original scene
#   detail is unknown, so its absolute noise level is underdetermined by that
#   material.  The mechanism's direction and character are verified (see
#   tools/verify.py); its absolute level is a modelling choice, not a measurement.
# The reference camcorder this project was largely calibrated against.
SONY_DCR_TRV310E = {"label": "Sony DCR-TRV310E", "layers": {
    "camera": {"sensor_type": 0.0, "sensor_noise": 0.07, "sharpness": 0.85,
               "chroma_aberration": 0.05, "dynamic_range": 7.5},
    "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 0.75, "edge_boost": 0.30},
    "codec": {"dv_enabled": 1.0, "dv_quality": 1.0},
    "player": {"tbc_enabled": 0.0, "head_wear": 0.25, "transport_jitter": 0.35},
    "display": {"enabled": 0.0},
}}

CAMCORDER_PRESETS = {
    "sony_dcr_trv310e": SONY_DCR_TRV310E,
    "panasonic_nv_gs78": {"label": "Panasonic NV-GS78", "layers": {
        "camera": {"sensor_type": 0.0, "sensor_noise": 0.08, "sharpness": 0.80,
                   "dynamic_range": 7.0},
        "format": {"luma_bandwidth": 4.0, "chroma_bandwidth": 0.7},
        "codec": {"dv_enabled": 1.0},
        "display": {"enabled": 0.0}}},
    "panasonic_m9000": {"label": "Panasonic M9000（肩扛 VHS）", "layers": PLAYER_PROFILES[
        "deck_panasonic_m9000"]["layers"] | {"display": {"enabled": 0.0}}},
}

FACTORY_PRESETS = {
    "family_archive_1992": {
        "label": "家庭录像（1992 录像机 → VHS 存档）",
        "note": "摄像机 + 格式 + 介质 + 播放 + 显示 五层齐备",
        "layers": {
            "camera": {"sensor_noise": 0.08, "noise_grain": 0.0, "dynamic_range": 8.0,
                       "sharpness": 0.80, "motion_blur": 0.18, "sensor_type": 0.0},
            "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.7,
                       "chroma_vertical_averaging": 0.9, "comb_depth": 0.45,
                       "cross_color": 0.50, "cross_luma": 0.35, "color_noise": 0.30,
                       "edge_boost": 0.25, "chroma_gain": 1.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.55, "wear": 0.25,
                     "dropout_rate": 0.35, "noise_floor": 0.18, "head_switching": 0.85,
                     "head_switch_lines": 16.0, "tracking_error": 0.45},
            "player": {"transport_jitter": 0.45, "tbc_enabled": 0.0, "head_wear": 0.30,
                       "frequency_response": 0.15},
            "display": {"gamma": 2.35, "scanlines": 0.35, "bloom": 0.35, "overscan": 0.0,
                        "vignette": 0.25, "interlace_flicker": 0.25, "geometry_warp": 0.15},
        }},
    "rental_tape_worn": {
        "label": "旧出租带（重度磨损 + 无 TBC）",
        "note": "多次走带、划伤、失落密集，且播放机无时基校正",
        "layers": {
            "camera": {"sensor_noise": 0.09, "sharpness": 0.82, "motion_blur": 0.20},
            "format": {"luma_bandwidth": 2.8, "chroma_bandwidth": 0.6,
                       "cross_color": 0.60, "cross_luma": 0.42, "color_noise": 0.38},
            "tape": {"tape_speed": 1.0, "tape_stability": 0.35, "wear": 0.65,
                     "crease_rate": 0.45, "dropout_rate": 1.4, "noise_floor": 0.34,
                     "surface_damage": 0.35, "chroma_loss": 0.18,
                     "head_switching": 0.95, "tracking_error": 0.6},
            "player": {"transport_jitter": 0.7, "tbc_enabled": 0.0, "head_wear": 0.55,
                       "frequency_response": 0.30},
            "display": {"gamma": 2.35, "scanlines": 0.4, "bloom": 0.32, "overscan": 0.0,
                        "vignette": 0.30, "interlace_flicker": 0.30},
        }},
    "pro_transfer_svhs": {
        "label": "专业转录（S-VHS + TBC）",
        "note": "编辑机 + 时基校正，尽量还原母带",
        "layers": {
            "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 0.85,
                       "cross_color": 0.28, "cross_luma": 0.18, "color_noise": 0.16},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.8, "wear": 0.12,
                     "dropout_rate": 0.15, "noise_floor": 0.10, "head_switching": 0.5},
            "player": {"transport_jitter": 0.15, "tbc_enabled": 1.0, "tbc_strength": 0.9,
                       "head_wear": 0.10, "frequency_response": 0.05},
            "display": {"enabled": 0.0},
        }},
    "mouldy_archive": {
        "label": "发霉存档带",
        "note": "潮湿存放：黏合剂吸水、磁粉层脱落。色度先死、亮度塌陷、高频颗粒",
        "layers": {
            "format": {"chroma_gain": 0.12},
            "tape": {"binder_loss": 0.35, "shed_grain": 0.75, "luma_pedestal": -0.16,
                     "dropout_rate": 1.8, "noise_floor": 0.45},
            "display": {"enabled": 0.0},
        }},
    "svhs_disc_rot": {
        "label": "S-VHS 烂碟转录",
        "note": "磁迹严重劣化：亮度基底塌陷、色度污染极重（参考素材实测 sat 0.94 / mean 0.09）",
        "layers": {
            "format": {"luma_bandwidth": 4.6, "chroma_gain": 1.0},
            "tape": {"tape_stability": 0.4, "wear": 0.7, "luma_pedestal": -0.30,
                     "dropout_rate": 1.6, "noise_floor": 0.35, "tracking_error": 0.6},
            "player": {"transport_jitter": 0.6, "head_wear": 0.55},
            "display": {"enabled": 0.0},
        }},
    "broadcast_1992_weak": {
        "label": "1992 电视台弱信号录像",
        "note": "广播链路信噪比不足：宽带噪声把亮度与色度一起淹没（与磁带劣化的"
                "「色度先死」相反）。绝对噪声电平是建模选择而非实测，见 profiles.py 中的说明",
        "layers": {
            "rf": {"signal_dbm": -74.0, "adjacent_channel": 0.20,
                   "broadcast_noise": 0.45},
            "tape": {"noise_floor": 0.30, "dropout_rate": 0.6},
            "display": {"enabled": 0.0},
        }},
    "vcd_352x240": {
        "label": "Video CD（MPEG-1 352x240）",
        "note": "VCD 编码：4:2:0 色度、粗量化、运动补偿预测。"
                "伪影跟着**预测成败**走，不是逐帧独立的噪声——"
                "这是它区别于 DV 与模拟通路的地方。",
        "layers": {
            "camera": {"sensor_noise": 0.03, "sharpness": 0.9},
            "format": {"luma_bandwidth": 3.1, "chroma_bandwidth": 0.6,
                       "chroma_vertical_averaging": 0.5, "cross_color": 0.2,
                       "cross_luma": 0.12, "color_noise": 0.12},
            "codec": {"codec_kind": 2.0, "vcd_quality": 1.0,
                      "vcd_motion_comp": 1.0, "vcd_bitrate_kbps": 1150.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.85, "wear": 0.1,
                     "dropout_rate": 0.05, "noise_floor": 0.06,
                     "head_switching": 0.0},
            "player": {"tbc_enabled": 1.0, "transport_jitter": 0.1,
                       "head_wear": 0.1},
            "display": {"enabled": 0.0},
        }},
    "curving_stripes": {
        "label": "卷曲条纹（差频干涉）",
        "note": "沿画面**弯曲**的条纹，由残余载波与视频通路的差频产生。"
                "与「整行时基误差」是不同现象：位移只把内容横向搬走，"
                "行内曲率保持在地板（0.00001）；这个让行内曲率升到 240 倍。"
                "条纹随读取过程淡入淡出与迁移，不是贴在画面上的固定纹理。",
        "layers": {
            "tape": {"stripe_moire": 0.55, "stripe_beat": 0.05,
                     "stripe_curve": 0.5, "row_timebase": 0.35,
                     "tape_stability": 0.5, "wear": 0.4,
                     "dropout_rate": 0.4, "noise_floor": 0.20},
            "player": {"transport_jitter": 0.5, "tbc_enabled": 0.0,
                       "head_wear": 0.4},
            "display": {"enabled": 0.0},
        }},
    "worn_transport_row_timebase": {
        "label": "走带时基故障（整行位移）",
        "note": "沿画面平滑变化的逐行水平位移：竖直边缘会**弯曲**而不是碎裂，"
                "而且位移形态逐帧迁移。与「行抖动」区分——后者在同一根轴上振荡、"
                "形态每帧重复。亮度与色度同移，不产生凭空偏色。",
        "layers": {
            "tape": {"row_timebase": 0.12, "line_jitter": 0.25,
                     "tape_stability": 0.5, "wear": 0.35,
                     "dropout_rate": 0.4, "noise_floor": 0.20},
            "player": {"transport_jitter": 0.5, "tbc_enabled": 0.0},
            "display": {"enabled": 0.0},
        }},
    "worn_deck_row_chroma": {
        "label": "老化色度解码（整行错色）",
        "note": "整条扫描线解出错误的颜色：色相被旋转、色度增益偏移，"
                "线仍然是饱和的。与「色度丢失行」不同——那个得到灰线，这个得到错色。"
                "误差随读取过程演化，不是逐帧重掷。",
        "layers": {
            "tape": {"row_chroma_error": 0.16, "row_chroma_phase": 42.0,
                     "row_chroma_gain": 0.22, "chroma_loss": 0.03,
                     "tape_stability": 0.45, "wear": 0.45,
                     "dropout_rate": 0.5, "noise_floor": 0.22},
            "format": {"chroma_gain": 1.0},
            "player": {"transport_jitter": 0.55, "tbc_enabled": 0.0,
                       "head_wear": 0.45},
            "display": {"enabled": 0.0},
        }},
    "minidv_clean": {
        "label": "MiniDV 干净",
        "layers": {"codec": {"dv_enabled": 1.0, "dv_quality": 1.0}, "display": {"enabled": 0.0}}},
    "minidv_shuttle": {
        "label": "MiniDV 快进/倒带搜索",
        "note": "解码器用缓冲区旧帧补齐——成片、带锯齿边缘的方块残留",
        "layers": {"codec": {"dv_enabled": 1.0},
                   "tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                            "shuttle_cluster": 0.55, "shuttle_carry": 0.45,
                            "shuttle_mode": 2.0}}},
    "hi8_shuttle": {
        "label": "Hi8 快进/倒带搜索",
        "note": "模拟行记录：满宽横向条带、垂直边缘参差",
        "layers": {"tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                            "shuttle_span": 8.0, "shuttle_wave": 14.0,
                            "shuttle_mode": 1.0}}},
}


def resolve_preset(layers: dict) -> dict:
    """Expand a preset through the schema, filling unspecified values."""
    return SCHEMA.resolve_all(layers)


# ---------------------------------------------------------------------------
# compatibility aliases
#
# The tooling refers to these profiles by their earlier names.  Aliases keep both
# spellings working; each one points at the canonical entry above.  Two names for
# one thing is a cost, so the mapping is explicit and kept in one place.
# ---------------------------------------------------------------------------
MEDIA_PROFILES["camcorder_hi8"] = MEDIA_PROFILES["hi8"]
MEDIA_PROFILES["clean_passthrough"] = {
    "label": "理想直通（无介质损伤）",
    "note": "用来取得「链路什么都没做」的对照",
    "layers": {"tape": {"wear": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0,
                        "head_switching": 0.0, "crease_rate": 0.0,
                        "line_jitter": 0.0, "walk_amplitude": 0.0,
                        "micro_jitter": 0.0, "vertical_instability": 0.0},
               "player": {"transport_jitter": 0.0, "head_wear": 0.0,
                          "frequency_response": 0.0, "tbc_enabled": 1.0},
               "display": {"enabled": 0.0}},
}
PLAYER_PROFILES["vcr_consumer"] = PLAYER_PROFILES["deck_consumer"]
DISPLAY_PROFILES["none"] = {"label": "不经显示设备",
                            "note": "采集卡直入：几何、扫描线、泛光都不参与",
                            "layers": {"display": {"enabled": 0.0}}}
CAMERA_PROFILES["ccd_camcorder_1990"] = CAMERA_PROFILES["ccd_single"]
CAMERA_PROFILES["clean_digital"] = {
    "label": "理想数字采集（无相机劣化）",
    "note": "用作对照：相机层全部中和",
    "layers": {"camera": {"sensor_noise": 0.0, "noise_grain": 0.0,
                          "fixed_pattern": 0.0, "chroma_aberration": 0.0,
                          "motion_blur": 0.0, "sharpness": 1.0, "exposure": 0.0,
                          "white_balance_temp": 0.0, "ccd_smear": 0.0,
                          "halation": 0.0, "ccd_gain": 0.0, "color_cast": 0.0,
                          "highlight_tint": 0.0, "lens_wear": 0.0, "lens_flare": 0.0},
               "format": {"edge_boost": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                          "color_noise": 0.0, "chroma_vertical_averaging": 0.0,
                          "comb_depth": 0.0, "burst_phase_error": 0.0,
                          "luma_bandwidth": 5.0, "chroma_bandwidth": 1.5,
                          "filter_kind": 0.0},
               "display": {"enabled": 0.0}},
}
CAMCORDER_PRESETS["sony_trv310e"] = CAMCORDER_PRESETS["sony_dcr_trv310e"]
FACTORY_PRESETS["pro_transfer"] = FACTORY_PRESETS["pro_transfer_svhs"]
FACTORY_PRESETS["rental_tape_horror"] = FACTORY_PRESETS["rental_tape_worn"]

# resolve_preset is defined above; expose it from the package's public names
__all__ = [n for n in dir() if not n.startswith("_")]
