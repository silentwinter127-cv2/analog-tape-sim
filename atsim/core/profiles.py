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
    "overlay": "06 字幕机",
}


# **The night-vision functions by name.**  `nv_mode` is a selector, not a slider -- a number between 0 and 3 gives the
# user no way to know which function they are choosing -- so the schema carries the four names and the interface
# presents them.  The mechanism still reads the number, so a settings file and the API are unchanged.
# **The four functions, named here so the label can carry them.**  `ParamSpec` has no `choices` field,
# so the selector's names live in the label and the help text rather than in the parameter machinery.


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
    # --- automatic control loops ---------------------------------------------
    # A handycam has no operator setting exposure or white balance; it has loops that measure the
    # picture, move a control, and take time doing it.  That is where a consumer camcorder's
    # character comes from, and a static offset cannot express it: a static cast cannot drift.
    #
    # All three are off by default, so a camera with no automatic control renders exactly as before.
    g = "自动控制"
    cam.define("ae_enabled", "自动曝光 (AE)", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="**这是消费级摄像机味道的主要来源。** 手持机没有人在调曝光，"
                    "它有一个环：测画面亮度、动一个控制、而且要花时间。"
                    "于是「突然进暗处」的表现是**慢慢变亮**，而不是瞬间变亮。"
                    "静止的偏移做不出这件事——静止的偏移不会漂移。")
    cam.define("ae_target", "AE 目标亮度", 0.46, 0.2, 0.7, unit="0-1", group=g,
               phys_min=0.35, phys_max=0.55,
               help="环要维持的画面平均亮度（中央加权，因为真实测光不是均匀平均）。"
                    "偏高会过曝，偏低会欠曝——这正是不同机器" +
                    "对同一场景给出不同曝光的原因。")
    cam.define("ae_tau_s", "AE 时间常数", 1.1, 0.05, 6.0, unit="s", group=g,
               phys_min=0.3, phys_max=3.0,
               help="环响应的时间常数。数值来自实测的渐进曲线：一个时间常数处走到约 63%。"
                    "**注意这不是「收敛速度」而是「逼近平衡的速率」**——"
                    "环在测量停止变化时停在目标上，而不是永远差 1/e。")
    cam.define("ae_limit_stops", "AE 调节范围", 2.0, 0.5, 4.0, unit="stop", group=g,
               phys_min=1.0, phys_max=3.0,
               help="环能推多远。真实 AE 有范围限制，超出就放弃——"
                    "这就是为什么极暗场景下画面仍然很暗，而不是被硬拉亮。")
    cam.define("awb_enabled", "自动白平衡 (AWB)", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               help="**色彩漂移的来源。** 环把画面的平均色度归零，所以灯泡下发黄、"
                    "一两秒后变中性，镜头转向窗户又摆回去。"
                    "静态色偏做不出这个——而这正是 90 年代家庭录像最典型的样子。")
    cam.define("awb_tau_s", "AWB 时间常数", 1.6, 0.05, 8.0, unit="s", group=g,
               phys_min=0.5, phys_max=4.0)
    cam.define("awb_limit", "AWB 调节范围", 0.42, 0.05, 0.8, unit="0-1", group=g,
               phys_min=0.2, phys_max=0.6,
               help="每通道的校正范围。真实的 AWB 有上限，超出就**放弃并保留色偏**——"
                    "这就是为什么在强色光下拍的素材仍然是那个颜色。")
    cam.define("af_enabled", "自动对焦 (AF)", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="对比度检测 AF：**它测不了焦点，只能测对比度**，所以它迈一步、看对比度有没有变好、"
                    "变差就反向。低对比度场景下它**找不到峰，会一直找**——"
                    "「清晰 → 模糊 → 寻找焦点 → 清晰」就是从这里来的。")
    cam.define("af_tau_s", "AF 步进间隔", 0.9, 0.05, 4.0, unit="s", group=g,
               phys_min=0.2, phys_max=2.0,
               help="每次试探之间的间隔。真实 AF 不会逐帧狂搜，它按一个节奏试。")
    cam.define("autopresettle_s", "开机已适应时长", 0.0, 0.0, 60.0, unit="s", group=g,
               phys_min=0.0, phys_max=30.0,
               help="**这段录像开始之前，相机已经适应这个场景多久了。**"
                    "0 = 录像从开机开始，此时相机最不稳定；"
                    "设成 10 秒则三个环都已基本稳定，画面上看不出开场摆动。"
                    "它按解析衰减计算，所以「已适应」不是代码里的特例，只是剩余暂态更短。")
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
    cam.define("axis_offset", "光轴偏心", 0.0, 0.0, 0.6, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.3,
               help="**真实镜头的装配不会正好对中传感器**，所以光轴会偏离画面中心。"
                    "**所有径向像差都是相对光轴、而不是相对画面中心度量的**——"
                    "所以偏心会把「像差为零的那一点」挪走：**画面正中的高光就获得了半径，不再是零。**"
                    "这就是为什么正中的灯在完美对中的镜头下不该有色边，而真机上能看到。")
    cam.define("coma", "彗差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**轴外的一点会成彗星状**：通过镜头外缘的光线放大率与通过中心的不同，"
                    "所以它不是点，而是**亮核加一条拖尾**。拖尾是**单向的、朝光轴外的**，并随视场半径增大。"
                    "这就是为什么街灯落在画面角落时像一颗抹开的泪滴。")
    cam.define("astigmatism", "像散", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**轴外光线在切向平面和径向平面上的焦距不同**，一点被成成一条短线，"
                    "而且**两个焦面上短线的方向是翻转的**。"
                    "签名是**方向性**：模糊沿某一个轴更强，而像场弯曲和球差都是圆的。")
    cam.define("spherical_aberration", "球差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**通过镜头外缘的光线聚焦更近**，所以焦点是软的、而且**光晕与亮度有关**："
                    "亮点长出大而圆的晕，暗点仍然紧实。"
                    "**这就是大光圈全开软、收小光圈变锐的原因**，也是它和像场弯曲（随半径）的区别。")
    cam.define("lens_distortion", "镜头畸变", 0.0, -1.0, 1.0, unit="-1..1", group=g,
               phys_min=-0.4, phys_max=0.4,
               help="**镜头本身的几何畸变**：简易镜头是桶形（负）、长焦是枕形（正）。"
                    "注意相机端原本**完全没有几何项**——从镜头出来的画面四边是绝对直的。"
                    "它和「05 显示设备」的管面曲面共用径向几何，但**符号相反**：那是显像管端面，这是玻璃。")
    cam.define("field_curvature", "像场弯曲", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**简易镜头把平面景物成在曲面上**，所以中心合焦的那个平面并不是四角合焦的平面。"
                    "结果就是**同一个对焦设定下、中间锐利而四周发软**——这也是为什么当年的机器"
                    "中心永远比角落好看，跟光圈多大无关。"
                    "模糊量是**视场半径**的函数：中心为零、向外增大，因为点扩散函数随视场角的平方增长。")
    cam.define("longitudinal_ca", "纵向色差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**镜头把三个波长聚焦在轴上的不同距离**，所以在感光面上总有一个合焦、另外两个不合焦。"
                    "**和横向色差正好相反：它不产生任何位移**——红边和蓝边在同一位置、只是都更宽，"
                    "所以彩色边出现在高光的**四周**，而不是偏向某一侧。"
                    "这是它和「横向色差」的本质区别：一个是位移，一个是散焦。")
    # --- 操作者行为（手持、颠簸、变焦）----------------------------------------
    # 这一层不是器件属性，而是**拿着它的人**的属性。它是让素材读起来像"某件事的录像"
    # 而不是"一张被加了故障的图"的那一层。
    #
    # **手持抖动不是噪声。** 最直观的实现是每帧一个随机偏移，那是错的：人的抖动是**窄带且有节奏**的，
    # 而这个节奏才是它读起来像手、而不像故障的原因。手是几个准周期分量的叠加：
    #   0.4-1.5 Hz  姿态摇摆（全身缓慢平衡修正）——幅度最大
    #   1.5-3 Hz    走路与呼吸
    #   4-8 Hz      手臂与肩部肌活动
    #   8-12 Hz     生理性震颤（运动系统固有，始终存在）
    # 幅度随频率下降，**震颤约为摇摆的百分之一**，而这个**比例**才是眼睛读到的东西。
    g = "操作者行为"
    cam.define("shake_intensity", "手持抖动", 0.0, 0.0, 4.0, unit="x", group=g,
               phys_min=0.0, phys_max=2.0,
               help="四个窄带分量按实测比例叠加：摇摆 0.9 Hz、走呼吸 2.1 Hz、臂肩 5.5 Hz、"
                    "生理震颤 9.5 Hz，幅度依次为画面的 1.0%/0.42%/0.16%/0.08%。"
                    "**1.0 约合 5.6 px 标准差**（720 宽）。"
                    "关键是**节奏**而不是幅度——所以它是振子而不是滤波过的随机游走，"
                    "频谱集中度实测 0.22（平坦谱为 0.008）。")
    cam.define("bump_rate_hz", "颠簸发生率", 0.0, 0.0, 3.0, unit="次/秒", group=g,
               phys_min=0.0, phys_max=1.0,
               help="被撞到：一个台阶、一扇门、别人碰到手肘。**这些是事件，不是连续运动**，"
                    "所以和失落到粘带一样是事件列表——稀疏的尖峰叠在一个安静的底上"
                    "（实测 80 分位偏差为 0.00 px）。")
    cam.define("bump_amount", "颠簸强度", 0.0, 0.0, 2.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="幅度取自一个**有长尾的分布**：大多数颠簸很小，少数很大。"
                    "形状是**猛撞后指数恢复**（先快后慢），不是阶跃也不是对称脉冲。")
    cam.define("zoom_travel", "变焦行程", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="变焦马达的行程。**消费机的变焦不是线性的**：小直流电机带齿轮组，"
                    "先要克服静摩擦（慢）、中段扫过去（快）、松手时**直接断电停住**——"
                    "末尾是停住而不是减速，因为机构摩擦力足够保持位置。")
    cam.define("zoom_kind", "变焦类型", 1.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0=线性；1=**消费机马达**（克服静摩擦 → 扫过 → 断电停住）；"
                    "2=专业伺服（两端都平滑）。中段斜率 0.91、末端 1.70——"
                    "断电的马达**仍然有速度**，这正是它和缓入缓出的区别。")
    cam.define("motion_blur", "运动模糊", 0.15, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.6, source="1/60s 快门")
    cam.define("ccd_smear", "CCD 垂直拖尾", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="隔行 CCD 读出期间的电荷泄漏",
               help="强光在垂直方向拉出的长条拖影。消费级 CCD 在夜间拍摄灯光时非常明显。")
    # --- 光学与传感器（笔记第 5 节）--------------------------------------------
    # 这三个是**属于「这一台相机」而不是属于介质或链路**的故障，也是让素材能被认出
    # 来自某一台机器的东西。
    g = "光学与传感器"

    cam.define("lens_fog", "镜头起雾", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="**结在前镜片上的水汽**——和它容易混淆的三种故障都不同："
                    "灰尘是**挡住**光、边缘锐利；镀膜磨损会散射但留下有结构的雾翳；"
                    "散焦是**整幅图像被卷积**、边缘对称展宽。"
                    "起雾是**高光泽扩散**：水膜表面自身就是个透镜，"
                    "所以亮部向暗部**平滑而无边界**地扩散，**黑位先抬且抬得最多**，"
                    "同时高光长出一圈很宽很软的裙边。"
                    "从冷车抱进暖房的手持机起雾，是最典型也最少被建模的故障之一，"
                    "而且它会在几分钟里随玻璃升温而恢复——所以它同时属于环境维度。")

    cam.define("bloom_amount", "电荷溢出", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.6,
               help="**饱和像元的电荷沿垂直转移通道溢出**，所以亮源产生的是**垂直条纹，不是圆形光晕**。"
                    "这正是它和 halation 的区别：圆形光晕是光学效应（镜头或盖板玻璃散射），"
                    "垂直条纹是电荷输运效应。CCD 手持机对着太阳拍会出现这条竖纹，"
                    "也就是笔记把它单列的原因。")
    cam.define("bloom_threshold", "溢出阈值", 0.82, 0.5, 1.0, unit="0-1", group=g,
               phys_min=0.7, phys_max=0.95,
               help="像元到多少亮度才算溢出。这是阱容对应的电平——真实 CCD 大约在 80% 左右饱和。")
    cam.define("dead_sites", "坏点数量", 0.0, 0.0, 8.0, unit="个", group=g, step=1.0,
               phys_min=0.0, phys_max=4.0,
               help="**这一台相机传感器上的失效像元数量。**"
                    "CCD 是几十万个独立结构组成的大阵列，其中一些会失效；"
                    "失效分两类且看起来不同：**死点**每帧都是黑的，**亮点（卡住）**每帧都是满值。"
                    "两者位置**永远固定**——这正是要点：会动的缺陷是噪声，不动的缺陷是**那台相机的记忆**。"
                    "真实机器上几个坏点是正常的，超过几个属于不良品。"
                    "**数量按传感器面积缩放**，所以预览和全分辨率看到的是同一台传感器。")
    cam.define("stuck_fraction", "亮点占比", 0.45, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.2, phys_max=0.7,
               help="失效像元里「卡在满值」而非「读出全黑」的比例。两者不等概率："
                    "短路读出为黑，复位失效为满。")
    cam.define("halation", "高光光晕/泛光", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.9, source="CCD 过饱和溢出 + 镜头内反射",
               help="过曝区域的扩展光晕，会带出紫红色边缘。与显示阶段的辉光不同：它发生在传感器上。")
    cam.define("halation_source_colour", "光晕跟随光源色", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="**散射光带的是光源自己的颜色**，所以蓝色细线周围的光晕应该是蓝白的。"
                    "0 = 固定的镜片镀膜色调（原行为），1 = 完全跟随炸白区的自身颜色。"
                    "实测：蓝色细线在固定色调的光晕下回到 B−R +0.000（自身 +0.024），"
                    "而且没有任何参数值能恢复——因为提高饱和度只能放大已存在的差，"
                    "无法恢复被中性光除掉的差。")
    cam.define("shadow_depth", "暗部与中间调压深", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="**只压暗中间调和暗部，亮部增益恰好为 1**（gain(L) = 1 − a·(1−L)²）。"
                    "用途：提高亮细线与其周围的对比，**而不给线本身加任何光**——"
                    "因为加光会同时抬高线和周围、从而减小它们的差，"
                    "也就稀释了线的颜色（这正是灯杆蓝色丢失的原因）。"
                    "实测：原来靠调低泛光来提对比，结果对比从 0.2185 掉到 0.2069，方向相反。")
    # --- the night-vision modes ------------------------------------------------
    cam.define("nv_mode", "夜视模式 0关/1NightShot/2Super/3ColorSlow", 0.00, 0.0, 3.0, unit="", group=g,
               phys_min=0.0, phys_max=3.0, source="索尼 Digital 8 / DSC-F717 的三种夜视功能",
               help="**从物理底层重建的索尼夜视功能。**"
                "**默认 0（关闭）：模式是被选择的功能，不是一个一直在运行的机制。**"
                "0 = 关；1 = NightShot（拆掉红外截止滤镜 + 开红外照明，显示为绿单色）；"
                "2 = Super NightShot（更长曝光 + 帧积累，所以【更亮且更少噪】）；"
                "3 = Color Slow Shutter（保留彩色、只放慢快门）。"
                "对照素材 `索尼F717夜视测试` 的分段实测：绿段 G 占比 0.44-0.58、色相散度 0.06-0.18；"
                "彩色段 G 占比 0.28-0.30、色相散度 0.24-0.37。")
    cam.define("nv_illuminator", "红外照明强度", 1.20, 0.0, 4.0, unit="0-4", group=g,
               phys_min=0.0, phys_max=0.6,
               help="**平方反比、且在锥形光束内**——所以近场被照亮、走廊远处归黑。"
                    "这是手持 NightShot 画面最好认的特征，而它来自灯而不是场景。"
                    "实测：`illuminator=1.0` 时场从近场 7.41 降到地平线 0.11，58/58 步单调下降。")
    cam.define("nv_beam_x", "照明光轴 X", 0.5, 0.0, 1.0, unit="x", group=g,
               phys_min=0.0, phys_max=1.0, help="红外灯瞄准的位置（画面横向）。")
    cam.define("nv_beam_y", "照明光轴 Y", 0.75, 0.0, 1.0, unit="y", group=g,
               phys_min=0.0, phys_max=1.0, help="红外灯瞄准的位置（画面纵向；机位在画面底部）。")
    cam.define("nv_beam_angle", "光束角", 0.55, 0.1, 2.0, unit="rad", group=g,
               phys_min=0.15, phys_max=1.5, help="红外灯的锥形光束宽度。")
    cam.define("nv_ir_albedo", "近红外反射占比", 0.70, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               help="**NightShot 的物理核心：拆掉红外截止滤镜后，画面由【近红外反射率】主导，**"
                    "而近红外反射率与可见光颜色【无关】——这是为什么皮肤、植被、白布会发亮，"
                    "而深色衣物和沥青变黑，这是任何调色都做不出来的。")
    cam.define("nv_ir_response", "红外响应", 1.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="红外截止滤镜被移除的程度（0 = 滤镜在位）。**这是近红外响应的总闸门**——"
                    "设为 0 时 `ir_highlights` 直接返回原图。\n"
                    "**我一度以为「输入已是夜视输出时应当关掉它」，而实测推翻了这个假设**："
                    "把真实 NightShot 帧喂进链路，关掉它在**每一点上都更差**——\n"
                    "  t=78s  实拍 mean 0.387 R/G 0.367 | ir 1.0 → 0.423/0.645 | ir 0.0 → 0.433/0.665\n"
                    "  t=60s  实拍 mean 0.242 R/G 0.587 | ir 1.0 → 0.265/0.687 | ir 0.0 → 0.282/0.711\n"
                    "**所以 1.0 保留为默认。**\n"
                    "**而这次测量暴露的真实偏差是**：均值一致地高约 **12%**、R/G 一致地偏高——"
                    "输入帧已经是夜视输出（mean 与 R/G 本来是对的），**而链路在它上面又叠了一层色调曲线**。"
                    "那个偏差**不在这一项**，尚未定位。")
    cam.define("nv_ir_focus_shift", "红外焦点偏移 (px)", 1.5, 0.0, 12.0, unit="px", group=g,
               phys_min=0.0, phys_max=6.0,
               help="**note.txt 第六条机制，而它自己说「这是很多人不知道的」。**\n"
                    "镜头是为【可见光】聚焦设计的，而 700–1000nm 折射率不同，**近红外的像面落在可见光像面之后**。\n"
                    "**后果不是画面变糊**——可见光像仍然锐利——**而是【近红外那一层的贡献】变软**："
                    "植被与被灯照亮的表面失去细节结构，可见光边缘依旧清晰。0 = 与可见光共焦。\n"
                    "**第一次实现被实测推翻并回退**：模糊施加在 `ir` 上（spread 归一化【之前】），"
                    "而归一化那一步除以 `lum_new.std()`——**模糊先缩小了分母，因子就变大去补偿**，"
                    "shift 2.0 时把均值从 0.3750 推到 **0.9697**。**归一化恰好把焦点偏移本该去掉的细节又装了回去。**\n"
                    "**现在施加在 spread 归一化【之后】**，所以那个因子已经用过了；"
                    "而电平靠【减去模糊场的局部均值】来守恒（不是重新缩放），**所以均值变化在构造上为 0**，"
                    "改变的只有相邻像素之间的【序】——那正是焦点偏移所做的事。")
    cam.define("nv_nir_material", "近红外材质判定 (1开/0关)", 1.0, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=1.0,
               help="**用视觉模型判定场景材质，再按材质的近红外反射率成像**，而不是把 RGB 转成灰度。"
                    "近红外反射率与可见光亮度【无关**（植被 2.10、皮肤 1.55、深色衣物 0.28——7.5 倍差距），"
                    "所以亮度曲线表达不出这个差异，而这是普通夜视滤镜最难自然模拟的部分。"
                    "**本机没有可用视觉模型时自动退回亮度估计**，那不是降级而是文档化的回退路径。")
    cam.define("nv_nir_grid", "材质判定网格", 8, 2, 32, group=g,
               phys_min=2, phys_max=32, help="把画面切成 N×N 块逐块判定材质；越大边界越细但每块越小。")
    cam.define("nv_nir_depth", "近红外距离衰减 (1开/0关)", 1.0, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=1.0,
               help="**用视觉模型推断相对深度，让红外灯的衰减按【距离】而不是按【半径】。**"
                    "note 的第一条机制是「物体距离增加后反射光减弱——近处物体可能很亮，远处背景却仍然很暗」，"
                    "而纯径向场只能压暗画面四角（实测默认档中心到边缘只差 21%），做不到这件事。"
                    "灯光照度按 1/d² 衰减，所以有逐像素深度就是一除。**" 
                    "深度是【相对值】（模型的零点任意），所以只用于衰减的形状，不代入任何绝对距离。**")
    cam.define("nv_nir_margin_floor", "材质判定置信下限", 0.02, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="块的首选与次选概率差低于此值时判为「不确定」，该块退回亮度估计。"
                    "**实测参考照片的 margin 从 0.692 到 0.030**，所以这不是假想情况："
                    "0.03 的判定不能把植被的 2.10 贴到水泥上。")
    cam.define("nv_cast_variation", "绿投射的空间变化", 0.0, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=0.8,
               help="**实拍的绿不是均匀的，而固定投影必然是均匀的。**\n"
                    "实测模型的色调散度是 **0.1714**，而真实三段是 **0.5519 / 0.1170 / 0.1129**"
                    "——**平均颜色对了，分布不对**，这正是「实拍图片并不是均匀的绿」。\n"
                    "物理上变化的是【有多少红外到达传感器】：红外灯的光随距离衰减，"
                    "所以远离灯的表面近红外回报更弱、颜色更多来自环境可见光——"
                    "**为权重拟合的那个蓝色泄漏应当在远场更弱、近场更强。**\n"
                    "变化依据是视觉模型给出的相对深度（已挂在 spec 上）；"
                    "**亮度不受影响**：这是逐像素缩放权重后再按自身亮度归一化。")
    cam.define("nv_shot_noise", "散粒噪声 (随信号√)", 0.10, 0.0, 0.5, group=g,
               phys_min=0.0, phys_max=0.3,
               help="**随【信号强度】变化的噪声——与读出噪声物理行为相反，两者都必须有。**\n"
                    "note 说：「加入随曝光和信号强度变化的噪声、暗部读出噪声、固定图样噪声，以及高亮区域的饱和。"
                    "**不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同。**」\n"
                    "读出噪声是【固定绝对值】，所以它在暗部【相对】最大；"
                    "散粒噪声按信号平方根增长，所以在绝对值上亮部最大。\n"
                    "**只有其一的模型必然在某一端是错的**，而这一点单个标量测不出来——"
                    "实测把 read_noise 从 0 提到 0.25，暗/亮时间噪声比反而从 0.281 升到 0.448。\n"
                    "逐像素标准差取 shot_noise × √lum：泊松到达的方差正比于均值，所以偏差按平方根。")
    cam.define("nv_shutter_frames", "慢快门积分帧数 (1=不积分)", 1.0, 1.0, 32.0, unit="帧", group=g,
               phys_min=1.0, phys_max=16.0,
               help="**一次曝光跨越几帧——这是「慢快门」而不是「多帧累积」。**"
                    "两者算出来的均值相同，物理意义不同：note 第四条说「根据曝光时间对运动物体进行时间积分，"
                    "而不是只复制几张半透明图层。快门越慢，运动物体在一次曝光期间移动得越远，残影就越明显」。"
                    "**多帧求和是多次独立曝光**（信号相干相加、噪声按平方根相加，画面【更干净】）；"
                    "**慢快门是一次曝光开着**（运动主体被拖出残影，且不会回来）。"
                    "把前者当成后者，就是为什么模型的模式 2 没有拖影而素材的 NIGHTSHOT 2 明显有。")
    cam.define("nv_green_mix", "绿单色显示", 1.00, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="1 = 纯绿单色取景，0 = 保留原色。"
                    "**是【单轴增益】而不是去饱和**——实测素材绿段的通道差大而色相散度小"
                    "（0.06-0.18 对彩色照片的 0.37），这正是单色取景器接到彩色录像机的结果。")
    cam.define("nv_green_gain", "绿通道增益", 1.15, 0.5, 1.6, unit="x", group=g,
               phys_min=0.7, phys_max=1.4,
               help="绿单色显示里绿通道的增益。**默认值 1.15 是从素材反解出来的，不是断言的。**\n"
                    "它决定单色投影后【所有存活内容】的 R:G:B 比值——亮度和残留色度都带这套权重，"
                    "所以它同时是色彩控制而不只是亮度控制。\n"
                    "素材三条 NIGHTSHOT 的逐通道平均绝对色度给出 G/B 1.150 / 3.714 / 3.034、"
                    "R/G 1.962 / 2.011 / 1.917。\n"
                    "**而 1.0 时投影给出 R:G:B [0.505 1.000 0.824]，即 G/B 0.824——蓝高于绿，"
                    "而素材是绿远高于蓝**；模型自己的帧实测 [0.0728 0.0240 0.0955]，蓝是素材的 4.48 倍、绿 1.00 倍。\n"
                    "**`chroma_residue` 无论取何值都修不了**，因为 out = mono + (rgb − lum)·r 两项带同一套权重。\n"
                    "在【亮度增益精确钉在 1.0】的约束下扫描：1.15 给出 G/B 2.667（目标 2.632）、"
                    "R/G 2.027（目标 1.963），误差 0.046，而 1.0 的误差是 0.706。")
    cam.define("nv_exposure_trim", "夜视曝光补偿 (档)", -0.10, -3.0, 3.0, unit="stops", group=g,
               phys_min=-1.0, phys_max=0.0,
               help="**夜视模式自己的曝光补偿，不是全局 `exposure`。**\n"
                    "**在【两张输入照片】上一起扫出来的**（对基准素材 NIGHTSHOT 0：mean 0.2346 / p90 0.5870 / p99 0.9511）：\n"
                    "  trim −0.10 → 旧照片 0.2310/0.5422/0.8930   新照片 0.2463/0.5684/0.8185   合并误差 【0.153 / 0.221】\n"
                    "  trim −0.50 → 旧照片 0.1791/0.3866/0.7072   新照片 0.1910/0.4195/0.6334   合并误差 0.835 / 0.806\n"
                    "**−0.10 对两张都最好，而 −0.50 是我早前的一个错误**：那一轮我拿用户单帧 `c_0091.png` 当目标，"
                    "而它比标准 NIGHTSHOT 0 参考暗约 2.4 倍（mean 0.096 对 0.2346），"
                    "**所以我把模型压暗去追一个不是基准的东西**——本工作记录过的「用错的配置测量」的第 n 次。\n"
                    "**它必须加在【AE 环路之后】**：环路里 `self.exposure = ...update(...)` 会【替换】这个值，"
                    "此前在构造函数里设置会被一行之后丢掉——实测 −0.10/−0.6/−1.0/−1.4/−1.8 输出逐位相同。\n"
                    "**而且它必须在 schema 中声明**：未声明时 `SCHEMA.resolve_all` 会在阶段看到它之前丢掉它"
                    "（实测：传 −1.2 到达阶段的是 −0.1）——**这是本工作第二次踩到同一个机制。**")
    cam.define("saturation_stops", "饱和点 (低于白点几档)", 0.30, 0.0, 3.0, unit="stops", group=g,
               phys_min=0.0, phys_max=2.0,
               help="**传感器的饱和点——它决定画面亮部能否被打爆。**\n"
                    "实测：模型最亮的像素是 **0.8686，全画面没有任何像素被裁切**，"
                    "而素材到 **1.0、0.66% 的像素在 0.98 以上**。\n"
                    "逐调用追踪证明 `_roll_off` **不是**原因——它裁切正确，位置就在它的白点上。"
                    "**原因是没有任何值到达那个白点**：默认 `dynamic_range` 为 8 时白点正好是 1.0，"
                    "而画面经曝光与白平衡后的最大值只有 0.87，所以 `lin >= white` 那一支从不触发，"
                    "肩部坐在它本该压缩的内容【上方】。\n"
                    "0 保持曲线原样；大于 0 则把白点降低相应档位，"
                    "于是本该停在 0.95 的高光会被钉在 1.0 上——**真实传感器就是这样过曝的**。\n"
                    "**传感器的饱和点是传感器自身的属性，不是它动态范围的属性**——"
                    "所以它是一个独立控制，而不是另一个 `dynamic_range`（后者调【低】反而把白点推得更高）。\n"
                    "**注意：新参数必须在 schema 中声明**，否则 `SCHEMA.resolve_all` 会在阶段看到它之前丢掉它"
                    "（实测：`merge_values` 后是 0.6，`resolve_all` 后是 ABSENT）。\n"
                    "**实测限度**：默认 0.30 把 max 从 0.8686 抬到 0.8857、p99 从 0.8061 到 0.8334，"
                    "而 clipped 仍是 0.0000。素材是 max 1.0000、p99 0.9511、clip 0.00657。"
                    "**所以这一项只部分闭合**：真实照片的最亮像素在 0.87 附近，"
                    "而曲线把能量重新分配进肩部而不是让它裁切——需要光源使画面自身有更宽的动态范围"
                    "（例如更强的照明或更亮的高光），才可能真正打爆。")
    cam.define("nv_highlight_saturation", "高光饱和驱动 (只作用于闸门以上)", 4.0, 0.0, 8.0, unit="x", group=g,
               phys_min=1.0, phys_max=3.0,
               help="**取景器的驱动量，只作用在【闸门以上】——这才是高光变白的机制。**\n"
                    "显示把画面构成 `lum × w`，所以某通道在 `lum × w == 1` 处饱和："
                    "归一化后的权重是 [0.6069 1.2021 0.9904]，**绿在 lum 0.832 处饱和**，蓝 1.010，红 1.648。\n"
                    "**闸门以下投影精确成立**，所以中间调保持实测的 R:G:B，为素材拟合的标定完全不动；"
                    "**闸门以上最强通道被钉住而其余继续爬升**，于是该像素**去饱和走向白色**，"
                    "其亮度得以离开比例自身的上限。**这正是 note 六条里唯一没有实现的那条：「高亮区域的饱和」。**\n"
                    "**为什么必须加闸门**：第一版把驱动乘在【每个】像素上——实测 gain 1.0→1.6 时"
                    "均值从 0.2370 升到 0.3493，即整体提亮，而那正是早前 `nv_highlight_saturation` 不可用的原因。"
                    "**在没有东西饱和的地方施加增益，得到的不是高光而是更亮的照片。**\n"
                    "**加闸门后实测**（目标是素材的 mean 0.2346 / p99 0.9511 / max 1.0000 / clip 0.00657）：\n"
                    "  gain 1.00 → mean 0.2370  p99 0.8061  max 0.8686  评分 0.3135\n"
                    "  gain 2.00 → mean 【0.2383】 p99 0.8454  max 【0.9799】 评分 【0.1663】\n"
                    "**均值几乎不动（+0.5%）而 max 升了 0.111——这是第一次高光被抬起而没有整体提亮。**\n"
                    "gain 2.0 以上不再变化，因为闸门以上的像素已全部裁切。")
    cam.define("nv_chroma_residue", "绿显示残留色度", 0.15, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=0.8,
               help="**移除 IR-cut 后色彩并没有完全消失**，透过来的一部分原图色度。\n"
                    "**默认 0.45 是扫出来的，而且这个参数此前是【不可达】的**：三个模式预设各自携带 "
                    "chroma_residue（0.35 / 0.90 / 1.0），而 `mode_spec` 每帧从预设重建 spec，"
                    "所以用户设 0.00 / 0.35 / 0.95 到达项的值【都是 0.350】、输出逐位相同。"
                    "已按 `ir_response` 的同一理由从三个预设中移除——**覆盖物理控制的预设就是删除那个控制**。\n"
                    "**它是三段素材分歧最大的量**：红占比 0.2892 / 0.2339 / 0.2011，相差 1.44 倍。\n"
                    "扫描（目标是素材自身的区间：色调散度 0.113–0.552、饱和度 0.144–0.339）：\n"
                    "  0.15 → 0.1169 / 0.2299     0.35 → 0.1706 / 0.2093\n"
                    "  0.45 → 0.2092 / 0.2013     0.55 → 0.2764 / 0.1951\n"
                    "**0.45 取色调散度的区间中点，且远离 0.65 处的不稳定跳变（那里到 2.73）。**\n"
                    "**亮度精确不变**：0 / 0.5 / 1 下实测 0.2301 / 0.2298 / 0.2301。")
    cam.define("nv_exposure_steps", "夜视曝光补偿", 0.00, 0.0, 3.0, unit="EV", group=g,
               phys_min=0.0, phys_max=2.5,
               help="**模式自己的曝光提升，走相机正常的曝光通路**——"
                    "所以在高光滚降、增益噪声和白平衡上都和普通曝光一样。"
                    "在末端乘一个系数只会让画面变亮，而没有真的被曝光过。")
    cam.define("nv_accumulate", "帧积累", 8.00, 1.0, 16.0, unit="frames", group=g,
               phys_min=1.0, phys_max=16.0,
               help="**Super NightShot 的关键：信号相干相加、噪声按平方根相加，"
                    "所以信噪比按 sqrt(N) 改善。**单次 N 倍曝光则会同时放大暗电流和散粒噪声，"
                    "两者物理上不同。实测：8 帧使高频能量降到 1/2.878（理论 sqrt(8)=2.83）。")
    cam.define("nv_read_noise", "夜视读出噪声", 0.0, 0.0, 0.2, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.1, help="夜视增益档带来的读出噪声。")
    cam.define("nv_chroma_noise", "彩噪 (0纯亮度噪/1各通道独立)", 0.15, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=1.0,
               help="**低照度下噪点是有颜色的**：实测三个 NIGHTSHOT 帧的暗部逐通道高频能量为"
                    "0.00736 / 0.00659 / 0.00836，通道间相差 27%。"
                    "read_noise 是单一标量，**无论开多大都做不出这个通道差**——这是性质缺失而不是没调好。"
                    "本项按 read_noise 的比例为每个通道独立抽样并减去三通道均值，"
                    "所以它**只加颜色、不改变噪声对亮度的贡献**。")
    cam.define("nv_illuminator_visible", "照明可见泄漏", 0.02, 0.0, 0.25, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.15,
               help="红外灯自身的微弱可见泄漏——**所以近场有紫偏、远场没有**，"
                    "这是给【灯】上色而不是给画面上色。")
    cam.define("source_glow", "光源色散射晕", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="**光源的光在空气和镜片里散射，散射光带的是光源自己的颜色**——"
                    "橙灯周围是橙晕、白灯周围是白晕。"
                    "这是它和「高光光晕」的本质区别：后者的色调是固定的（偏冷），"
                    "也和「镜头光斑」不同，后者的色调是镀膜色。**两者都不跟随光源。**"
                    "实测参考素材：炸白核心下方 60 行和 100 行处 B−R 到 −0.170 和 −0.510，"
                    "而模型只有 −0.014 和 −0.034——差 15 倍，这是近场最大的缺口。")
    cam.define("overflow_drain", "溢出暗环（深度）", 0.0, 0.0, 0.8, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.4,
               help="**光点溢出时，反溢出电路要抽走多余电荷，而它是从邻居那里抽的**，"
                    "所以高光外圈被压到**低于背景**——真实画面里炸白点周围那圈暗边。"
                    "注意方向和光晕相反：光晕向外**加亮**，这个向外**压暗**，"
                    "所以任何光晕参数都做不出它。深度是**该点自身亮度的一个比例**，"
                    "不是减一个常数：亮背景上暗环更深，全黑背景没有电荷可抽、也就没有暗环。")
    cam.define("overflow_width", "溢出暗环（宽度）", 2.0, 1.0, 6.0, unit="像素", group=g,
               phys_min=1.0, phys_max=3.0,
               help="暗环宽度，单位是**感光点**——抽电荷的电路作用于相邻的光电二极管，"
                    "不是画面上的某个距离，所以这个宽度随传感器的像素尺寸走。")
    cam.define("ccd_gain", "增益档位", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=1.0, source="AGC 在低照度下提升增益",
               help="低照度时自动增益提升，会同时放大读出噪声。这是夜间画面颗粒感的来源。")
    g = "色彩科学"
    cam.define("color_cast", "色彩倾向", 0.0, -1.0, 1.0, unit="0-1", group=g,
               phys_min=-0.5, phys_max=0.5, source="各品牌 CCD 滤色片与 DSP 色调",
               help="**正值偏冷（偏青蓝）——Sony 消费级常见；负值偏暖（偏红）。**\n"
                    "**此前的注释把这两个方向写反了**，而代码一直是正确的：`_color_cast` 施加的增益是\n"
                    "    `[1 − 0.16c,  1 + 0.03c,  1 + 0.20c]`   （R, G, B）\n"
                    "所以 c > 0 把 R 压低、B 抬高（偏冷），c < 0 把 R 抬高、B 压低（偏暖）。\n"
                    "**用户看出来的正是这个矛盾**：界面文字说「负值偏青蓝」，而滑块与画面给的是相反的方向。\n"
                    "这会整体改变画面的时代感。")
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
    cam.define("spike_spread", "衍射星芒分色", 0.6, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**孔径的衍射星芒不是一张图样**：条纹间距随波长变化，"
                    "所以红、绿、蓝形成星芒的**角度和长度都略有不同**，"
                    "结果是**彩色的射线**——高光向外散出分立的红绿蓝条纹。"
                    "消色差的星芒无论调多大都做不出它：一个几何图形乘上一个颜色，"
                    "三个通道拿到的是同一组射线。0 = 消色差，1 = 明显的分色。")
    cam.define("flare_defocus", "闪光逐通道离焦", 0.5, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**纵向色差让三个通道聚焦在不同距离上**，所以孔径的**离焦像在蓝通道比红通道更大**。"
                    "结果是亮光源**周围和下方**出现彩色光晕，而不是光源本身被染色——"
                    "参考素材里路灯下方那条蓝纹、灯杆上的蓝色拖尾、以及炸白灯周围的硬边环，都指向它。"
                    "它和「01 成像系统」的 `纵向色差` 是同一套物理，"
                    "但那个作用于**画面**，这个作用于**闪光**——闪光已经是离开孔径焦点之后的光。")
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
    # The receiver's Y/C separator.  Four real designs of different eras, and the choice
    # is what decides the shape of the coloured fringes on hard edges: a cheap set uses a
    # band split, a better one a glass delay line, a good one two lines or a 2-D comb.
    fmt.define("separation_kind", "Y/C 分离方式", 0.0, 0.0, 5, unit="0-4", group=g,
               step=1.0, phys_min=0.0, phys_max=4.0,
               source="消费级接收机的四种分离方案（同年代分层）",
               help="**这是器件选择，不是效果旋钮。** 0=带通分离（廉价机，无延迟线）；"
                    "1=盒式孔径（小横向孔径，载波只被部分抵消，所以彩边粗而宽）；"
                    "2=1H 延迟线（一行玻璃延迟线，抽头相隔一行，副载波自然反相抵消）；"
                    "3=2H 延迟线（0.5×本行 + 0.25×上一行 + 0.25×下一行，垂直孔径更宽，"
                    "串色更少但垂直亮度更糊）；4=二维梳状（横向 ±1 像素、纵向 ±1 行，"
                    "**对角抽头与中心同相位**，所以不需要延迟线就能去掉载波，四者中垂直拖尾最小）。"
                    "梳状滤波器的相位抵消是**自动产生**的，不是靠系数调出来的——"
                    "这就是为什么真实接收机在这几种方案之间选择，"
                    "而**选哪一种决定了边缘彩边的形态**。")
    fmt.define("chroma_vert_blend", "垂直色度反馈", 0.5, 0.0, 0.99, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               help="**递归垂直色度反馈**——每行的色度是上一行**输出**的一半加当前输入的一半，"
                    "也就是沿列的一阶 IIR，而不是邻行平均。"
                    "这正是 ntsc-rs 的 `chroma_vert_blend`，也是它自己承认的缺陷来源："
                    "输出依赖像素上方的整段历史，所以颜色会拖出几十行。"
                    "**对称平均做不到这件事**——平均在两条行内就消失，而且它是对称的，不会漂移。"
                    "实测参考片：行间色度相关从 lag1 的 +0.98 掉到 lag16 的 +0.20，"
                    "相干长度约 20-30 行且**没有固定周期**（因为反馈是按列独立的，每列历史不同）。"
                    "注意：**只在帧内递归**，不跨帧携带状态——跨帧会把相邻帧耦合起来，"
                    "从而破坏「任意顺序请求帧、缓存仍然正确」这条前提。")
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
    # A deck's own separator.  Kept separate from the format layer's copy because a
    # camcorder and a deck are two different receivers, and each chose its own design.
    ply.define("separation_kind", "Y/C 分离方式", 0.0, 0.0, 5, unit="0-4", group=g,
               step=1.0, phys_min=0.0, phys_max=4.0,
               source="消费级接收机的四种分离方案（同年代分层）",
               help="**器件选择，不是效果旋钮。** 0=带通分离（廉价机，无延迟线）；"
                    "1=盒式孔径（小横向孔径，载波只被部分抵消，彩边粗而宽）；"
                    "2=1H 延迟线（抽头相隔一行，副载波自然反相抵消）；"
                    "3=2H 延迟线（垂直孔径更宽，串色更少但垂直亮度更糊）；"
                    "4=二维梳状（横向 ±1 像素、纵向 ±1 行，**对角抽头与中心同相位**，"
                    "不需要延迟线即可去载波，四者中垂直拖尾最小）。"
                    "**这是播放设备的属性**，与 format 层的同名参数相互独立——"
                    "摄像机和录像机是两台不同的接收机。")
    ply.define("chroma_vert_blend", "垂直色度反馈", 0.5, 0.0, 0.99, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               help="递归垂直色度反馈，说明见 format 层的同名参数。"
                    "**播放端也需要它**，因为哪个阶段做分离取决于接线；"
                    "参数只放在 format 层而播放端读自己的层，设置就会静默失效——"
                    "这个项目在 separation_kind 上已经犯过一次同样的错。")
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
    # **The minimum was 1.5, which made the parameter one-sided.**  A gamma is a curve that can go either
    # way, and the stage applies `rgb ** (1/g)`, so a value below 1 darkens the mid-tones.  Measured, asking
    # for 0.50 / 0.80 / 1.00 / 1.25 all left the stage holding **1.5** because the schema raised them, so
    # the lower half of the range the name implies was unreachable rather than mis-implemented.
    dsp.define("gamma", "伽马", 2.35, 0.4, 3.2, unit="x", group=g,
               phys_min=0.4, phys_max=2.8, source="消费级 CRT 典型 2.2-2.4")
    dsp.define("contrast", "对比度", 0.55, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("brightness", "亮度", 0.5, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("saturation", "饱和度", 0.5, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("scanlines", "扫描线", 0.35, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("interlace_flicker", "隔行闪烁", 0.25, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("geometry_warp", "几何形变", 0.15, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("trapezoid_x", "梯形（宽）", 0.0, -0.5, 0.5, unit="0-1", group=g,
               phys_min=-0.15, phys_max=0.15,
               help="**偏转线圈的扫描与显像管端面不成直角**，所以上边缘与下边缘的宽度可以不等。"
                    "这就是梯形失真（keystone），磁轭被撞歪的机器上很明显。正值下宽上窄。"
                    "和桶形/枕形畸变是**互相独立**的两种故障。")
    dsp.define("trapezoid_y", "梯形（高）", 0.0, -0.5, 0.5, unit="0-1", group=g,
               phys_min=-0.15, phys_max=0.15,
               help="左右两侧高度不等。与横向梯形是两个独立的故障，所以是两个参数。")
    dsp.define("convergence", "会聚误差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.6,
               help="**三个电子束从三支枪以不同角度出发、穿过荫罩**，所以红绿蓝三幅像的大小"
                    "并不完全一致。误差在**中心为零、向边缘增大**——这就是真实显像管"
                    "**边缘有彩色镶边、中心却干净**的原因。模型把各通道绕中心做略微不同的缩放，"
                    "而不是整体平移：整体平移会让整幅画面均匀镶边，那是廉价近似的做法。")
    dsp.define("curvature", "管面曲面（鱼眼）", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="**显像管端面是球面**，所以画面绕中心被放大、**边缘向外鼓出**。"
                    "映射是随半径增大的径向放大：**中心几乎不动，边缘明显**——"
                    "这正是「鱼眼」和「整体缩放」的区别。和梯形（磁轭歪）是两个独立故障："
                    "一台机器可以管面是弯的、而磁轭对得很正。")
    dsp.define("spot_sigma", "电子束光点", 0.42, 0.10, 2.5, unit="行", group=g,
               phys_min=0.25, phys_max=1.0,
               help="**扫描线不是画上去的暗条纹，是电子束两次扫过之间的空隙**，"
                    "而空隙的形状取决于**光点的有限尺寸**。这里是高斯光点的宽度（单位：行）。"
                    "光点小于行距时空隙变黑、线条清晰；光点接近行距时相邻两次扫描重叠、"
                    "**扫描线结构就消失了**——**这就是调焦会改变扫描线可见度的原因**，"
                    "而固定周期的余弦永远做不出这件事。")
    dsp.define("scan_pitch", "扫描行距", 2.0, 1.05, 6.0, unit="px", group=g,
               phys_min=1.05, phys_max=3.0,
               help="两次扫描之间的像素距离。**这是显示器的属性，不是信号的**："
                    "480 行的画面并不必然对应 480 像素的面板。")
    dsp.define("mask_pitch", "荫罩节距", 0.0, 0.0, 8.0, unit="px", group=g,
               phys_min=0.0, phys_max=4.0,
               help="**荫罩是彩色显像管里那块金属栅格，它的影子就印在玻璃上。**"
                    "这不是只有水平方向的效果：栅格的图案是**竖直**的，与扫描线成直角。"
                    "缺了它，显像管看起来就只是「一幅画加几条横线」。节距是**玻璃的属性、"
                    "不是信号的**，所以同一幅画面在更细的管子上会显示更细的栅格。"
                    "0 表示关闭。")
    dsp.define("mask_type", "荫罩型式", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0 **栅条式**（特丽珑：连续的竖直细线）；"
                    "1 **槽孔式**（竖直缝交替错开，呈砖块状）；"
                    "2 **点孔式**（最常见的荫罩，孔在错开的二维点阵上）。"
                    "**图案本身就是管子的身份**，所以这是三个选项而不是一个强度。")
    dsp.define("mask_depth", "荫罩深度", 0.35, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.1, phys_max=0.6,
               help="栅格影子的深浅。真实的取值取决于管子多亮、以及观看距离；"
                    "它**不改变画面平均亮度**（掩膜按均值归一化），"
                    "否则这个参数就成了名字骗人的亮度控制。")
    dsp.define("breathing", "高压呼吸", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**末级阳极由行输出变压器供电，而它的调整率是有限的。**"
                    "画面越亮、束电流越大、电压越塌，而**偏转灵敏度随阳极电压下降而上升**，"
                    "于是**画面变大**。所以从暗场景切到亮场景的瞬间，画幅会胀一下。"
                    "这是一个**动态**故障，所以它读的是当前帧的亮度而不是一个设定值——"
                    "也是第一个逐帧变化的几何项。")
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

    _apply_unit_spreads(s)
    # --- 字幕机（字符发生器）----------------------------------------------------
    # **字符发生器不是排版引擎。** OSD 芯片有字模 ROM、每字一个固定字格、一个位置寄存器；
    # 它把字形画进格子，**不测量、不调字距、不换行、不重排**。这不是简化，而是这台设备实际做的事，
    # 也正是摄像机叠字看起来那样的原因：等距、定尺寸、永不比例化。
    #
    # 用户给的三个字体**全部是点阵来源**（实测：VCR-OSD-Mono 204 字形仅 ASCII、
    # 柳宋 7577 字形含 6763 汉字、A-POS 22169 字形含 20902 汉字），
    # 而且**西文与汉字的前进宽度正好 1:2**（1200:2400、504:1008）——
    # 所以一个汉字占两个半格，这就是那个年代的半角/全角约定。
    osd = s.layer("overlay")
    g = "字幕机"
    osd.define("at", "发生器位置", 0.0, 0.0, 1.0, unit="0-1", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="**0 = 画面成形之后**（摄像机自身的 OSD）：字幕写在已成形的画面上，"
                    "所以之后发生的磁带损伤落在画面上、**不落在字上**。"
                    "**1 = 介质之前**（字幕机往磁带上录）："
                    "录像机的失落、磁头磨损、时基误差**连字一起毁**。"
                    "两者都是真实设备；差别正好在磁带很差的时候看得出来。")
    osd.define("enabled", "启用字幕机", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0)
    osd.define("font", "字体", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0 = **VCR OSD Mono**（仅英文/数字——对汉字会**明确报告缺字**而不是画方块）；"
                    "1 = **柳宋**（中英文数字，6763 汉字）；2 = **A-POS 点阵**（中英文数字，20902 汉字）。"
                    "字体不可用时不会让渲染失败：叠字是可选的，而链路其余部分已经做完了。")
    osd.define("text", "字幕文本", "", 0.0, 1.0, unit="text", kind="text", group=g,
               help="可多行（换行即多行）。**不换行、不重排**——一行超出右边缘的部分直接丢失，"
                    "因为真实字符发生器没有断行逻辑。汉字占两格、西文占一格。")
    osd.define("cell_h", "字格高度", 24.0, 8.0, 96.0, unit="px", group=g, step=1.0,
               phys_min=12.0, phys_max=48.0,
               help="字格像素高度；**半格宽度由字体自己的前进宽度换算而来**，不是猜测值。"
                    "把格子当成长度的一半会让字形与格子错位、相邻字符串重叠——第一版就是这样。")
    osd.define("halign", "水平对齐", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0 左、1 居中、2 右，都在安全区内。**字符发生器没有排版引擎，但有位置寄存器**——"
                    "OSD 芯片本来就能把一行放在安全区的左/中/右。对齐不需要断行，所以它不是排版功能，"
                    "而是这台设备本来就有的控件。**再也不用打空格来挪字幕了。**")
    osd.define("valign", "垂直对齐", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0 上、1 中、2 下。多行时整个文字块一起对齐，所以**行间距由字格高度决定**，"
                    "不需要靠空行来撑。")
    osd.define("pos", "位置预设", 0.0, 0.0, 8.0, unit="0-8", group=g, step=1.0,
               phys_min=0.0, phys_max=8.0,
               help="一次设好水平与垂直，省得调两个参数：0 左上、1 上中、2 右上、3 左中、4 正中、"
                    "5 右中、6 左下、7 下中、8 右下。**8（右下）是家用录像最常见的 OSD 位置。**")
    osd.define("col", "列偏移", 0.0, 0.0, 60.0, unit="格", group=g, step=1.0)
    osd.define("row", "行偏移", 0.0, 0.0, 20.0, unit="行", group=g, step=1.0)
    osd.define("margin", "安全区边距", 0.03, 0.0, 0.25, unit="比例", group=g,
               phys_min=0.02, phys_max=0.12)
    osd.define("shadow", "投影强度", 0.5, 0.0, 1.0, unit="0-1", group=g,
               help="字下方偏移一份暗影。这是消费级发生器为了保证亮场景可读性用的最省的做法。")
    osd.define("outline", "字形描边", 1.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="**每个字形外面一圈深色边。默认开启，这是真实摄像机的做法。**"
                    "它让字幕在任意画面上都可读，**而且不遮挡画面**——"
                    "这正是摄像机自身 OSD 用的办法，也是之前白字压白画面时字幕消失的原因。"
                    "一像素的膨胀就是一圈边，代价是三次平移取最大，而不是卷积。")
    osd.define("background", "底色块", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               help="在文字后面烧一个实心黑框。**默认随位置**："
                    "「画面成形之后」（摄像机自身 OSD）**不画框**——真实手持机不会遮住一块画面；"
                    "「介质之前」（字幕机录到磁带）**默认画框**，因为那个位置的字要压在磁带上、"
                    "底下是什么画面无法预期。显式设定会覆盖这个默认。")
    osd.define("opacity", "不透明度", 1.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.5, phys_max=1.0)
    osd.define("red", "字色 R", 0.92, 0.0, 1.0, unit="0-1", group=g)
    osd.define("green", "字色 G", 0.94, 0.0, 1.0, unit="0-1", group=g)
    osd.define("blue", "字色 B", 0.90, 0.0, 1.0, unit="0-1", group=g)

    # --- style: the outline has its own pen, and the glyph can be sheared -----------------
    osd.define("italic", "斜体（剪切）", 0.0, -0.8, 0.8, unit="shear", group=g,
               phys_min=0.0, phys_max=0.4,
               help="**点阵发生器没有真正的斜体，它是把字形剪切。**"
                    "所以斜着的 OSD 文字边缘是台阶状的——剪切作用在**阈值化之后**的掩膜上，"
                    "像素结构因此保留下来。正值向右倾。")
    osd.define("stretch_x", "字宽（拉伸）", 1.0, 0.4, 2.5, unit="x", group=g,
               phys_min=0.75, phys_max=1.35,
               help="横向缩放字形。**点阵发生器画瘦高字靠的是改字格，不是滤波**，"
                    "所以这里是**在掩膜上重采样**，硬边缘保留下来。1.0 是字体本身的比例。"
                    "和下面的「字高」是**两个独立轴**：真实机器上横向由时钟、纵向由行数决定。")
    osd.define("stretch_y", "字高（拉伸）", 1.0, 0.4, 2.5, unit="x", group=g,
               phys_min=0.75, phys_max=1.35,
               help="纵向缩放字形。与「字宽」独立。把两个字轴分开，"
                    "才能做出当年窄体/宽体字幕机那种字形。")
    osd.define("letter_gap", "字距", 0.0, -4.0, 12.0, unit="px", group=g, step=1.0,
               phys_min=-1.0, phys_max=4.0,
               help="**字符之间**额外增加的像素。这与「字宽」是不同的控制："
                    "真实发生器的字格是固定的，而字符之间的空隙取决于 ROM 被读出的方式。"
                    "负值让字挤在一起，那是当年为了在有限字格里塞下更多字的做法。")
    osd.define("line_gap", "行距", 0.0, -8.0, 24.0, unit="px", group=g, step=1.0,
               phys_min=-2.0, phys_max=8.0,
               help="**行与行之间**额外增加的像素。")
    osd.define("outline_red", "描边色 R", 0.02, 0.0, 1.0, unit="0-1", group=g)
    osd.define("outline_green", "描边色 G", 0.03, 0.0, 1.0, unit="0-1", group=g)
    osd.define("outline_blue", "描边色 B", 0.05, 0.0, 1.0, unit="0-1", group=g)
    osd.define("shadow_x", "阴影横向偏移", 2.0, -8.0, 8.0, unit="px", group=g, step=1.0)
    osd.define("shadow_y", "阴影纵向偏移", 2.0, -8.0, 8.0, unit="px", group=g, step=1.0)
    osd.define("style", "样式模板", 0.0, 0.0, 5.0, unit="0-5", group=g, step=1.0,
               phys_min=0.0, phys_max=5.0,
               help="一次设好字色、描边色和斜体：**0 自定义**、1 绿字黑边（OSD 经典）、"
                    "2 红字黑边（REC 经典）、3 白字黑边、4 黄字黑边、5 青字黑边。"
                    "**0 以外的值会覆盖上面那些颜色参数。**")

    return s



# ---------------------------------------------------------------------------
# per-unit spread: which parameters differ between two machines of one model
# ---------------------------------------------------------------------------
# The value of this table is as much in what it omits as in what it contains. A spread is
# declared only where there is a reason to expect unit variation:
#
#   * alignment and gain tolerances -- an amplifier's gain is set by resistors with a
#     tolerance, so the same model comes out a few per cent apart; a servo loop is trimmed;
#   * component tolerances whose effect is known -- a CCD's dark current varies per die,
#     which is sensor noise; an electrolytic capacitor's ESR varies, which is supply ripple;
#   * wear, which differs per unit because it depends on use rather than on manufacture.
#
# Artistic parameters (`halation`, `lens_flare`, `highlight_tint`) are left at zero on purpose.
# An invented spread is worse than none: it looks like provenance and stops the next person
# from checking. The fraction is one sigma, clipped to two when a unit is drawn, because an
# alignment tolerance is a bound and beyond twice it the machine is broken rather than a unit
# of the model.
UNIT_SPREADS: dict[str, tuple[float, str]] = {
    # camcorder sensor -- per-die variation, which is what "this camera" means
    "sensor_noise": (0.30, "CCD dark current and read noise vary per die"),
    "fixed_pattern": (0.25, "per-die fixed-pattern non-uniformity"),
    # camcorder lens and signal path -- alignment tolerances
    "chroma_aberration": (0.20, "lens centring tolerance"),
    "sharpness": (0.12, "aperture-correction trim"),
    "ccd_gain": (0.15, "video amplifier gain trim"),
    "ccd_smear": (0.12, "transfer efficiency varies per device"),
    "dynamic_range": (0.06, "well capacity and clip point"),
    "color_cast": (0.05, "white-balance null trim; a small residual cast is normal"),
    "exposure": (0.10, "AE reference trim"),
    "white_balance_temp": (0.12, "AWB reference trim"),
    # deck -- servo trims and head wear
    "head_wear": (0.35, "wear depends on use, not on the model"),
    "transport_jitter": (0.20, "capstan servo gain trim"),
    "frequency_response": (0.15, "replay equaliser trim"),
    "luma_bandwidth": (0.05, "replay amplifier bandwidth trim"),
    "chroma_bandwidth": (0.05, "chroma amplifier bandwidth trim"),
    "comb_depth": (0.08, "delay-line and summing network tolerances"),
    "cross_color": (0.10, "separation residue depends on the comb's alignment"),
    "cross_luma": (0.10, "as above"),
    "burst_phase_error": (0.60, "APC loop and crystal tolerance; a few degrees is normal"),
    "tbc_strength": (0.08, "corrector loop gain trim"),
    "tbc_residual": (0.25, "corrector residual after alignment"),
    "ringing": (0.20, "replay filter Q"),
    "color_noise": (0.20, "chroma amplifier noise figure"),
    # display -- convergence, geometry and drive trims
    "gamma": (0.04, "gun and video-drive tracking"),
    "geometry_warp": (0.20, "yoke and convergence adjustment"),
    "defocus": (0.15, "focus and astigmatism adjustment"),
    "contrast": (0.08, "video drive preset"),
    "brightness": (0.06, "cut-off preset"),
    "saturation": (0.08, "chroma drive preset"),
    "color_temp": (0.06, "grey-scale tracking"),
    "scanlines": (0.10, "beam spot size and focus"),
    "vignette": (0.12, "shadow-mask and yoke geometry"),
    "interlace_flicker": (0.10, "field timing and phosphor persistence"),
    "crt_age": (0.30, "hours of use, not the model"),
    "bloom": (0.10, "high-voltage regulation per set"),
}


def _apply_unit_spreads(s: "ParamSchema") -> None:
    """
    Set each declared spread on its parameter, and check the name exists.

    Setting the field rather than editing the define calls keeps the declaration in one readable
    table -- with the reason next to the number -- and cannot break the file, which two earlier
    attempts at splicing did.
    """
    for key, (spread, _why) in UNIT_SPREADS.items():
        found = False
        for layer in s.layer_names():
            specs = s.layer(layer).specs
            if key in specs:
                specs[key].unit_spread = float(spread)
                found = True
        if not found:
            raise KeyError("UNIT_SPREADS names a parameter that does not exist: %r" % key)


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

    # --- demonstrations of the dimensions added later -------------------------
    # Every one of these defaults to off -- an unrequested dimension must not move an existing preset --
    # so without a preset that switches them on, nobody would find them.
    "cold_start_handheld": {
        "label": "冷机开机 · 手持（新维度演示）",
        "note": "一台冷机在手里拍的：三个自动环还在收敛、手在抖、录像机电容已经老化。"
                "注意画面亮度在前两秒里还在变——那是 AE 在追目标，不是曝光参数在动。",
        "kind": "composite",
        "layers": {
            "camera": {
                # the loops, all settling from switch-on
                "ae_enabled": 1.0, "ae_tau_s": 0.9, "ae_target": 0.44,
                "awb_enabled": 1.0, "awb_tau_s": 1.6,
                "af_enabled": 1.0, "af_tau_s": 0.5,
                # the hand
                "shake_intensity": 0.85, "bump_rate_hz": 0.5, "bump_amount": 0.7,
                # a consumer CCD, as these machines were
                "sensor_type": 0.0, "sensor_noise": 0.075, "fixed_pattern": 0.05,
                "sharpness": 0.8, "chroma_aberration": 0.06, "ccd_smear": 0.4,
                "halation": 0.5, "dynamic_range": 7.2, "lens_wear": 0.12,
                # the note's optical and sensor faults, all belonging to this one camera.
                # `lens_fog` is deliberately *not* set here: it is a state that clears as the glass
                # warms, so it is driven by the timeline below, alongside the warmth.
                "bloom_amount": 0.35, "bloom_threshold": 0.80,
                "dead_sites": 2.0, "stuck_fraction": 0.5,
            },
            # head wear is the machine's, not the medium's: it is on the player layer, which is also
            # where its per-unit spread lives
            "tape": {"dropout_rate": 0.9, "tracking_error": 0.16},
            "player": {"head_wear": 0.34, "burst_phase_error": 1.4, "frequency_response": 0.85},
            "display": {"bloom": 0.4, "scanlines": 0.35, "brightness": 0.46, "contrast": 1.05},
        },
        # the machine's condition and the timeline that moves it, together
        "environment": {"warmth": 0.0, "age": 0.65, "humidity": 0.2},
        "timeline": [
            {"layer": "environment", "key": "warmth", "absolute": True,
             "points": [{"t": 0.0, "value": 0.0, "curve": "exp"},
                        {"t": 6.0, "value": 1.0, "curve": "hold"}]},
            # the fog clears as the front element warms: the same physical process as the warmth track
            # above, acting on the glass instead of on the electronics.  It starts heavy and falls away
            # over about eight seconds.
            {"layer": "camera", "key": "lens_fog", "absolute": True,
             "points": [{"t": 0.0, "value": 0.55, "curve": "exp"},
                        {"t": 8.0, "value": 0.02, "curve": "hold"}]},
        ],
    },
    "warm_archival": {
        "label": "已预热的存档带（对照组）",
        "note": "同一台机器、同一盘带，但机器已经热了、手是稳的、自动环已经收敛。"
                "和上面那条对照着看，能分出哪些是机器的状态、哪些是介质本身的缺陷。",
        "kind": "composite",
        "layers": {
            "camera": {
                "ae_enabled": 1.0, "ae_tau_s": 0.9, "ae_target": 0.44,
                "awb_enabled": 1.0, "awb_tau_s": 1.6,
                "sensor_type": 0.0, "sensor_noise": 0.075, "sharpness": 0.8,
                "chroma_aberration": 0.06, "ccd_smear": 0.4, "halation": 0.5,
                "dynamic_range": 7.2, "lens_wear": 0.12,
                # the same camera, so the same failed sites and the same iris -- but warm, so no fog
                "bloom_amount": 0.35, "bloom_threshold": 0.80,
                "dead_sites": 2.0, "stuck_fraction": 0.5,
            },
            # head wear is the machine's, not the medium's: it is on the player layer, which is also
            # where its per-unit spread lives
            "tape": {"dropout_rate": 0.9, "tracking_error": 0.16},
            "player": {"head_wear": 0.34, "burst_phase_error": 1.4, "frequency_response": 0.85},
            "display": {"bloom": 0.4, "scanlines": 0.35, "brightness": 0.46, "contrast": 1.05},
        },
        "environment": {"warmth": 1.0, "age": 0.65, "humidity": 0.2},
        "timeline": [],
    },
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
