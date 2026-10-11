"""
The fault catalogue: the whole signal chain, stage by stage.

This is the apparatus described as a *diagnostic table* rather than as a set of
knobs.  The reference material for this project arrived as exactly such a table --
tape and cassette, video head and drum, rotary transformer and preamp, luminance
channel, chroma channel, servo and transport, control/power, RF output, the
receiver, and the environment -- and the value of writing it down this way is that
it makes the *mechanism* explicit and shows honestly which mechanisms this tool
models, which it approximates, and which it does not model at all.

Two rules are enforced here:

* Every entry names a **physical quantity**, not a look.  "Chroma plays back at the
  wrong frequency because the 4.43 MHz VCO drifted" is a mechanism; "colours look
  wrong" is not.
* An entry that the simulator does not implement says so, in `model`.  A fault
  catalogue that claims completeness it does not have is worse than a short one,
  because it stops people looking for the real cause.

Where an entry maps onto parameters, `params` lists them as
`("layer", "key")` so the UI can jump straight from a symptom to the control.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fault:
    """One failure mode of one part of the chain."""
    stage: str                 # which link in the chain
    symptom: str               # what is seen on screen (or heard)
    causes: str                # the failing part
    physics: str               # the physical quantity that changes
    model: str                 # how this simulator expresses it, or that it does not
    params: tuple = ()         # (layer, key) pairs that drive it
    status: str = "modelled"   # modelled | approx | not-modelled

    @property
    def is_modelled(self) -> bool:
        return self.status in ("modelled", "approx")


# ---------------------------------------------------------------------------
# The chain, in signal order.  Stage keys line up with the node graph so the
# canvas and this table describe the same apparatus.
FAULTS: tuple[Fault, ...] = (
    # ---------------------------------------------------------------- 磁带/带盒
    Fault(
        stage="tape", symptom="雪花、彩色弱、声音差、走带不稳、间歇",
        causes="磁带划伤、磁粉脱落、发霉、消磁、带盒张力不良",
        physics="磁介质损失、表面粗糙、摩擦系数变化、磁迹信噪比下降",
        model="磁带磨损/表面损伤/粒子污染/色度丢失行，由 seeded 损坏计划驱动",
        params=(("tape", "wear"), ("tape", "surface_damage"),
                ("tape", "debris"), ("tape", "chroma_loss")),
    ),
    Fault(
        stage="tape", symptom="偶发白/黑横条、瞬间雪花",
        causes="磁粉脱落点、灰尘、划伤",
        physics="局部磁通缺失，RF 包络瞬间跌落",
        model="失落事件：Poisson 到达过程，位置与长度按行给出（可见缺陷时间线）",
        params=(("tape", "dropout_rate"),),
    ),
    Fault(
        stage="tape", symptom="满屏雪花、完全无色",
        causes="磁带消磁、录错机型（不同速度档录制）",
        physics="磁迹剩磁接近噪声底",
        model="可近似：把 noise_floor 推到极端并关闭 chroma",
        params=(("tape", "noise_floor"),),
        status="approx",
    ),

    # ------------------------------------------------------------ 视频磁头/磁鼓
    Fault(
        stage="head", symptom="纯雪花、噪点大、无彩色、清晰度差、时好时坏",
        causes="磁头脏、磨损、断裂、磁鼓转速不对",
        physics="积垢、磁头尖磨损、磁隙污染、轴承磨损、鼓伺服失锁",
        model="磁头堵塞（起格/条纹）、磁头磨损（高频损失+噪声）、"
              "磁鼓速度偏差（行频漂移）",
        params=(("tape", "head_clog"), ("tape", "clog_bands"),
                ("player", "head_wear"), ("tape", "tape_stability")),
    ),
    Fault(
        stage="head", symptom="画面底部一条细横带",
        causes="两个视频磁头换接",
        physics="切换期间无有效信号，色度载波丢失",
        model="磁头切换带：每场末尾 N 行，硬边界，色度载波丢失",
        params=(("tape", "head_switching"), ("tape", "head_switch_lines"),
                ("tape", "head_switch_mid_line")),
    ),
    Fault(
        stage="head", symptom="画面绕动、上部扭曲、行不同步",
        causes="鼓伺服/主导伺服、CTL 丢失、张力不对、导柱高度/倾斜偏差",
        physics="走带控制轨迹与磁头位置相位差",
        model="跟踪误差（整条切换带被横向推开）+ 跟踪失谐"
              "（底部向上渐强的失控斜坡：位移+雪花）",
        params=(("tape", "tracking_error"), ("tape", "tracking_noise_rows"),
                ("tape", "tracking_noise_wave"), ("tape", "tracking_noise_snow")),
    ),

    # ------------------------------------------------------- 旋转变压器/前置放大
    Fault(
        stage="preamp", symptom="无图像、信号弱、噪点、彩色差",
        causes="旋变开路/短路、前置放大坏、供电不良",
        physics="线圈断、接触氧化、焊点开裂、元件老化",
        model="未建模：本工具的链路假设前置放大工作正常。"
              "若要模拟，应从 RF 包络整体跌落入手，而不是加噪声",
        status="approx",
    ),

    # ----------------------------------------------------------------- 亮度通道
    Fault(
        stage="luma", symptom="无图像、画面暗、模糊、拖尾、白切、同步差",
        causes="FM 解调、限幅、去加重、失落补偿异常",
        physics="电容干涸、IC 坏、滤波失谐、虚焊",
        model="亮度带宽（模糊）、轮廓增强/高频预加重（白切与过冲）、"
              "输出增益与黑电平偏移（暗/黑电平失控）",
        params=(("format", "luma_bandwidth"), ("format", "edge_boost"),
                ("player", "preemphasis"), ("player", "ringing"),
                ("player", "output_gain"), ("player", "output_bias")),
    ),
    Fault(
        stage="luma", symptom="横向拖尾、边缘镶边",
        causes="去加重网络失谐、群延迟不平坦",
        physics="相位响应非线性，边缘能量在时间上扩散",
        model="IIR 滤波器家族（恒 K 型/巴特沃斯带群延迟）与振铃；"
              "零相位高斯做不出这个效应",
        params=(("format", "filter_kind"), ("format", "filter_sections"),
                ("player", "ringing"), ("player", "ringing_freq")),
    ),

    # ------------------------------------------------------------ 色度通道/色度板
    Fault(
        stage="chroma", symptom="无彩色、彩色闪、偏色、色不同步、彩色噪点、彩色拖尾",
        causes="ACC/APC 不稳、4.43 MHz 振荡漂移、627 kHz 变换异常、色度记录放大坏",
        physics="晶振漂移、电容变值、供电纹波、温漂、虚焊、IC 热稳定性差",
        model="色度相位误差（偏色/色不同步）、色度带宽（彩色拖尾）、"
              "色度噪声（彩色闪）、色度丢失行",
        params=(("player", "burst_phase_error"), ("format", "chroma_bandwidth"),
                ("format", "color_noise"), ("tape", "chroma_loss")),
    ),
    Fault(
        stage="chroma", symptom="彩色横条、逐场交替的色带",
        causes="信号源（测试卡）色度发生器故障；或多代复制累积",
        physics="色度通道被周期性调制，而亮度通道不受影响",
        model="注意：这一项**不是** VHS 固有特性。实测参考素材里色度纹波是"
              "基线的 3.1 倍而亮度仅 1.4 倍，说明故障在源的色度路径。"
              "标准测试图 tools/out/test_card.png 复现了该外观，"
              "但链条本身不产生它",
        status="not-modelled",
    ),

    # ------------------------------------------------------------- 伺服/走带机构
    Fault(
        stage="servo", symptom="行/场不同步、滚动、抖动、横条、速度不对、上部扭曲",
        causes="鼓伺服、主导伺服、CTL 丢失、张力不对、导柱/压带轮/皮带问题",
        physics="控制磁头脏、导柱高度/倾斜偏差、压带轮硬化、皮带松、"
                "轴承磨损、张力弹簧疲劳",
        model="磁带稳定度（行抖动/缓慢漂移/亚像素抖动/垂直不稳定）、"
              "走带速度档：SP/LP/EP 按磁迹宽度、重放输出、失落概率、"
              "跟踪敏感度四种机制分别退化",
        params=(("tape", "tape_stability"), ("tape", "line_jitter"),
                ("tape", "walk_amplitude"), ("tape", "micro_jitter"),
                ("tape", "vertical_instability"), ("tape", "tape_speed")),
    ),
    Fault(
        stage="servo", symptom="画面上部扭曲、顶部几行倾斜",
        causes="场同步后的伺服建立时间不足",
        physics="垂直消隐期结束到伺服锁定之间的过渡",
        model="可近似：head_switching 的深度斜坡已表现带内渐进，"
              "但「上部」的对称现象未单独建模",
        params=(("tape", "head_switching"),),
        status="approx",
    ),

    # ------------------------------------------------------------- 系统控制/电源
    Fault(
        stage="power", symptom="间歇故障、忽明忽暗、彩色闪、死机",
        causes="电源纹波、复位异常、传感器坏、保护误动",
        physics="电容干涸、虚焊、接插件氧化、温漂",
        model="未建模：本工具没有时间维度上的电源不稳定。"
              "若需要，应做成「整机故障事件」而非逐帧参数",
        status="not-modelled",
    ),

    # ----------------------------------------------------------- 射频调制器/输出
    Fault(
        stage="rf", symptom="电视雪花、网纹、颜色差、声音干扰",
        causes="RF 调制器坏、频道冲突、线缆差",
        physics="调制器失谐、屏蔽差、阻抗不匹配",
        model="射频链路：频道成型、中频噪声、邻频差拍、行 AFC 失锁"
              "（位移与色相同源，产生撕裂彩虹）",
        params=(("rf", "signal_dbm"), ("rf", "noise_figure_db"),
                ("rf", "sync_loss_threshold_db"), ("rf", "adjacent_channel"),
                ("rf", "channel_mhz")),
    ),

    # ------------------------------------------------------------------- 外部/电视
    Fault(
        stage="display", symptom="本机 AV 输出正常，但电视画面差",
        causes="电视制式不对、AV 线坏、接口氧化",
        physics="制式不匹配、线缆断/氧化、电视故障",
        model="显示阶段：CRT gamma、对比度/亮度/饱和度、扫描线、隔行闪烁、"
              "几何形变、过扫描、暗角、泛光、反射、制式（NTSC/PAL）",
        params=(("display", "gamma"), ("display", "scanlines"),
                ("display", "geometry_warp"), ("display", "overscan"),
                ("display", "bloom"), ("display", "reflection")),
    ),
    Fault(
        stage="display", symptom="颜色淡、无彩色",
        causes="电视色饱和度设定、制式不匹配（PAL 节目按 NTSC 解码）",
        physics="色度解调轴错误或副载波频率不匹配",
        model="饱和度参数；制式切换会改变副载波频率与行频",
        params=(("display", "saturation"),),
    ),

    # ---------------------------------------------------------------- 环境/保存
    Fault(
        stage="storage", symptom="间歇、彩色差、噪点、霉斑",
        causes="温湿度、灰尘、磁化、静电",
        physics="磁粉脱落、霉变、剩磁畸变、静电吸附",
        model="可近似：霉斑与灰尘表现为 spatially-fixed 的粒子污染"
              "（固定在磁带表面、随走带缓慢漂移）",
        params=(("tape", "debris"), ("tape", "surface_damage")),
        status="approx",
    ),
)


def by_stage(stage: str) -> tuple[Fault, ...]:
    return tuple(f for f in FAULTS if f.stage == stage)


def stages() -> tuple[str, ...]:
    seen: list[str] = []
    for f in FAULTS:
        if f.stage not in seen:
            seen.append(f.stage)
    return tuple(seen)


STAGE_LABELS: dict[str, str] = {
    "tape": "磁带 / 带盒",
    "head": "视频磁头 / 磁鼓",
    "preamp": "旋转变压器 / 前置放大",
    "luma": "亮度通道",
    "chroma": "色度通道 / 色度板",
    "servo": "伺服 / 走带机构",
    "power": "系统控制 / 电源",
    "rf": "射频调制器 / 输出",
    "display": "外部 / 电视",
    "storage": "环境 / 保存",
}


def coverage() -> dict:
    """How much of the catalogue this simulator actually expresses."""
    total = len(FAULTS)
    modelled = sum(1 for f in FAULTS if f.status == "modelled")
    approx = sum(1 for f in FAULTS if f.status == "approx")
    missing = [f for f in FAULTS if f.status == "not-modelled"]
    return {
        "total": total, "modelled": modelled, "approx": approx,
        "not_modelled": len(missing),
        "missing_symptoms": [f"{STAGE_LABELS.get(f.stage, f.stage)}：{f.symptom}"
                             for f in missing],
    }


def as_dicts() -> list[dict]:
    return [
        {
            "stage": f.stage,
            "stage_label": STAGE_LABELS.get(f.stage, f.stage),
            "symptom": f.symptom,
            "causes": f.causes,
            "physics": f.physics,
            "model": f.model,
            "params": [list(p) for p in f.params],
            "status": f.status,
        }
        for f in FAULTS
    ]
