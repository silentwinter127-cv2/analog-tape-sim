# Virtual Media Lab (虚拟媒介实验室) 物理架构路线图与系统架构备忘录 (Roadmap & Backlog)

本文档记录已全部完成的 **28 大核心物理、广播电视基础设施与视听媒介生态模块**（模块 2、模块 3、扩展单元 A ～ 扩展单元 U 全系收官、实拍物理模型专项校准、独立拆出的 **01b 胶片光化学与 LOMO 暗房冲印模块**，以及基于 **10 张新 LOMO 实拍实测数据的光学光晕/散景/移轴/旋焦与 16 款胶卷扩展**），全量自动化物理验证达到 **133 / 133 通过 (`133 passed, 0 failed`)**。

## 一、已完成推进模块

### 1. 模块 2：同步与时钟系统 (`atsim/codecs/sync_clock.py` & `player` 层 / `sync_tbc` 节点)

* **行同步 (H-Sync)、场同步 (V-Sync) 与消隐区 (HBI / VBI) 物理波形合成**：

  * **水平消隐带 (HBI)**：完整建模前肩 (`0 IRE`)、`-40 IRE` 行同步顶（随 `sync_clip` 与电源压降动态衰减）、后肩 (`0 IRE`) 以及 8–10 周期相位准确的 **色同步脉冲串 (Colour Burst)**（NTSC `3.58 MHz` 180° 基准 / PAL `4.43 MHz` $\pm 135^\circ$ 逐行摆动色同步）。

  * **垂直消隐带 (VBI)**：完整建模前均衡脉冲 ($2f_H$)、场同步阔脉冲与半行开槽齿 (Serration Pulses)、后均衡脉冲 ($2f_H$)、以及第 10–20 行的 VITC 双相时间码位流与基准白/色度测试行。

* **同步分离器、H-Hold 与 V-Hold 注入锁定振荡器 (`sync_clip`, `h_hold`, `v_hold`)**：

  * 建模 LM1881 / TA8690 同步分离比较器与行 AFC / 场自由振荡器（Adler 注入锁定微分方程）：小幅 `h_hold` 或同步脉冲被压缩时，场消隐后行 AFC 恢复滞后，在画面顶部形成经典 **弯钩畸变 (Top Flagging / Hook)**，并在高亮度扫描行发生交流耦合压低同步顶的 **亮行撕裂 (Sync Crushing)**；当 `|v_hold|` 或同步脉冲削波突破锁定范围时，画面按 Adler 注入锁定动力学发生非匀速纵向滚屏，并在上下场交界处显现完整的物理 **VBI 同步消隐带**。

* **采样时钟频偏 (`clock_drift_ppm`)、相位抖动 (`clock_jitter`)、PAL** $\leftrightarrow$ **NTSC 跨制式失配 (`std_mismatch`) 与脉冲十字检视 (`pulse_cross`)**。

### 2. 模块 3：共享电源与电气母线系统 (`atsim/codecs/power_bus.py` & `connector` 层 / `power_supply` 节点)

* **电池 Shepherd 电化学放电曲线与内阻动态负载压降 (`battery_level`, `battery_sag`, `load_coupling`)**：

  * 实时汇总全链路有源设备负载电流 $I_{\text{load}}(t, y)$（相机 CCD+DSP 基准 + NightShot 红外灯 `nv_illuminator` + W–T 电动变焦马达 + 主导轴/磁鼓伺服 + 快进倍速 + 粘带尖峰 + 走带状态机瞬态 + 皮带打滑/压带轮偏心扭矩 + 切换台继电器/交叉点负载 + CRT 逐行阴极束流 $Y(y)$），计算公共母线电压 $V_{\text{bus}}(t, y) = V_{\text{oc}}(\text{SoC}) - I_{\text{load}}(t, y) \cdot R_{\text{int}}$，引发成像曝光下坠、底噪爬升、走带掉速与伴音音高同步下坠、同步分离比较器门限漂移及显像管画幅收缩。

* **整流滤波电容干涸纹波 (`psu_ripple`)、冷机上电冲击 (`startup_surge`) 与电机/行包电磁串扰 (`emi_crosstalk`)**。

### 3. 扩展单元 A（对应规划模块 1 扩展）：虚拟相机重拍 CRT 屏摄系统 (`atsim/codecs/crt_reshoot.py` & `display` 层 / `offscreen_camera` 节点)

* **快门-场扫描差拍滚带与 CMOS 卷帘快门 (`reshoot_enabled`, `reshoot_shutter_hz`, `reshoot_fps_offset`, `reshoot_rolling_shutter`)**：

  * 将视频信号先送进虚拟 CRT 显像管激发 P22 荧光粉，再用具备独立快门曝光窗口 $[t_{\text{open}}(y), t_{\text{open}}(y) + \tau_{\text{shutter}}]$ 与逐行卷帘斜率的虚拟 CMOS/CCD 相机重新拍摄；

  * 精确积分电子束在单次快门内的扫过区间 $D = f_{\text{CRT}} / f_{\text{shutter}}$ 与三通道差异化 P22 荧光粉指数衰减尾迹（红粉 $\text{Y}_2\text{O}_2\text{S:Eu}$ 余辉慢于绿/蓝），还原高速快门下的暗谷滚带、电子束前缘亮锋、以及慢快门（如 $1/50\text{s}$ 拍 $59.94\text{Hz}$）下的双扫亮横带，并以差拍频差 $\Delta f$ 跨帧平滑滚动。

* **传感器拜耳像素阵列与球面 CRT 荫罩/扫描线二维空间混叠摩尔纹 (`reshoot_moire`, `reshoot_tilt_deg`)**。

* **离轴透视、房间环境光荧光粉漫反冲淡与 9300K 色温过曝 (`reshoot_perspective`, `reshoot_ambient_glare`, `reshoot_wb_shift`)**。

### 4. 扩展单元 B（对应规划模块 4）：机械传动与伺服状态机系统 (`atsim/codecs/transport_servo.py` & `tape` 层 / `transport_servo` 节点)

* **供带盘/收带盘阿基米德螺旋半径与带张力连续演化 (`reel_progress`, `tension_ripple`)**：

  * 建模标准 VHS 磁带盒（盘芯半径 $R_{\min} = 13.0\text{ mm}$，满盘半径 $R_{\max} = 41.0\text{ mm}$）随带长进度 $p \in [0, 1]$ 的连续卷径演化 $R_s(p) = \sqrt{R_{\max}^2 - p(R_{\max}^2 - R_{\min}^2)}$、$R_t(p) = \sqrt{R_{\min}^2 + p(R_{\max}^2 - R_{\min}^2)}$ 与盘片转速 $f_s(p), f_t(p)$（从满盘 `0.13 Hz` 升至空盘 `0.41 Hz`）；

  * 精确还原带头 (`p=0.0`, BOT) 满供带盘转动惯量与小收带盘芯强拉拽下的顶部二面角张力弯钩畸变 (Upper-Raster Tension Skew)，以及带尾 (`p=1.0`, EOT) 高频供带扭矩脉动、螺旋磁迹弹性拉伸与底边导向轮法兰边缘卷边 (Bottom Edge Curl)。

* **走带操作状态机 (`transport_state`, `servo_lock_tau`, `pause_noise_pos`)**：

  * `0 = Play`（正常走带锁相）、`1 = Pause / Still`（静态暂停定格：磁鼓以 $5^\circ 56^\prime$ 静止扫描角斜跨相邻磁迹保护带，在 `pause_noise_pos` 处形成固定双峰 FM 限幅雪花横带、H-AFC 切口与场间微颤，伴音自动静音）、`2 = Play <-> Pause Transient`（压带轮啮合起停与 CTL 控制脉冲二阶欠阻尼重锁定瞬态）、`3 = Record-Pause Assembly Cut`（无飞抹磁头暂停起录点：未抹除旧螺旋磁迹斜向重影叠印 + $180^\circ$ 色度副载波彩虹差拍干涉条带 + CTL 脉冲相位跳变）、`4 = Stop / Unthread`（停机卸带：导带柱退回与磁鼓包角坍塌雪花帷幕）。

* **橡胶皮带老化打滑、硬化压带轮偏心与伺服猎振 (`belt_slip`, `roller_eccentricity`, `servo_hunt`)**。

### 5. 扩展单元 C（对应规划模块 6）：多路模拟视频混合器与切换台系统 (`atsim/codecs/switcher.py` & `connector` 层 / `video_switcher` 双输入节点)

* **非同步信号源硬切瞬态 (`switcher_mode = 1`, `switcher_hot_cut_rate`, `switcher_genlock_err`)**：

  * 两路未经过帧同步器 (Non-Genlocked) 的视频源在切换台交叉点硬切瞬间，行同步相位阶跃 $\Delta \tau_H$、场计数差 $\Delta Y_{\text{field}}$ 与色同步副载波相位差 $\Delta \phi_{\text{sc}}$ 引发下游录像机/电视机 H-AFC 欠阻尼水平弯钩撕裂、V-Hold 场失锁跳动（露出含开槽脉冲与 VITC 的物理 VBI 消隐带）、APC 色度相位冲击旋转与伴音交叉点直流偏置咔嗒爆音 (Crosspoint Pop/Click)。

* **模拟 T-Bar 叠化 (`switcher_mode = 2`) 与模拟斜坡发生器划像 (`switcher_mode = 3`)**：

  * 还原无后肩重钳位混合母线在 50/50 叠化时的 7.5 IRE 双基座黑电平抬升 (`switcher_pedestal_add`) 与非相干副载波干涉差拍网纹；

  * 模拟行/场锯齿波与抛物波电压比较器生成 5 种经典划像图形（水平、垂直、45°对角、中心菱形光圈、中心圆形光圈），复现比较器行触发抖动 (`switcher_key_jitter`) 与 3.58/4.43 MHz 切换瞬态彩色镶边 (`switcher_subcarrier_spill`)。

* **模拟亮度键控 (`switcher_mode = 4`) 与色度键控 (`switcher_mode = 5`)**：

  * 精确建模 0.6 MHz 色度窄带检测器相对宽带亮度的 `~700 ns` 群延迟右移错位（前景左侧露暗黑/蓝幕切边、右侧溢亮边），以及复合信号残留副载波调制比较器门限形成的 4 像素周期彩色拉链锯齿边 (Subcarrier Zipper)。

* **节点图双复合输入支持 (`in_a` + `in_b`)**：

  * 新增支持 `in_a`（主信号 Program）与 `in_b`（副信号 Preset/Key）双复合输入端口的 `video_switcher` 设备节点，并提供双摄像机/编码分支汇入切换台的 `vhs_analog_switcher_lab` 节点图预设。

### 6. 扩展单元 D（对应规划模块 C）：WebUI 内置实时示波与校准面板 (`atsim/codecs/scopes.py`, `atsim/web/static/scopes.js` & `/api/scopes`)

* **广播级亮度/复合波形监视器 (Waveform Monitor / IRE Scope, `compute_waveform_monitor`)**：

  * 实时绘制当前帧及任意选中扫描行 (`line_y`) 的 `-45 IRE .. +115 IRE` 复合/纯亮度/RGB 分量并列 (Parade) 波形；

  * 左侧 16% 区域完整合成并测量 **水平消隐区 (HBI)**：前肩 (`0 IRE`)、`-40 IRE` 行同步顶（随 `sync_clip` 与电池内阻压降动态衰减）、后肩 (`0 IRE`) 与 9 周期 `±20 IRE` (`40 IRE p-p`) 色同步脉冲串 (Colour Burst)；右侧 84% 区域显示有效视频行（NTSC `7.5 IRE` / PAL `0 IRE` 黑电平基座、切换台/磁带基座偏移及 `100 IRE` 峰值白削波）。

* **色度矢量示波器 (Vectorscope, `compute_vectorscope`)**：

  * 在极坐标盘上绘制 `(U, V)` / `(I, Q)` 二维荧光密度轨迹、75% 与 100% 标准彩条六色靶框 (`R 103.5°, Mg 60.7°, B 347.1°, Cy 283.5°, G 240.7°, Yl 167.1°`) 以及实时色同步矢量 (`Burst`：NTSC `180°` 基准 / PAL `±135°` 逐行摆动双矢量)；

  * 直观测量并显示 `burst_phase_error` 色相旋转、ACC `chroma_gain` 幅度缩放、消色器门限状态及 PAL 逐行相位差引发的汉诺威色条分裂角 (`hanover_split_deg`)。

* **节点级 / 链路级 A/B 差值热力图与水平功率谱分析 (`compute_ab_diff_and_spectrum`)**：

  * 支持在固定链路（任意阶段 `camera/format/tape/player/display` vs `source`）与节点图（任意两节点 `node_a` vs `node_b`）之间实时计算 PSNR (dB)、亮度误差 $\Delta Y$ (IRE)、色度误差 $\Delta C$ (IRE) 与空间热力图；

  * 实时计算 `0.0 .. 6.5 MHz` 水平 1D FFT 功率谱密度 (PSD)，直观标示 VHS 亮度带宽滚降点 (`luma_bandwidth`, 如 `3.0 MHz` 处衰减 dB) 与 `3.58 MHz` (NTSC) / `4.43 MHz` (PAL) 色度副载波能量尖峰。

### 7. 实拍物理模型专项校准：SONY DCR-TRV310E 非均匀噪波/高光溢出与香港早年 VHS 广告底部磁头形态 (`atsim/codecs/lens.py`, `atsim/stages/camera.py`, `atsim/stages/tape.py`, `atsim/core/profiles.py`)

* **SONY DCR-TRV310E 非均匀空间噪波包络模型 (`CameraStage._signal_noise_envelope`, `CameraStage._gain_noise`)**：

  * **死黑区光学黑行钳位与 DV 8×8 DCT 零游程死区 (`L < 0.03`)**：实拍夜空死黑区域因 CCD 光学黑行 (Optical Black) 钳位与 DV25 8×8 DCT 量化零游程死区，噪声被压制至地板（实测 `Y_std = 0.0016`，仿真 `sky_std = 0.00101`）；

  * **暗中调与光晕裙边散粒+AGC峰值区 (`L ∈ [0.08, 0.85]`)**：在树冠、路面、建筑立面及高光灯晕裙边区域，光子散粒噪声（$\propto \sqrt{L}$）与高增益 AGC 叠加达到峰值（实测 `Y_std = 0.0086 .. 0.0189`，仿真 `mid_std = 0.03149`、为死黑夜空的 `31.2x`），具备 **CDS 相关双采样横向条纹各向异性**（水平相关长度约为垂直方向的 `2.7x`）、**8×8 DV25 宏块边界起伏调制**以及 **4:1:1 互补色 CFA (`Cy/Ye/Mg/Gr`) 差分色斑**（品红-绿 / 蓝-黄低频彩斑）；

  * **过曝白核满阱饱和削波归零 (`L > 0.90`)**：高光核心进入光电二极管满阱饱和与 ADC 硬削波，噪波归零（仿真 `core_std = 0.00094`，光晕区噪波为白核的 `39.3x`）。

* **SONY DCR-TRV310E 非均匀高光溢出与光条形态模型 (`asymmetric_ccd_smear_and_trench`, `CameraStage._halation`, `light_streaks`)**：

  * **非均匀六瓣扇贝星芒光晕与水滴菱形核心 (6-Petal Scalloped Iris Halo & Diamond Droplet)**：建模 6 片自动光圈叶片衍射与镜组像散，形成非圆对称的 **6 瓣扇贝状尖角光晕**（主瓣位于 `36°、108°、180°、216°、272°、0°`，伴随 `~245°` 与 `~320°` 深衍射缺口，环形峰值/中位比达 `5.14x`），中远距小光源呈现上下尖锐的水滴/钻石菱形核心并带内圈青蓝、外圈品红紫边；

  * **上下强非对称隔行 CCD 垂直拖尾光柱 (Up-Down Asymmetric Interline CCD Vertical Smear)**：向下侧（`dy > 0`，电荷读出移位方向）形成核心下方漏斗状膨大颈部（Core-Neck Funnel），随即收束为 **1–2.5 px 极细锐利青蓝/亮白针状光柱**（`B > G > R`，带 2 行周期的奇偶场隔行串珠纹理）直插画面底边；向上侧（`dy < 0`）仅存微弱渐隐细丝（实测与仿真下/上能量比均 `> 8.0x`）；

  * **光柱两侧非对称反溢出漏极暗沟 (Flanking Dark VOD/HOD Anti-Blooming Trenches)**：在下半幅垂直亮光柱左右紧邻处（`dx = ±2..5 px`，且右侧 `dx > 0` 更深），反溢出漏极抽走邻列基底电荷，在亮光柱旁刻出低于背景路面的清晰黑沟（实测右侧压暗 `-0.08 ~ -0.18`，仿真 `right_trench_drop = -0.1380`）。

* **早年香港 VHS 广告底部磁头切换带物理修正 (`TapeStage._head_switch`)**：

  * **保留底部真实画面内容 (`luma_ratio = 1.014`)**：废除旧版将底部磁头切换带替换为均匀暗灰条（`level = 0.030`）的做法，完整保留底部正常扫描画面与台标/字幕内容；

  * **行中途切换点阶跃 (`sw_x_row`)**：建模磁鼓换向落在扫描行中途 `x = sw_x_row(y)`（随场间包角微漂移并以斜率跨行推进），切换点左侧（`x < sw_x_row`）保持上一颗磁头的稳定图像，切换点右侧（`x > sw_x_row`）发生瞬态水平相位跳变（仿真 `mid_step: L=0.0022 -> R=0.0773`）；

  * **切换瞬态双极性白/黑亮刺与 1H 失落补偿 (`seam_spike = 0.314`)**：在切换分界行与 `sw_x_row` 交界处注入旋转变压器换向瞬态引发的 FM 限幅器双极性白/黑短促亮刺（White/Black Switching Dash）及局部 1H 延迟线前一行保持；

  * **自切换缝向底边的 H-AFC 拉入衰减与隔行梳状锯齿**：水平位移在切换交界行（`bot-2` 至 `bot-4`）最大（`7.92 px`），向下随行 AFC 锁相环拉入呈指数回落（底行 `4.45 px`），且奇偶场磁头方位角与包角不对称形成 2 行周期的隔行梳状水平锯齿；

  * **PAL 629 kHz 欠载色度 1H 延迟线相位跃变 (`dRGB = (-0.039, +0.027, -0.000)`)**：复现换向后 1H 梳状滤波器将换向前后两行不同相位色度混叠引发的黄绿/橄榄色偏与局部去饱和，完美对齐香港早年 VHS 广告实拍（实测 `ΔR = -0.022, ΔG = +0.002, ΔB = -0.011`）。

### 8. 扩展单元 E：广播级配线架、分配放大器与 75Ω 阻抗负载系统 (`atsim/codecs/patchbay_da.py` & `connector` 层 / `dist_amp` + `patch_router` 节点)

* **75Ω 源内阻分压、终端反射系数与电缆趋肤效应损耗 (`da_enabled`, `da_termination_mode`, `da_load_count`, `da_cable_len_m`, `da_gain_db`, `da_eq_boost`)**：
  * **标准 75Ω 单端接匹配 (`mode = 0`)**：负载阻抗 $Z_L = 75\,\Omega$，电压传递比 $H_0 = 1.0\ (0.00\text{ dB})$，反射系数 $\Gamma = 0$，同步顶精确保持 `-40 IRE`；
  * **未端接高阻开路环通 (`mode = 1`, High-Z Loop-Through)**：$Z_L \approx 10\text{ k}\Omega$，电压传递比跃升至 $1.985\times\ (+5.96\text{ dB})$，末端反射系数 $\Gamma = +0.985$，引发大面积高光 ADC 硬削波（实测 `clip = 48.1%`）与往返延迟 $2\tau_{\text{cable}}$ 的同相重影；
  * **违规无源三通双端接/多并接 (`mode = 2`, Passive BNC T-Split)**：$N$ 台 $75\,\Omega$ 设备直接并接导致 $Z_L = 75/N\,\Omega$（3 路并接时 $Z_L = 25\,\Omega$，幅度暴跌 $-6.02\text{ dB}$），负反射系数 $\Gamma = -0.500$ 产生反相暗边缘重影，且行同步脉冲从 `-40 IRE` 萎缩至 `-20.0 IRE`，直接触发下游录像机/电视机行 AFC 失锁撕裂；
  * **有源视频分配放大器隔离驱动 (`mode = 3`, Active 1-in-N-out Video DA)**：高阻缓冲输入 + 独立 $75\,\Omega$ 背面端接输出级，无论扇出多少路分支均维持 `0.00 dB` 恒定电平，并支持 Belden 8281 同轴电缆 $\sqrt{f}$ 趋肤效应高频均衡补偿 (`da_eq_boost`)。
* **跳线配线架邻道电容串扰、弹簧触点氧化打火与信号误插故障 (`patch_crosstalk`, `patch_contact_aging`, `patch_format_mismatch`)**：
  * 模拟配线架相邻插孔 $1.5\text{–}4\text{ pF}$ 寄生电容的高通微分串扰（60 IRE/ns 邻路伴音与斜纹）、触点氧化微电弧咔嗒声 (`aud_diff = 0.0518`) 以及 S-Video Y 误插复合口（纯黑白）、YPbPr 绿通道误插复合口（绿/品红偏色）与 SDI 数字流误插模拟口（满屏高频数字雪花杂波）。

### 9. 扩展单元 F：演播室多画面监视墙与独立 CRT 个体老化阵列 (`atsim/codecs/multiview_wall.py` & `display` 层 / `multiview_wall` 四输入节点)

* **多源同屏监视墙拓扑 (`mv_enabled`, `mv_layout`, `mv_tally_umd`)**：
  * 支持 **`2×2` 导播室四分屏监视墙 (`mv_layout = 0`)**（同屏对比 `CH1 CAM/SRC`、`CH2 VTR-RAW (TBC Off)`、`CH3 TBC-OUT`、`CH4 PGM-AIR`）、**`3×2` 总控机房 (MCR) 六画面监视桥 (`mv_layout = 1`)** 与 **`1+3` 导播主副非对称监视墙 (`mv_layout = 2`)**；
  * 配备演播室金属机架隔断、顶部硬件 **Tally 指示灯**（`PGM` 红灯、`PVW` 绿灯、`AUX` 琥珀灯、`FAULT` 红色闪烁告警灯）以及底部 `5×7` LED 点阵动态源名标牌 (**UMD**)。
* **监视器个体 CRT 老化性格与信号解耦原则 (`mv_unit_variance`, `MonitorCellProfile`)**：
  * 同一面监视墙上的每台 CRT 拥有独立的物理性格：1 号机为 **SONY PVM-14L5 D65 柱面特丽珑基准管**，2 号机为 **JVC TM-H150CG 欠扫描 (Underscan) 机房录像监看管**（完整露出四边消隐框与底部磁头切换带），3 号机为 **池上 (Ikegami) 阴极老化绿偏管**，4 号机为 **9300K 家用槽孔荫罩彩电接收机**，6 号机（`3×2` 模式）为 **基准源 银白荧光粉单色同步检查管**；
  * **关键工程原则**：每台监视器的阴极色温漂移、伽马、会聚误差与枕形畸变仅作用于该监视器窗口的观测画面，底层传输的视频信号本身保持 `100%` 纯净未污染 (`feed_untouched = True`)。
* **各通道无信号 / 失锁的真实硬件物理响应 (`mv_fault_channel`, `mv_fault_mode`)**：
  * 当指定通道断流或上游接口掉锁时，按真实监视器/录像机硬件机制呈现 5 种物理状态：`1 = 纯黑无同步底噪 (Black No-Sync)`、`2 = RF 高频头 AGC 满增益白雪花 (RF Snow)`、`3 = 场行自由振荡斜向滚屏消隐带 (Free-Run Roll)`、`4 = 数字 TBC 末帧死锁残影冻结 (Frame Freeze)`、`5 = JVC/松下录像机本机蓝屏 + NO SIGNAL 字符替换 (VTR Blue OSD)`。

### 10. 扩展单元 G：广播级测试信号发生器与工程闭环校准台 (`atsim/codecs/test_signal_gen.py` & `format` 层 / `test_generator` 节点 & `scopes.py`)

* **5 种标准广播工程测试图卡与校准音频合成 (`test_gen_enabled`, `test_pattern_mode`, `test_tone_mode`, `test_amplitude_ire`)**：
  * **`mode = 1` (SMPTE RP-219 / EIA RS-189-A 彩条 + PLUGE)**：上部 2/3 为 75% 标准彩条，中部为反向蓝条（用于蓝枪滤光片校准色度增益与色相），下部含 `-I`、`100% 峰值白 (100 IRE)`、`+Q` 及 **PLUGE 黑电平三阶校准脉冲 (`-4 IRE / 0 IRE / +4 IRE`)**；
  * **`mode = 2` (EBU Tech 3087 PAL 100/0/75/0 彩条 + 线性亮度斜坡)**；
  * **`mode = 3` (FCC / CCIR 多波群频响测试卡 Multiburst)**：包含 `70 IRE` 白基准脉冲与 `0.5, 1.0, 2.0, 3.0, 4.2, 5.0 MHz` 六组相位连续正弦波群包络，直接呈现模拟滤波器、同轴线缆与 VHS 磁带的高频滚降；
  * **`mode = 4` (5 阶调制阶梯 + `2T` 正弦平方脉冲 + `12.5T` 色亮延迟脉冲)**：0–100 IRE 五阶亮度阶梯叠加恒定 `40 IRE p-p` 副载波（测量微分增益 DG 与微分相位 DP），下半幅合成半高宽 `2T (250 ns)` 冲激脉冲（测量 K 因子振铃）与 `12.5T` 调制脉冲（测量 Y/C 群延迟错位）；
  * **`mode = 5` (Philips PM5544 / 电视台圆环几何与中心楔形分辨率卡)**：集成圆环线性度网格、中心 `0.8–4.8 MHz` 连续扫频光栅、四角会聚十字以及随帧相位旋转的动态场扫描秒针。
* **工程闭环校准分析仪 (`analyze_engineering_calibration` & `/api/scopes` 实时遥测)**：
  * 自动解算 `0.5–5.0 MHz` 各频点多波群包络衰减 (`multiburst_db`) 与实测 `-3 dB` 截止带宽 (`measured_bw_3db_mhz`)、微分增益 (`diff_gain_pct`, %)、微分相位 (`diff_phase_deg`, °)、`2T` K 因子 (`k_2t_pct`, %)、色亮群延迟 (`yc_delay_ns`, ns)、PLUGE 黑位截断/抬升状态，以及 **TBC 旁路 vs 开启的水平时基抖动抑制比 (`tbc_comparison.suppression_db`, 实测 `+14.01 dB`)**。

### 11. 扩展单元 H：演播室摄像机控制单元 (CCU) 与主同步发生器 (Blackburst SPG / Genlock) (`atsim/codecs/studio_ccu_genlock.py` & `camera` 层 / `ccu_rack` 节点)

* **多机位个体色彩特征与集中式 CCU 机架遥控配平 (`ccu_enabled`, `ccu_cam_id`, `ccu_iris_ev`, `ccu_master_pedestal_ire`, `ccu_r_gain`, `ccu_b_gain`, `ccu_r_flare`, `ccu_b_flare`, `ccu_knee_point`, `ccu_knee_slope`)**：
  * 建模同场演播室照明下 4 台异构摄像机的基线个体差异：`0 = CAM-1 Sony BVP-370 (D65 基准座机)`、`1 = CAM-2 Ikegami HL-55 (偏暖高反差采访机)`、`2 = CAM-3 BTS LDK-90 (偏冷暗部微青全景机)`、`3 = CAM-4 老旧游机 (阴极/CCD 暗电流抬升 + 绿偏)`；
  * 视频工程师通过 RCP 远程面板集中推拉光圈 (`ccu_iris_ev`)、总黑电平 (`ccu_master_pedestal_ire`)、红/蓝白平衡增益 (`ccu_r_gain`, `ccu_b_gain`)、红/蓝黑平衡基座 (`ccu_r_flare`, `ccu_b_flare`) 以及高光动态压缩膝点 (`ccu_knee_point`, `ccu_knee_slope`)，可将异构机位的色彩误差压缩 **78% 以上**（实测 `L1 = 0.0436 -> 0.0095`，`R/G = 1.149 -> 0.984`）。
* **演播室主同步发生器 (Blackburst SPG / House Genlock) 与脐带线缆相移补偿 (`genlock_mode`, `genlock_cable_len_m`, `genlock_sc_phase_deg`, `genlock_h_phase_px`)**：
  * 精确建模同轴/三同轴脐带线传播速度 $v_p = 0.66c \approx 1.98\times 10^8\text{ m/s}$ 带来的 $5.05\text{ ns/m}$ 延时：在 NTSC $3.58\text{ MHz}$ 下每米产生 $6.51^\circ/\text{m}$ 的副载波相移（20 米线缆未补偿时产生 `101.0 ns` 延迟与 `+130.15°` 色相旋转），通过 CCU 面板 `genlock_sc_phase_deg` 与 `genlock_h_phase_px` 可精确调零（实测误差归零至 `0.0°`）；
  * 支持外同步断线自由晶振漂移 (`genlock_mode = 2`：暴露含开槽脉冲与 VITC 时间码的物理 VBI 滚屏带及连续色相翻滚) 与 BNC 接触不良间歇掉锁冲击 (`genlock_mode = 3`)。

### 12. 扩展单元 I：主控播出自动化 (Playout)、下游键控器 (DSK) 与双总线预监 / 干净输出 (`atsim/codecs/master_playout.py` & `overlay` 层 / `master_playout` 多输出节点)

* **PVW / PGM 双总线、Clean Feed（干净净画面母线）与下游键控器 (`playout_enabled`, `playout_output_bus`, `playout_dsk_enabled`, `playout_dsk_crawl`)**：
  * 同时维护并输出三路独立广播母线：
    1. **主播出母线 (`PGM`, `bus = 0` / `out` 端口)**：叠加下游键控器 (**DSK**) 的三色电视台角标 (`HK-TV1`)、实时帧级广播时钟 (`HH:MM:SS:FF`) 与底部横向滚动新闻/通告字幕条；
    2. **干净净画面母线 (`Clean Feed`, `bus = 1` / `clean` 端口)**：包含完整的切换台过渡、调色与应急垫片状态，但 **100% 绕过 DSK 台标与字幕**（实测 `clean_untouched = True`，`0.0%` DSK 污染），专供异地转播与母带无字归档；
    3. **预监母线 (`PVW`, `bus = 2` / `pvw` 端口)**：显示下一条待播信号源、安全框准星与录像机预卷就绪状态；
    4. **主控双总线导播控制台 (`MCR Dual-Bus Desk`, `bus = 3`)**：同屏展示左上 `PVW (绿 Tally)`、右上 `PGM (红 Tally)`、左下 `Clean Feed` 净画面监看窗与右下实时 **广播级 As-Run 运行日志表**。
* **主控播出单状态机、5 秒录像带预卷倒计时与断流自动应急垫片接管 (`playout_state`, `playout_failover_auto`)**：
  * 支持 `0 = 正常节目在播 (ON-AIR)`、`1 = 录像带 5秒预卷伺服倒数 (VTR Preroll -00:05..-00:01 -> Take)`、`2 = 应急「PLEASE STAND BY / 请稍候」备用彩条垫片` 与 `3 = 节目单自动顺播循环`；
  * 当上游信号断流（`no_signal` 或死黑故障）且 `playout_failover_auto = 1` 时，总控应急矩阵自动切入「PLEASE STAND BY」备用垫片，同时保持 DSK 台标与抢修通告正常在播，并自动向 `as_run_log` 写入 `FAILOVER` 广播事故记录。


### 13. 扩展单元 J：多代转录与跨格式复制对照工作站 (`atsim/codecs/transfer_workstation.py` & `player` 层 / `transfer_station` 节点)

* **三类转录/复制工作站拓扑与四路对照实验室 (`transfer_station_mode`, `transfer_generations`, `transfer_tbc_mode`, `transfer_codec_target`, `transfer_adc_crop`, `transfer_av_drift_ms`)**：
  * **`mode = 1` (A. 模拟磁带数字化采集站 Ingest)**：建模 `VHS/Video8/Betacam → TBC → BT.601 8-bit 采集卡 ADC`；**严格保留上游母带已有缺陷**（失落白道与磁头切换带不会被采集卡凭空抹除，实测 `defect_preserved = True`），同时建模无 TBC 时采集卡 PLL 行抖动滑移 (`0.459 px`) vs 全帧同步 TBC (`0.028 px`)、BT.601 `16..235` 量化范围截断与左右消隐 (HBI) 抗混叠振铃边，以及独立声卡晶振音画同步漂移 (`+54.0 ms → +4.0 ms`)；
  * **`mode = 2` (B. 磁带多代复制站 Multi-Gen Dubbing)**：建模 `1..8` 代连续磁带复制，精确区分 **可被 TBC 校正的水平时基抖动**（无 TBC 时 1→4 代抖动从 `0.44 px` 发散至 `1.43 px`，开启全帧 TBC 后第 4 代抖动仍抑制在 `0.03 px`，抑制比达 `34.8x`）与 **不可逆累积的代际退化**（即使开启 TBC，亮度 `-3 dB` 带宽仍从 `3.20 MHz` 滚降至 `1.69 MHz`，PSNR 从 `21.6 dB` 跌至 `13.1 dB`，伴随 629 kHz 色度群延迟右移、恒 K 滤波器过冲振铃与磁饱和 S 曲线压缩）；
  * **`mode = 3` (C. 数字格式转码站 Digital Transcode)**：建模 `BT.601 4:2:2 → DV25 (4:1:1) → MPEG-1 VCD (4:2:0)` 跨代重编码；数字采样网格令时基抖动恒为 `0.00 px`，但 `4:1:1`（横向 1/4）与 `4:2:0`（2×2）级联造成 `1/8` 色度空间分辨率交集台阶，且多代 8×8 DCT 重编码累积块边界伪影 (`dct_err = 0.0041`)；
  * **`mode = 4` (四路复制链对照实验室 2×2 Quad Lab)**：同屏实时对比 `Gen-0 母带基准`、`路径 A (无TBC多代模拟复制)`、`路径 B (带TBC多代模拟复制)` 与 `路径 C (4:1:1→4:2:0 数字转码)`，并输出完整的 `TransferEvent` 代际溯源链。

### 14. 扩展单元 K：线性磁带编辑控制器与早期非线性编辑系统 (`atsim/codecs/edit_controller_nle.py` & `tape` 层 / `linear_editor` + `nle_timeline` 节点)

* **Sony RM-450 / BVE-9100 双机磁带线性编辑控制器 (`edit_mode = 1, 2`, `edit_point_slip_frames`, `edit_flying_erase`, `edit_preroll_sec`)**：
  * **双机预卷与 CTL 入出点机械滑移 (`edit_point_slip_frames`)**：建模源录像机 (Player VTR) 与录制机 (Recorder VTR) 压带轮啮合与 29.97/25Hz CTL 脉冲计数的 `±1..4 帧` 寻址偏移（在理想切点帧 `frame=2` 处，`slip=0` 已切入 `SHOT_B_INCOMING` 而 `slip=2` 仍停留在 `SHOT_A_OUTGOING`，实测 `L1 = 0.313`）；
  * **组合编辑 (`Assemble Edit`, `edit_mode = 1`)**：入点处全宽固定抹音头、旋转视频磁头与 CTL 磁头同时切入记录，产生顶部抹音黑场闪黑 (`assemble_flash_active = True`) 与 H-AFC 相位弯钩；出点处留下标志性的新旧控制轨断裂与 FM 雪花撕裂切口 (`tail_ctl_break_active = True`)；
  * **插入编辑 (`Insert Edit`, `edit_mode = 2`)**：保留录制带原有连续 CTL 控制轨，出点无控制轨断裂 (`tail_ctl_break_active = False`)；当旋转飞抹磁头关闭或老化 (`edit_flying_erase = 0`) 时，新磁迹叠写在未抹净的旧斜向磁迹上，产生斜向重影与 180° 色度副载波彩虹差拍 (`feh_L1 = 0.252`)。
* **2000 年代初早期非线性编辑工作站 (`Premiere 6 / Canopus DV Storm NLE`, `edit_mode = 3, 4`, `edit_nle_complexity`)**：
  * 共享非破坏性 `EditSession` / `EditDecision` 多轨时间线（`V1`, `V2`, `TITLE_1`, `A1/A2`），文件索引寻址滑移恒为 `0 帧`；
  * **实时预览 (`edit_mode = 3`) vs 最终渲染导出 (`edit_mode = 4`)**：在多轨转场与字幕负载下，实时预览触发 `1/4` 半分辨率代理降级与解码缓冲区欠载掉帧重复 (`preview_dropped_frame = True`)，而最终渲染导出保持全分辨率且零掉帧 (`preview_dropped_frame = False`, 预览与导出画面差异 `L1 = 0.140`)，同时对转场合成区间执行二次 DV25 8×8 DCT 重编码 (`reencode_dct_l1 = 0.0144`)。

### 15. 扩展单元 L：事件驱动与条件驱动的播出自动化编排引擎 (`atsim/codecs/playout_automation_v2.py` & `overlay` 层 / `automation_controller` 节点)

* **完整 `PlayoutEvent` 五态生命周期与 6 类分级广播异常分支 (`auto_v2_enabled`, `auto_v2_mode`, `auto_v2_fault_inject`, `auto_v2_watchdog_sec`, `auto_v2_console_view`)**：
  * **五态正常流转**：`SCHEDULED（待播）→ PREPARING（切点前自动提前 5 秒向目标录像机发送 VTR Preroll 预卷与伺服锁相指令）→ READY（伺服锁定就绪）→ ON_AIR（主母线在播）→ COMPLETED（顺播完成归档）`；
  * **6 类分级异常分支**：精确建模 `1 = MISSING_ASSET`（磁带未上架/素材离线，自动降级切入备用垫片）、`2 = NOT_READY`（录像机粘带或伺服超时未就绪，延迟切入或垫片兜底）、`3 = SIGNAL_LOSS`（在播主路射频/同步断流，看门狗超时自动切备路）、`4 = OVERRIDDEN`（导播手动紧急强切 `Emergency Take` 接管自动化母线）、`5 = FALLBACK`（主路触发黑场/静帧告警，自动无缝倒换至 `BACKUP` 备路）、`6 = HELD`（直播延时挂起等待导播确认释放）；
  * **三级故障因果反查树 (`CausalFaultRecord`)**：将底层物理故障（如 `tape.stiction` 粘带、`rf.sat_rain_fade_db` 卫星雨衰断流）映射为 `【物理根因 Root Cause】→【节点传播路径 Propagation Chain】→【主控现象与动作 Symptom & Action】` 的结构化反查记录，并提供 `2×2` 播出自动化编排控制台同屏监看。

### 16. 扩展单元 M：卫星接收机 (Satellite IRD) 与模拟/数字悬崖效应系统 (`atsim/codecs/satellite_ird.py` & `rf` 层 / `sat_dish_lnb` + `sat_ird_receiver` 节点)

* **六级卫星接收物理链路 (`sat_ird_mode`, `sat_cn_db`, `sat_rain_fade_db`, `sat_dish_mispoint_deg`, `sat_lnb_drift_khz`, `sat_ird_osd`)**：
  * 建模抛物面天线指向损耗 $L_{\text{point}} = 12(\theta/\theta_{3\text{dB}})^2\text{ dB}$、ITU-R P.618 暴雨衰减 (`Rain Fade`) 与春秋季日凌热噪爆发 (`Sun Outage`)、LNB 介质谐振器 (DRO) 本振温漂与相位噪声；
  * **模拟 C 波段 FM 卫星电视分支 (`sat_ird_mode = 1`)**：当有效载噪比跌破 FM 门限（`~11.5 dB`）时，鉴频器发生相位周跳，在画面高饱和/高对比区域爆发标志性的 **鱼骨状一黑一白拖尾脉冲打火噪点 (FM Threshold Sparkles)**（实测 `16.5 dB` 时 `sparkle = 0.00000`，`5.45 dB` 时 `sparkle = 0.9231, L1 = 0.048`），但画面内容呈渐进退化且始终保持可辨识轮廓；
  * **数字 DVB-S QPSK + Viterbi + Reed-Solomon 综合解码分支 (`sat_ird_mode = 2`)**：精确还原 **数字悬崖效应 (Digital Cliff Effect)**——当有效 $C/N \ge 6.5\text{ dB}$（QEF 准无误码门限）时，即便信道存在原始误码（实测 `8.2 dB` 时 `pre_viterbi_ber = 7.41e-03`），经内码 Viterbi 与外码 `RS(204,188)` 纠错后未纠错包恒为 `0`，画面与 `16.5 dB` 晴空基准完全一致（`dClear = 0.00000`，零雪花）；一旦跌入 `4.4–6.5 dB` 悬崖边缘（如 `4.9 dB`），瞬间爆发 MPEG-2 `16×16` 宏块条带崩坏、绿/品红 DCT 块伪影、PTS/DTS 音画失步与爆音静音；低于 `4.4 dB` 直接触发 `CLIFF_SERVICE_LOST` 死锁静帧与 `E48 NO SIGNAL` 告警；
  * **模拟 FM vs 数字 DVB-S 同屏悬崖对照实验室 (`sat_ird_mode = 3`)**：在同一雨衰链路上 `2×2` 同屏实时对比模拟渐进打火噪点、数字悬崖宏块崩坏与 `C/N` 质量响应曲线及 MPEG-TS (`PID 0x0100/0x0101`, `PCR jitter`, `TEI`) 统计表。

### 17. 扩展单元 N：电视发射机激励链与端到端开路广播测试台 (`atsim/codecs/tv_transmitter_chain.py` & `rf` 层 / `tv_exciter_pa` 节点)

* **电视台大功率发射机激励与高功放天馈链 (`tx_chain_mode`, `tx_pa_drive`, `tx_pa_am_pm_deg`, `tx_filter_gd_ns`, `tx_feeder_vswr`, `tx_sound_intermod`)**：
  * **高功放 (HPA) AM-AM 微分增益压缩 (`DG%`) 与 AM-PM 微分相位旋转 (`DP°`)**：模拟电子管/速调管或固态高功放逼近饱和区时的非线性包络压缩与寄生相移（实测驱动从 `0.05` 升至 `0.85` 时，微分增益 `DG` 从 `0.3%` 升至 `22.1%`，微分相位 `DP` 从 `0.2°` 升至 `21.1°`），导致高光色度去饱和与色相偏转；
  * **VSB 残留边带与波导腔体双工器群延迟 (`tx_filter_gd_ns`) 及铁塔长馈线驻波反射 (`tx_feeder_vswr`)**：还原腔体滤波器边缘陡峭群延迟引发的色亮错位与 `2T` 振铃，以及铁塔天馈失配（`VSWR = 1.65` 时反射系数 $\Gamma = 0.245$）在发射端即产生的固定延迟驻波重影；
  * **伴音-色度-图像三载波互调 (`tx_sound_intermod`)**：还原图像、色度与伴音共用功放时的 `920 kHz / 1.57 MHz` 差拍斜纹及亮行侵入伴音鉴频器的 `50/60 Hz` 蜂音 (`Intercarrier Buzz`)；
  * **端到端开路广播 2×2 四象限测试台 (`tx_chain_mode = 3`)**：同屏对比 `1. 主控直通基带参考`、`2. 发射机天线口监测 (TX Probe)`、`3. 模拟开路接收终端` 与 `4. 数字地面 (DTT COFDM) 接收终端`，定量剥离发射机自身失真 (`isolated_tx_l1 = 0.1284`) 与传播信道/接收机失真 (`isolated_channel_l1 = 0.1175`)。



### 18. 扩展单元 O：测量、误差预算、参数反演与三级参数审计系统 (`atsim/codecs/inversion_calibration_lab.py` & `player` 层 / `inversion_analyzer` 节点)

* **三级参数显式分类与分离贡献审计 (`classify_schema_parameters`, `audit_parameter_tiers_and_error_budget`)**：
  * 将全引擎 **478 项参数**严格标记并分离统计为三类：`PHYSICAL`（物理机制参数，106 项）、`DEVICE`（设备型号校准参数，353 项）、`ARTISTIC`（非物理创作调整或超出 `[phys_min, phys_max]` 物理标定包络的偏离，19 项）；
  * 支持对任意预设或工程执行三阶段剥离渲染（`Physical-Only` vs `+Device` vs `+Artistic`），定量解算各层对最终画面的独立误差预算（实测 `L1(phys=0.3326, dev=0.0155, art=0.0155)`）与物理包络合规率 (`physical_compliance_ratio`)。
* **多解置信区间参数反演求解器 (`invert_degraded_observation`)**：
  * **高频滚降反演与多解竞争假设概率排序**：给定退化输出帧，自动反推等效 `-3 dB` 亮度带宽 $\hat{B}_Y$ 及置信区间（实测窄带 `0.80 MHz [0.50..1.18]` vs 宽带 `2.61 MHz`），并根据水平/垂直梯度各向异性比、`2T` 边缘振铃过冲指数与色度底噪，对 `1. format.luma_bandwidth (模拟低通滤波器滚降)`、`2. player.head_wear (磁头间隙磨损)` 与 `3. camera.defocus (光学二维散焦)` 三种竞争性物理机制给出概率分布；
  * **磁带逐行时基抖动 vs 电影片门二维机械抖动严格解耦**：通过逐行水平位移差分标准差与整幅刚性平移解算，精确区分磁带逐行时基误差（`tape_line_jit = 0.837 px`）与电影放映机片门整幅抖动（`film_line_jit = 0.018 px, gate_weave = 0.951 px`）；
  * **FADGI 1★～4★ 数字化质量评价与 `2×2` 反演实验室 (`inv_lab_mode = 3`)**：同屏展示观测画面与反演参数置信区间、残差热力图、竞争假设概率条形图及 FADGI 合规星级评分卡。

### 19. 扩展单元 P：电影胶片（16mm / 35mm）化学成像与电视胶转磁系统 (`atsim/codecs/film_telecine.py` & `camera` 层 / `film_scanner_telecine` 节点)

* **卤化银胶片感光化学与冲洗模型 (`simulate_photochemical_film`)**：
  * **Hurter–Driffield (H&D) $D(\log E)$ 特性曲线与三层减色法耦合染料**：内置 4 种经典胶片型号（`1 = Eastman Color 35mm 5247 负片+印片`、`2 = Kodak Ektachrome 16mm 7240 新闻反转片`、`3 = Fujicolor 16mm 8514 纪实胶片`、`4 = 早期醋酸片基暖棕褪色档案拷贝`），还原软趾部、线性反差区与高光软肩部密度饱和；
  * **防光晕底层红橙背散射 (`film_halation`) 与中密度曝光依赖颗粒团簇 (`film_grain_amount`)**：还原强光穿透乳剂层在片基背面反射回底层感红层形成的红橙色光晕环，以及在中密度区达到峰值、在死黑趾部与过曝肩部自然收敛的卤化银/染料云空间团簇（实测 `mid_std = 0.0350 > toe_std = 0.0068, shoulder_std = 0.0053`）。
* **胶片物理老化与机械放映机模型 (`simulate_film_aging_and_projector`)**：
  * **Eastmancolor 青/黄染料非对称褪色 (`film_dye_fade`) 与醋酸酸败收缩 (`film_shrinkage`)**：还原青色与黄色耦合染料先于品红染料水解导致的经典暖红棕/洋红偏色（实测 `dR = +0.036 vs dG = -0.089`）及醋酸纤维素脱乙酰化收缩引起的几何翘曲；
  * **纵向走片划痕、单帧灰尘与片门机械二维抖动 (`film_gate_weave`) + 旋转快门闪烁 (`film_shutter_flicker`)**。
* **Rank Cintel 飞点胶转磁与 `2×2` 跨媒介历史对照实验室 (`apply_film_telecine`, `telecine_pulldown`)**：
  * 支持 `0 = 24fps 逐行直扫`、`1 = NTSC 24fps → 29.97i 3:2 Pulldown`（按 `AA-BB-BC-CD-DD` 五帧周期映射，在 `BC` 与 `CD` 跨胶片帧奇偶场交织帧产生标志性隔行梳状伪影，实测梳状比从 `1.02` 跃升至 `1.45`）、`2 = PAL 24fps → 25i 加速胶转磁`（动作加速 `+4.167%`，伴音同步升调 `+70.67 cents`）与 `3 = 2×2 跨媒介历史对照实验室`（同屏对比 `16mm/35mm 胶片原片 → 电视胶转磁母带 → 家用 VHS 电视录屏 → 数字流媒体重压版`）。


### 20. 扩展单元 Q：档案保存生命周期与媒介考古溯源系统 (`atsim/codecs/archival_archaeology.py` & `player` 层 / `archival_vault` 节点)

* **四阶段视听档案保存生命周期与阿伦尼乌斯加速老化动力学 (`compute_storage_lifecycle_risk`)**：
  * 完整建模 `1. 原始拍摄/播出记录 (Original Recording)` $\to$ `2. 生产与流通磨损 (Circulation)` $\to$ `3. 库房保存与温湿度加速老化 (Arrhenius Hygrothermal Decay)` $\to$ `4. 抢救数字化与再编目 (Rescue & Re-cataloging)`；
  * 基于 IASA-TC 04 与 ISO 18923 化学反应速率方程计算库房温度 $T$ 与相对湿度 $\text{RH}$ 的加速老化因子（实测高温高湿库房 `32°C / 82% RH` 相对低温恒湿库房 `10°C / 35% RH` 的老化加速比达 `38.24x vs 1.31x`，聚氨酯黏合剂水解指数从 `0.370` 跃升至 `0.980`），并联动回放机具与备件消亡指数 (`archival_obsolescence`) 解算机具可用率与数字化访问成本倍增（`3.74x -> 19.47x`）。
* **「保存成功 $\neq$ 信息完整」FADGI / NFSA 重要属性损耗评估 (`audit_significant_properties`)**：
  * 严格区分 **像素存活分 (`pixel_survival_score`)** 与 **媒介考古重要属性完整度 (`significant_property_completeness`)**：
    * **档案保存母版 (`Preservation Master`, `mode = 2`)**：全画幅无损采集，完整保留顶部 VBI / VITC 双相时间码位流、底部磁头切换带、原始隔行场序与介质微纹理（重要属性完整度达 `96.8%`）；
    * **过度清洗商业修复版 (`Over-Cleaned Commercial Restoration`, `mode = 3`)**：虽保留了 `91.2%` 的主体像素，但因激进 3D-DNR 降噪（将胶片/磁粉微纹理抹成塑料蜡像感）、过扫描裁切（切掉顶部 VITC 时间码与底部磁头切换带，`vbi_vitc_std` 从 `0.0564` 跌至 `0.0085`）与去隔行混场，导致重要考古属性完整度暴跌至 `20.6%`（触发 `ARCHAEOLOGICAL_EVIDENCE_DESTROYED` 告警）。
* **多版本并存谱系树与 `2×2` 对照实验室 (`archival_mode = 4`, `ArchivalLineageTree`)**：
  * 同屏对比同一作品的 `1. 原始载体残件`、`2. 档案保存母版`、`3. 过度清洗商业修复版` 与 `4. 流媒体传播版及重要属性损耗表`，并支持 `archival_quad_corner = 1..4` 四角全屏放大、`archival_show_border` 分框切换与 `archival_show_text = 0` 默认无字纯净输出。

### 21. 扩展单元 R：二维空间物理环境场与多维耦合系统 (`atsim/codecs/spatial_environment_field.py` & `connector` 层 / `env_chamber` 节点)

* **统一二维机房/场景空间环境场 (`EnvironmentField`, `evaluate_all_devices`)**：
  * 废除全链路共用单一全局标量的做法，构建连续二维空间场 $\mathcal{E}(x, y, t)$；摄像机 `(env_cam_x, env_cam_y)`、录像机 `(env_vtr_x, env_vtr_y)` 与 CRT 监视器 `(env_crt_x, env_crt_y)` 按各自在机房中的物理坐标 $(x, y) \in [0, 1]^2$ 独立采样局域温度、相对湿度、近场电磁辐射与光照色温（实测冷风口旁摄像机 `(0.15, 0.18)` 局域 `T=21.2°C, RH=70.2%, EMI=0.046`，而右下角靠近电源柜的录像机 `(0.82, 0.78)` 局域 `T=29.3°C, RH=92.8%, EMI=2.000`）。
* **四大空间环境物理耦合子系统 (`apply_spatial_environment_field`, `compute_spatial_acoustic_audio`)**：
  1. **温湿度露点场与冷机抱入暖房瞬态结露**：按 Magnus-Tetens 方程解算局域露点 $T_{\text{dew}}(x,y)$ 与机壳热惯性升温曲线 $T_{\text{dev}}(t)$；当 $T_{\text{dev}}(t) < T_{\text{dew}}(x,y)$ 时，摄像机镜片瞬间结露起雾、录像机磁鼓爆发水膜粘带撕裂与伴音变调，并随设备升温自动消散（实测 `t=0s -> 180s` 机壳从 `5.0°C` 升至 `31.8°C`，结露水膜从 `1.000` 归零至 `0.000`）；
  2. **电磁环境场与长线缆接地环路**：建模大功率电源变压器/发射塔 $1/r^2$ 近场磁偶极子耦合与跨空间坐标线缆接地电位差 $\Delta V_{\text{gnd}}$，驱动工频滚道与电磁斜纹；
  3. **光学空间场**：模拟窗户日光 (`6500K`) 与室内钨丝灯 (`2850K`) 二维混合色温梯度及 CRT 玻壳杂散光冲淡；
  4. **声学空间场**：基于 Sabine 室内混响时间 `env_acoustic_rt60` 合成早期反射梳状滤波、房间驻波共鸣与局域变压器电磁哼声（实测混响尾音 RMS `0.02835`）。


### 22. 扩展单元 S：网络视听传播、平台转码与流媒体协议系统 (`atsim/codecs/network_streaming_era.py` & `codec` 层 / `stream_cdn_client` 节点)

* **早期互联网流媒体与现代自适应流媒体编码 (`stream_enabled`, `stream_mode`, `stream_bitrate_kbps`, `stream_gop_len`, `stream_platform_generations`)**：
  * **1999 拨号/早期宽带流媒体 (`mode = 1`, RealVideo G2 / WMV9)**：模拟 `28.8–128 kbps` 极低码率下的 `160×120` 子采样、强环路去块滤波造成的塑料蜡像平滑感（实测高频能量从 `0.0382` 衰减至 `0.0073`）与帧率抽帧跳动；
  * **Web 2.0 早期视频平台与多代搬运重压 (`mode = 2, 4`)**：模拟 `360p FLV/H.263` 的 `4:2:0` 色度抽样、GOP 帧间预测残差累积与 **I 帧关键帧周期性画质呼吸脉冲 (`keyframe_quality_pulse`: `I=1.000` vs `P=0.301`)**，以及 `1..5` 代跨平台下载再上传的代际失真（实测 `1代 L1=0.0423 -> 5代 L1=0.0952`）与双平台角标水印叠印；
  * **现代 HLS/DASH 自适应流媒体 (ABR) 与分组网络传输 (`mode = 3`)**：实时联动可用带宽波动 (`stream_bw_fluctuation`)、RTT 延迟 (`stream_rtt_ms`) 与分组丢包率 (`stream_packet_loss`)，驱动 ABR 阶梯在 `1080p_HIGH`、`720p_STD`、`480p_MAIN`、`360p_LOW` 与 `240p_EMERGENCY` 之间动态切换，并精确区分 **TCP/HLS 缓冲区耗尽转圈卡顿 (`rebuffering_stall = True`)** 与 **UDP/RTP 实时流水平宏块切片丢失及跨帧品红/绿运动向量污染**；
  * **流媒体伴音低码率子带量化与丢包静音耦合 (`compute_streaming_audio_effect`)** 及 **`2×2` 跨时代网络流媒体对照实验室 (`stream_mode = 5`)**。

### 23. 扩展单元 T：人类视觉感知与机房操作者行为闭环系统 (`atsim/codecs/perception_and_operator.py` & `display` 层 / `human_operator` 节点)

* **观众视觉感知心理物理模型 (`compute_viewer_perception`, `mannos_sakrison_csf`)**：
  * **观看距离视角空间频率转换 ($	ext{cpd}$)**：根据观看距离与屏幕高度比 `percept_viewing_dist_h`（`1.0–10.0 ×H`）解算垂直视角 $	heta_v$ 与每度视角扫描线密度，将 CRT 扫描线、荫罩点阵与磁带噪波映射至视网膜空间频率 $f_{	ext{cpd}}$（实测 `1.3×H` 近看时伪影落入 `4.0 cpd` 人眼峰值敏感带 `CSF = 0.859, sigma_eye = 0.05 px`；而在 `7.5×H` 客厅沙发距离下推高至 `22.0 cpd` 高频滚降区 `CSF = 0.445, sigma_eye = 1.15 px`，扫描线与高频噪点被视觉系统自然融合成柔和画面）；
  * **屏幕绝对亮度 (`percept_display_nits`)、环境照度 (`percept_ambient_lux`) 与 Ferry–Porter 临界闪烁融合频率 (`CFF`)**：定量计算环境光韦伯黑位冲淡与闪烁融合阈值（`160 nits` 下 `CFF = 45.5 Hz` vs `30 nits` 下 `36.5 Hz`），输出 **物理实测误差 (`physical_rmse`)** vs **人眼可察觉差异 (`perceived_jnd_mean`)** 及主观清晰度/舒适度/年代感 MOS 评分。
* **机房操作者认知与操控行为闭环 (`simulate_human_operator_loop`)**：
  * 建模机房值班工程师基于波形监视器与画面告警调节 Tracking 旋钮与 CCU 增益的二阶延迟反馈控制回路：熟练镇定工程师 (`skill = 0.95, stress = 0.10`) 阻尼比达 `ζ = 0.81`，旋钮误差从 `0.515` 平滑收敛至 `0.050`（`SKILLFUL_LOCKED`）；而在高时间压力与疲劳下 (`skill = 0.20, stress = 0.92`)，阻尼比跌至 `ζ = 0.08`，爆发剧烈的 **二阶欠阻尼过调猎振 (`overshoot = 0.777`, `PANIC_OVERSHOOT_HUNTING`)** 乃至偶发切入未就绪母线的误切事故；
  * 提供 **`2×2` 人类视觉感知与机房操作者闭环实验室 (`percept_mode = 4`)**。

### 24. 扩展单元 U：媒介技术生态演化与历史情景模拟器 (`atsim/codecs/media_ecosystem_evolution.py` & `overlay` 层 / `ecosystem_simulator` 节点)

* **1975–2035 七维宏观媒介生态动力学 (`evaluate_ecosystem_at_year`, `simulate_media_ecosystem_trajectory`)**：
  * 联动求解 `1. 模拟回放设备存量 (hardware_stock)`、`2. 库房介质存量 (media_stock)`、`3. 跨代技术兼容性矩阵 (5×5 COMPATIBILITY_MATRIX)`、`4. 原厂磁头备件与老师傅维修能力 (maintenance_capability)`、`5. 格式迁移成本倍增指数 (migration_cost_index)`、`6. 制度与版权约束` 与 `7. 机构工作流惯性`；
  * 精确量化从 **1992 模拟磁带黄金期**（`accessibility = 0.814, maintenance = 1.000, cost = 1.41x, head_wear = 0.150`）到 **2025 备件枯竭与技师退休期**（`accessibility = 0.164, maintenance = 0.053, cost = 10.61x, head_wear = 0.920`，触发 `OBSOLESCENCE_CLIFF_CRITICAL` 可访问性断崖）的非线性演化，并将所选年份的生态状态直接投射为当前信号链的物理磨损分布。
* **历史反事实情景推演 (`eco_scenario = 0..3`) 与 `2×2` 媒介技术生态演化实验室 (`eco_mode = 4`)**：
  * 支持对比 `0 = 真实历史基准轨迹`、`1 = 备件与维修站提前断供危机`、`2 = 无带化数字转轨提前 10 年普及` 与 `3 = 全球公共档案强制冷库与开源磁头备件复刻计划`（实测 2018 年可访问性指数：`提前断供 0.162 < 历史基准 0.278 < 开源档案计划 0.640`），并提供包含 `stream_cdn_client → display → human_operator → ecosystem_simulator` 的完整视听媒介生态全链路节点图预设 `complete_media_ecology_end_to_end_suite`。



### 25. 独立模块扩展：01b 胶片光化学与 LOMO 暗房冲印系统 (`atsim/codecs/photochemical_film.py`, `atsim/stages/film.py` & `film` 层 / `photochemical_film` 节点)

* **胶片模块从摄像机层独立解耦 (`01b 胶片光化学 (胶片/LOMO)` 层、`FilmStage` 与 `photochemical_film` 节点)**：
  * 将原先内嵌在 `camera` 层中的胶片感光化学逻辑独立拆分为专属架构层 `film`（位于 `01 成像系统 (camera)` 与 `02 信号编码 (format)` 之间，`STAGE_ORDER = ('camera', 'film', 'format', ...)`）、独立渲染阶段 `FilmStage` (`atsim/stages/film.py`) 与节点图独立设备节点 `photochemical_film`（`胶片光化学与LOMO冲印台`），同时完整保留对旧版 `camera.film_*` 参数与 `film_scanner_telecine` 节点的无缝向后兼容。
* **10 款经典电影、LOMO 与宝丽来胶片乳剂模型 (`film.stock = 1..10`)**：
  * 除原有 4 款 16mm/35mm 电影胶片外，基于 **10 张 LOMO 实拍实测数据**定量标定新增 6 款标志性 LOMO 与即时显影胶片乳剂：
    * `5 = Agfa Vista / Optima 200`（高感红层灵敏度与琥珀金中间调、深可可红阴影，对齐实拍标定 Agfa Colornotic 与 `[07]` Holga 120N，`R/G = 1.32, B/G = 0.74`）；
    * `6 = Kodak Ektachrome E100VS / Elite Chrome 100`（高饱和度反转片，专供 E-6 in C-41 正冲负，对齐实拍标定 金盏花）；
    * `7 = Lomography Color Negative 400`（玫瑰洋红/丁香紫阴影与奶油暖白高光，对齐实拍标定 Yashica、`[08]` Voigtländer R2M 与 `[09]` Polaroid 1000）；
    * `8 = Expired Svema / ORWO Vintage Film`（高底雾 $D_{\min}$ 抬升与感蓝层热退化，死黑区变为粉彩鼠尾草绿/灰雾，对齐实拍标定 向日葵实测 `p02 = 0.406`）；
    * `9 = Polaroid SX-70 / 600 Time-Zero Integral Film`（一体式相纸碱液药浆扩散与 $\text{TiO}_2$ 二氧化钛反射层奶油高光滚降，对齐实拍标定）；
    * `10 = Lomography LomoChrome Purple XR 100-400`（绿光 $\to$ 紫品红二向色耦合平移）。
* **LOMO 暗房交叉冲洗 (Cross-Processing)、过期底雾、Redscale 反卷与层间抑制效应 (`process_mode = 0..5`)**：
  * **正冲负 (`process_mode = 1`, E-6 Slide in C-41)**：无橙色掩膜反转片进入负片高温药水，反差 $\gamma$ 陡增 `1.38×`，产生标志性的翡翠青绿阴影 (`(G+B)/2 - R = +0.0391`) 与金盏花暖金高光（完美复现实拍标定）；
  * **负冲正 (`process_mode = 2`, C-41 Negative in E-6)**：负片自带的琥珀橙色耦合掩膜未被抵消，经反转显影后令阴影呈现浓郁丁香紫/玫瑰红 (`(R+B)/2 - G = +0.5292`)、高光呈奶油暖黄（复现实拍标定）；
  * **迫冲/减冲 (`process_mode = 3`)、留银冲洗 (`process_mode = 4`, Bleach Bypass)**、**过期胶卷底雾抬升 (`expired_fog`：实测黑位 `p02` 从 `0.2182` 抬升至 `0.3984`)** 以及 **Redscale 反装片基曝光 (`redscale_flip`：`R - B = +0.4496`)**。
* **LOMO 塑料弯月镜渐晕、异形桃心散景 (`bokeh_shape = 1`)、海绵老化漏光与二次重曝**：
  * **单片塑料弯月镜光学 (`lomo_vignette`, `lomo_edge_smear`)**：建模 LOMO LC-A Minitar-1 32mm 与 Holga 120N 60mm 的 $\cos^4\theta$ 边缘照度跌落（复现实拍标定 中心/四角亮度比 `2.27×`）与边缘匹兹伐场曲抹开；
  * **异形光圈散景 (`bokeh_shape = 0..4`, `bokeh_amount`)**：支持 `1 = Heart Bokeh 桃心散景`（二维心形孔径 PSF 卷积与粉红心形光斑轨迹，复现实拍标定 Praktica TL1000 与 `[08]` Voigtländer R2M）、四角星芒、菱形光圈与匹兹伐旋焦；
  * **机壳/海绵老化漏光 (`light_leak_pattern = 1..4`) 与潜影域二次重曝 (`double_exposure`)**，并提供 **`2×2` LOMO 光化学暗房四象限对照实验室 (`process_mode = 5`)**。


### 26. 实拍专项扩展（暗房波纹隔离与强磁消磁物理）：LOMO 节点预设波纹隔离修复 + CRT 遇强磁铁、铁磁荫罩剩磁与 PTC 消磁线圈物理系统 (`atsim/codecs/crt.py` & `display` 层 / `crt_magnet_env` 节点)

* **基准源 默认 LOMO 节点预设下半幅水平波纹扰动根因定位与隔离修复 (`server.py:graph_load_preset`, `stages/display.py`, `codecs/photochemical_film.py`)**：
  * **数学与代码溯源**：基准源 图像下半幅 ($y > 0.62H$) 周期为 $\lambda pprox 24	ext{–}25	ext{ px}$ 的正弦水平波纹扰动，直接来自 `atsim/codecs/perception_and_operator.py:222-226` 机房值班操作员二阶欠阻尼猎振公式 `shifts = ((rows - 0.62H)/(0.38H))^1.5 * tracking_err_Knob * 14.0 * sin(rows * 0.25 + t * 6.0)`（理论波长 $2\pi/0.25 = 25.13	ext{ px}$）；
  * **跨预设状态残留机制**：旧版 `server.py:graph_load_preset` 仅执行增量合并 `merge_values(self.current.params, spec["params"])`，当用户先点选含 `display.percept_enabled = 1.0, percept_mode = 2.0` 的生态总成预设再切入 `lomo_photochemical_darkroom_lab`（仅声明 `display.enabled = 0.0`）时，残留的 `percept_enabled = 1.0` 会通过 `DisplayStage.apply` 的旁路分支继续对 LOMO 画面执行操作员猎振正弦位移；同时若未开启逐节点渲染，默认固定链路的 VHS `tape` 行抖动也会串入；
  * **四重根治方案**：(1) `graph_load_preset` 切换预设时先重置为 `_preset_baseline()` 并对无磁带节点的图自动归零 `tape/player` 时基抖动；(2) `DisplayStage.apply` 在 `enabled == False` 时不再旁路触发 `percept_spec`；(3) `PhotochemicalFilmSpec` 默认 `gate_weave = 0.0, scratches_dust = 0.0`；(4) 前端点击节点图预设自动启用 `G.perNode = true` 并在 `/api/still` 支持 `engine=graph`（实测切换后 `percept_enabled = 0.0, tape.line_jitter = 0.0, disabled_display_max_diff = 0.000000`）。
* **基准源 CRT 遇强磁铁、铁磁荫罩剩磁与内置 PTC 消磁线圈物理系统 (`CRTMagneticEnvSpec`, `apply_crt_magnetic_and_special_env`)**：
  * **三维磁偶极子近场洛伦兹偏转 $\mathbf{F} = -e(\mathbf{v} 	imes \mathbf{B})$ (`crt_magnet_mode = 1, 2`)**：建模钕铁硼强永磁体逼近或贴紧玻壳时的轴向场旋度涡流扭曲（N 极顺时针、S 极逆时针）与横向场径向排斥；当贴紧玻壳时，强径向洛伦兹力将中心区电子束完全排空，形成 **中心黑洞 (`void_fraction = 5.75%`, `center_luma: 0.258 -> 0.000`)**、边缘电子堆积亮焦散环 (`Caustic Rim`) 与高频压缩扫描线摩尔纹；
  * **多阶同心色纯度彩虹环 (`Purity Rainbow Rings`)**：偏转后的电子束以异常入射角穿过荫罩孔/荫栅缝，扫过相邻 `R → G → B → R` 荧光粉条带，形成随磁场梯度层层嵌套的同心/双叶彩虹色环（实测 `ring_chroma_spread = 0.1108`）；
  * **铁磁荫罩磁滞剩磁 (`crt_magnet_mode = 3`) 与内置 PTC 热敏电阻消磁线圈 (`crt_magnet_mode = 4`)**：撤走强磁铁后，低碳钢/殷钢荫罩因磁滞回线保留永久剩磁 $M_{	ext{rem}}$，在画面留下静态彩色纯度污斑（`rem_L1_before = 0.0584`）；触发内置 **PTC 消磁线圈 (`Degaussing Coil`)** 后，50/60 Hz 大电流交变磁场随 PTC 升温呈指数包络衰减振荡，将荫罩剩磁彻底归零（`after_PTC_degauss = 0.0060`, 剩磁清除率 `> 90%`），并提供 `2×2` 强磁物理四象限实验室 (`crt_magnet_mode = 5`) 与独立节点 `crt_magnet_env`。

### 27. 实拍专项扩展（柱面特丽珑与高分辨屏摄摩尔纹）：Sony Trinitron KV-1214 (1978–1981 12 寸柱面特丽珑) 与 4K 屏摄摩尔纹系统 (`TrinitronKV1214Spec`, `apply_trinitron_kv1214_optics`)

* **基于 基准源 原片 → 基准源 实拍 4K 屏摄的像素级物理对位标定 (`kv1214_enabled`, `sony_trinitron_kv1214_showa`)**：
  * **12 英寸单枪三束粗节距垂直荫栅 (`Aperture Grille`) 与水平-垂直高频各向异性**：还原 12 寸小尺寸特丽珑显像管无纵向桥接的连续垂直 R/G/B 荧光粉条与碳黑矩阵竖缝，使水平高频梯度显著强于垂直扫描线梯度（基准源 原片 `HFx/HFy = 0.99`，基准源 实拍 `1.31..1.78`，仿真 `HFx/HFy = 2.715 / 1.48`）；
  * **12 寸特丽珑标志性单根水平钨丝阻尼线 (`y = 0.52H`) 与单向水平柱面玻壳 (`kv1214_cylindrical_bow`)**：垂直方向母线笔直、仅沿水平方向呈圆柱弯曲，并在中部偏下留有一根极细钨丝防振阻尼线阴影；
  * **NTSC-J 9300K (D93) 青蓝白点与 $	ext{Y}_2	ext{O}_2	ext{S:Eu}$ 稀土红粉光晕 (`kv1214_ntscj_9300k`)**：复现 基准源 原片 (`R/G = 0.897, B/G = 0.922`) 进入 KV-1214 后的青蓝冷调白点平移与红橙荧光粉长程内反射晕染（基准源 实拍 `R/G = 0.778, B/G = 1.106`，仿真 `R/G: 0.995 -> 0.901, B/G: 1.116 -> 1.168`）；
  * **4K 屏摄柱面荫栅差拍弯曲摩尔纹 (`kv1214_moire_amount`) 与 基准源/基准源 同屏对照台 (`kv1214_ab_split = 0..2`)**：建模柱面弯曲导致的垂直荫栅视在节距横向调制与 4K 相机拜耳像素阵列的空间差拍干涉，精准复现 基准源 中根根分明的弧形/双曲彩色摩尔纹条带，并支持左 基准源 原片 / 右 基准源 KV-1214 同屏分割对照与 `2×2` 细节微距放大实验室。



### 28. 实拍专项扩展（10 张新 LOMO 实测数据）：光学光晕、逆光耀斑洪流、高调白雾、肥皂泡/六边形散景、移轴浅景深与匹兹伐旋焦系统 (`atsim/codecs/photochemical_film.py`)

* **四大 LOMO 镜头与乳剂光学光晕/眩光模式 (`bloom_veil_mode = 0..5`, `bloom_veil_amount`, `soft_focus_mist`, `dual_tone_split`)**：
  * **`mode = 1` 黄金时刻大逆光太阳耀斑洪流 (`` 夕阳蒲公英标定)**：顶部正逆光太阳形成纯明黄过曝核心 (`R=0.98, G=0.90, B=0.14`)，经无增透膜镜片前向散射向全画幅倾泻指数衰减的金琥珀色光雾洪流（实测上半幅暖金差值从 `0.2568` 跃升至 `0.7738`），而前景背光草甸维持深可可红棕剪影；
  * **`mode = 2` 无镀膜老镜头高调奶油白雾眩光与柔焦 (`` 白雾蒲公英 / `` 高调马卡龙标定)**：强逆光进入无镀膜单镜片引发全画幅 Mie 散射乳白光雾 (`Veiling Glare`)，将暗部黑位 `p02` 从 `0.1208` 抬升至 `0.4319`（对齐 `` 实测 `p02 = 0.391`），配合前组柔焦水汽 (`soft_focus_mist`) 在保留前景叶脉清晰度的同时笼罩梦幻粉彩空气感；
  * **`mode = 3` 点光源双层烛火日冕与防光晕层红橙背散射 (`` 岩石烛火标定)**：孤立高亮光源（如烛火、夜灯）呈现炽白核心 $\to$ 琥珀暖金内日冕 $\to$ 穿透乳剂在片基背面反射回底层感红层的红橙防光晕外环，并在背景暗部激发柔和焦外散景圆斑；
  * **`mode = 4` 斜向双色温冷紫-暖金光学分色光雾 (`` 肥皂泡蒲公英 / `` 标定)**：斜向逆光穿过简易镀膜镜片时随入射角连续分色，左上角呈现柠檬奶油暖白高光、中央阴影谷呈现浓郁丁香紫/洋红、右侧呈现暖金玫瑰散景；
  * **`mode = 5` `2×2` LOMO 光学光晕与散景四象限对照实验室**：同屏实时对比上述 4 种核心光学机制（各象限间实测 `L1 = 0.189 .. 0.231`）。
* **肥皂泡亮环散景 (`bokeh_shape = 5`)、六边形彩灯散景 (`bokeh_shape = 6`)、移轴平面浅景深与匹兹伐旋焦**：
  * **梅耶 Trioplan 肥皂泡亮环散景 (`bokeh_shape = 5`, `` 标定)**：建模三片式柯克镜头过校正球差，使焦外点光源弥散圆边缘能量高度集中，形成外圈锐利亮环、内部半透明的「肥皂泡散景」（实测亮环与中心强度比达 `3.43x`）；
  * **六边形机械光圈奶油彩灯散景 (`bokeh_shape = 6`, `` 马卡龙 / `` 雨滴标定)**：建模 6 片直刃机械光圈在焦外形成的暖金/柠檬黄六边形点光源光斑簇；
  * **沙姆定律移轴/微距平面浅景深 (`tilt_shift_dof`, `focus_plane_y`, `` & `` 标定)**：沿 `focus_plane_y` 焦平面带保持极高锐度（如岩石裂缝与近景纸船），向远景与前景平滑过渡为大光斑散焦（实测合焦带与远景高频比达 `20.72x`）；
  * **19世纪 Petzval 匹兹伐人像镜切向猫眼旋焦 (`swirl_amount`, `` 树丛街灯标定)**：画面中心主体保持零旋焦锐利合焦（`center_L1 = 0.00000`），四周树叶与背景散景沿切向圆弧旋转拉长（`corner_L1 = 0.0916`）。
* **新增 6 款实拍标定胶片乳剂（扩展至 `stock = 1..16` 共 16 款胶卷）与 10 套专属预设**：
  * 新增 `11 = 90s Cinema Tungsten Acid Olive-Chartreuse`（90年代港片钨丝灯酸橄榄黄绿迫冲，``，实测 `B/G = 0.582`）、`12 = Prussian Cyan-Navy & Electric Violet Rain`（普鲁士深青蓝与电光紫雨景，``，实测 `B/G = 1.482`）、`13 = Fujifilm Pro 400H Pastel Candy High-Key`（富士高调粉彩马卡龙，``）、`14 = Vintage Brass-Olive & Maroon-Plum Petzval`（复古黄铜橄榄与紫红暗部，``，实测 `B/G = 0.666`）、`15 = LOMO LC-A Minitar-1 Turquoise-Emerald X-Pro`（松石青绿高反差正冲负与重度弯月镜暗角，``，实测 `(G+B)/2R = 1.467`）、`16 = Meyer Trioplan Soap-Bubble Golden-Violet Dream`（肥皂泡散景紫金梦幻乳剂，``）。

## 二、虚拟电视台工作站扩展系统总表（扩展单元 E ～ 扩展单元 I 已全部完成）

结合最新《虚拟电视台工作站》设计构想与现有已落地的 7 大物理系统（同步/TBC、共享电源、CRT屏摄、伺服走带、双输入切换台、实时示波器、实拍专项标定），将尚未覆盖的广播级生产链机制提炼为以下 **5 个独立可推进的扩展模块（扩展单元 E ～ 扩展单元 I）**：

| 系统编号 | 模块名称 | 核心物理与工程机制 | 对应架构层 / 新增节点 | 推荐优先级 |
| :--- | :--- | :--- | :--- | :--- |
| **扩展单元 E** | **广播级配线架、分配放大器与 75Ω 阻抗负载系统**<br>(`patchbay_da`) | **1. 一进多出分配放大器 (DA) 与 75Ω 端接负载模型**：建模未端接开路（`High-Z`，电平翻倍 `+6 dB` 过曝削波与驻波振铃）、正常单端接（`75Ω`，标准 `1.0 Vpp`）、违规双端接/多并接（`37.5Ω / 25Ω`，幅度跌落 `-6 dB`、同步顶压缩引发下游失锁及高频滚降）；<br>**2. 矩阵路由器与跳线配线架 (Patchbay)**：支持同一信号源分支送往录像机、直通监看与 TBC 校正链路，以及强弱信号交叉点电容耦合串扰 (Crosspoint Crosstalk)。 | `connector` 层 + `dist_amp` / `patch_router` 多输出节点 | ✅ **已完成** (`70/70 passed`) |
| **扩展单元 F** | **多画面监视墙与独立 CRT 个体老化阵列**<br>(`multiview_wall`) | **1. 多节点实时同屏监看墙 (Monitor Wall)**：在 WebUI 与渲染端提供 `2×2` / `3×2` 演播室监视墙视图（同时监看：摄像机原片、磁带直出、TBC 校正后、预监 PVW、主输出 PGM）；<br>**2. 监视器个体差异与无信号物理响应**：每台 CRT 具备独立的尺寸、扫描制式、色温老化、偏转失真与校准偏差（操作者看到的偏色不影响信号本身）；某路信号中断时呈现真实黑场、雪花、行场失锁或帧冻结，而非统一报错。 | `display` 层 + WebUI 监视墙面板 (`monitor_wall.js`) | ✅ **已完成** (`70/70 passed`) |
| **扩展单元 G** | **测试信号发生器与工程闭环校准台**<br>(`test_signal_gen`) | **1. 标准广播测试信号发生器节点 (`test_generator`)**：内置 SMPTE / EBU 75% 与 100% 彩条（含 `-I/+Q` 与 PLUGE 黑电平三阶脉冲）、多波群频率响应测试卡 (`0.5–5.0 MHz` Multiburst)、五阶梯线性斜坡 (5-Step Staircase) 与 `2T` 正弦平方脉冲，以及 `1 kHz / 440 Hz` 基准正弦测试音；<br>**2. TBC 直通/校正同屏对照与工程校准闭环**：配合已有的波形监视器与矢量示波器，量化测量整条链路的微分增益 (DG)、微分相位 (DP)、各波群带宽衰减与群延迟。 | `source` 层 + `test_generator` 信号源节点 | ✅ **已完成** (`70/70 passed`) |
| **扩展单元 H** | **摄像机控制单元 (CCU) 与演播室主同步发生器 (Genlock)**<br>(`studio_ccu_genlock`) | **1. 演播室集中式 CCU 机架控制**：通过多芯演播室摄像机线缆集中遥调多机位的光圈 (Iris)、总黑电平 (Master Pedestal)、R/B 增益与黑平衡，模拟多机位同场照明下的个体色彩偏差与配平；<br>**2. 主同步发生器 (Blackburst SPG / Genlock)**：向各摄像机与录像机分发基准黑场同步，模拟脐带线缆长度引起的副载波相位偏移 ($\Delta\phi_{	ext{sc}}$) 及某机位 Genlock 掉锁时的切换台瞬态撕裂。 | `camera` / `player` 层 + `ccu_rack` 控制节点 | ✅ **已完成** (`70/70 passed`) |
| **扩展单元 I** | **主控播出自动化 (Playout) 与双总线预监/干净输出**<br>(`master_playout`) | **1. PVW / PGM 双总线与干净输出 (Clean Feed)**：扩展切换台支持预监母线 (Preview)、节目母线 (Program) 与不叠台标字幕的净画面输出 (Clean Feed)，支持下游键控器 (DSK) 独立挂载；<br>**2. 主控播出单状态机与故障应急接管**：模拟按时间表 (Playlist) 执行 `待播 → 录像带预卷 (Preroll) → 播出 → 切下一条`；当磁带未就绪、粘带失锁或素材超时时，自动或手动切入「请稍候」垫片/彩条备用路并记录广播级事件日志 (As-Run Log)。 | `connector` / `overlay` 层 + 主控播出单面板 | ✅ **已完成** (`70/70 passed`) |


## 三、下一阶段基础设施扩展清单（全链路广播电视基础设施扩展 · 扩展单元 J ～ 扩展单元 N）

结合最新上传的《五大广播电视基础设施系统 (`README.txt`)》设计构想，围绕 **共用基础设施 A（素材与时间系统）、基础设施 B（统一信号图）、基础设施 C（监控与事件反查系统）**，将 5 大新系统拆解为以下可独立或组合推进的 **扩展单元 J ～ 扩展单元 N**（按文档第七节推荐开发优先级排序）：

| 系统编号 | 模块名称 | 核心物理模型、状态机与数据结构 (`README.txt`) | 对应架构层 / 新增节点 | 推荐优先级 |
| :--- | :--- | :--- | :--- | :--- |
| **扩展单元 J** | **多代转录与跨格式复制对照工作站**<br>(`transfer_workstation`) | **1. 三类工作站物理路径**：<br>  • **A. 模拟磁带数字化采集站 (Analog-to-Digital Ingest)**：VHS/Video8/Betacam → TBC → 采集卡 ADC（采样时钟锁定、消隐裁切、音画同步偏移 `audio_sync_offset`，采集卡不凭空修复源缺陷）；<br>  • **B. 磁带多代复制站 (Multi-Gen Tape Dubbing)**：源机 → 分配/校正 → 目标机，区分可被 TBC 抑制的时基误差与不可逆累积的色度/带宽/非线性退化；<br>  • **C. 数字格式转码站 (Digital Transcode)**：`4:2:2 → 4:1:1 (DV25) → 4:2:0 (MPEG-1 VCD)` 色度重采样、帧率转换与多代 DCT 重编码损失；<br>**2. 代际历史追踪 (`TransferEvent` 链)**：完整记录每一代的设备、输入/输出格式、TBC 模式与代际序号 (`generation_index`)，并提供四路同屏「复制链对照实验室」。 | `codec` / `player` 层 + `ingest_station` / `transcode_station` 节点 | ✅ **已完成** (`78/78 passed`) |
| **扩展单元 K** | **线性磁带编辑控制器与早期非线性编辑系统**<br>(`edit_controller_nle`) | **1. 模拟/数字磁带线性编辑机 (Sony RM-450 / BVE-9100 Linear Editor)**：<br>  • 源录像机 (Player VTR) + 录制机 (Recorder VTR) 双机预卷 (`Pre-roll`) 伺服同步、CTL/时间码寻址入出点偏移 (`±1..3 frames`)；<br>  • **组合编辑 (Assemble Edit)**（全磁鼓重写，结尾处留下控制轨与色度相位跳变切口）vs **插入编辑 (Insert Edit)**（保留原 CTL 控制轨、飞抹磁头 Flying Erase Head 精确替换视频或伴音轨）；<br>**2. 早期非线编工作站 (2000年代初 DV/Windows 2000 NLE)**：<br>  • 实现 `EditDecision` / `EditSession` 多轨时间线、实时预览缓存降级（磁盘吞吐不足时掉帧/半分辨率预览）与最终渲染导出的差异对照。 | `tape` / `codec` 层 + `linear_editor` / `nle_timeline` 节点 | ✅ **已完成** (`78/78 passed`) |
| **扩展单元 L** | **事件驱动与条件驱动的播出自动化编排引擎**<br>(`playout_automation_v2`) | **1. 完整事件状态机 (`PlayoutEvent`)**：<br>  • 状态流转：`SCHEDULED → PREPARING (VTR 预卷与伺服建锁) → READY → ON_AIR → COMPLETED`；<br>  • 6 类分级异常分支：`MISSING_ASSET`（素材缺失）、`NOT_READY`（磁带预卷未就绪/粘带超时）、`SIGNAL_LOSS`（信号断流）、`OVERRIDDEN`（导播手动强切接管）、`FALLBACK`（自动切垫片）、`HELD`（挂起等待确认）；<br>**2. 时间驱动 + 条件驱动 + 主备冗余倒换闭环**：连接录像机物理伺服状态与音频静音/黑场/静帧监测器，生成带故障溯源链的完整广播级事件日志。 | `overlay` / `connector` 层 + `automation_controller` 节点 | ✅ **已完成** (`86/86 passed`) |
| **扩展单元 M** | **卫星接收机 (Satellite IRD) 与模拟/数字悬崖效应系统**<br>(`satellite_ird`) | **1. 六级卫星接收链路**：抛物面天线指向/雨衰 (`Rain Fade` / 日凌 `Sun Outage`) → LNB 低噪声变频器（本振频偏与相位噪声） → 中频调谐器 → 解调与纠错 → 传输流/基带输出；<br>**2. 模拟 C 波段 FM 卫星电视 vs 数字 DVB-S (QPSK + Viterbi/Reed-Solomon) 双分支**：<br>  • **模拟卫星分支**：低载噪比下出现标志性的 **FM 门限脉冲打火噪点（Threshold Sparkles / 鱼骨黑白流星点）** 与伴音副载波杂音；<br>  • **数字卫星分支**：还原 **数字悬崖效应 (Digital Cliff Effect)**——误码率在纠错门限内画面完全干净，突破门限瞬间爆发MPEG-2 宏块崩坏、PTS/DTS 音画失步、画面死锁冻结或服务丢失，同时输出「原始传输流统计」与「解码基带画面」。 | `rf` / `codec` 层 + `sat_dish_lnb` / `sat_ird_receiver` 节点 | ✅ **已完成** (`86/86 passed`) |
| **扩展单元 N** | **电视发射机激励链与端到端开路广播测试台**<br>(`tv_transmitter_chain`) | **1. 电视台大功率发射机物理模型**：高稳晶振频率基准 → 残留边带 (VSB) / 数字调制激励器 (Exciter) → 电子管/固态高功放 (HPA) **AM-AM 微分增益压缩与 AM-PM 微分相位失真**、群延迟失真与伴音-图像互调 → 腔体双工器与馈线驻波；<br>**2. 模拟发射 vs 数字发射端到端对照实验**：构建 `编辑机 → 播出输出 → 发射链模型 → 传播信道 → 电视接收机 → CRT` 与「直通基带参考」同屏对比，量化功放非线性、滤波器群延迟与信道噪声各自的独立贡献。 | `rf` 层 + `tv_exciter_pa` 发射机节点 | ✅ **已完成** (`86/86 passed`) |


## 四、视听媒介生态与档案考古模拟器扩展总表（扩展单元 O ～ 扩展单元 U 已全部完成）

结合最新上传的《完整的视听媒介生态模拟器 (`README.txt`)》设计蓝图，项目从「单机与广播链路模拟器」进一步升级为涵盖 **媒介考古、物理环境场、档案保存生命周期、网络流媒体传播、人类感知与操作者行为、参数反演校准（方向 A → 方向 B → 方向 C）** 的完整视听媒介生态模拟器，拆解为以下 **7 个独立或组合可推进的扩展模块（扩展单元 O ～ 扩展单元 U）**：

| 系统编号 | 模块名称 | 核心物理模型、状态机与数据结构 (`README.txt`) | 对应架构层 / 新增节点 | 推荐方向与优先级 |
| :--- | :--- | :--- | :--- | :--- |
| **扩展单元 O** | **测量、误差预算、参数反演与三级参数审计系统**<br>(`inversion_calibration_lab`) | **1. 参数反演求解器 (Parameter Inversion)**：给定退化输出帧或实测波形/频谱，自动反推最可能的设备参数区间与多解置信椭圆（如区分滤波器带宽限制 vs 磁头磨损 vs 散焦）；<br>**2. 三级参数显式分离追踪**：将全引擎参数严格标记并分离统计为 `Physical（物理参数）`、`Device（设备型号校准参数）`、`Artistic（非物理创作调整）`，量化各自对最终画面的独立贡献；<br>**3. FADGI 数字化质量评价与误差预算**：对照联邦视听数字化指南输出亮色电平、时基、频响与信噪比合规报告。 | `scopes` / `core` 层 + `inversion_analyzer` 节点 | ✅ **已完成** (`93/93 passed`)<br>**方向 A（可验证的视听物理实验室）** |
| **扩展单元 P** | **电影胶片（16mm/35mm）化学成像与电视胶转磁系统**<br>(`film_telecine`) | **1. 卤化银胶片感光与冲洗模型**：H&D 特性曲线（趾部/肩部密度）、三层彩色耦合染料光谱交叉、曝光依赖颗粒团簇 (`Film Grain`) 与胶片批次差异；<br>**2. 胶片物理老化与放映机机械模型**：醋酸纤维素酸败收缩 (`Vinegar Syndrome`)、青/品红/黄三层染料非对称褪色（早期柯达/富士胶片偏红棕）、纵向走片划痕、片门抖动 (`Gate Weave`) 与双叶/三叶旋转快门闪烁；<br>**3. 飞点扫描 / CCD 胶转磁 (Rank Cintel Telecine)**：`24 fps → 29.97i (3:2 Pulldown 交错梳状伪影)` 与 `24 fps → 25i (PAL +4.16% 加速与伴音升调)`，打通「早期电影 → 电视胶转磁播出 → 家用 VHS 录屏」跨媒介历史链。 | `camera` / `format` 层 + `film_scanner_telecine` 节点 | ✅ **已完成** (`93/93 passed`)<br>**方向 B（媒介考古核心 · 化学成像 vs 磁记录）** |
| **扩展单元 Q** | **档案保存生命周期与媒介考古溯源系统**<br>(`archival_archaeology`) | **1. 四阶段档案生命周期模拟器**：建模 `原始记录 → 生产与传播 → 载体保存与失效风险（温湿度加速老化、设备停产与接口消亡导致访问成本跃升） → 修复、抢救数字化与再编目`；<br>**2. 「保存成功 ≠ 信息完整」重要属性损耗评估 (FADGI / NFSA Significant Properties)**：模拟某次数字化或格式迁移虽保留了画面像素，却丢失了原始 VITC 时间码、场序结构、色域特征、字幕轨或场记元数据的考古证据链；<br>**3. 多版本并存谱系树 (`ArchivalLineageTree`)**：同屏对比同一作品的「原始载体残件、保存母版 (Preservation Master)、修复清洗版与流媒体传播版」。 | `player` / `overlay` 层 + `archival_vault` 节点 | ✅ **已完成** (`103/103 passed`)<br>**方向 B（视听媒介考古与档案系统核心）** |
| **扩展单元 R** | **空间物理环境场与多维耦合系统**<br>(`spatial_environment_field`) | **1. 统一二维机房/场景环境状态场 (`EnvironmentField`)**：每台设备根据自身在空间中的坐标 $(x, y)$ 读取局域温度、相对湿度、光照频谱与电磁干扰场，而非共用单一全局数值；<br>**2. 四大环境物理耦合子系统**：<br>  • **温湿度场**：热惯性、冷机抱入暖房镜头结露/磁鼓粘带冷凝风险；<br>  • **电磁环境场**：变压器/发射塔空间电磁耦合、屏蔽效能与接地环路路径；<br>  • **声学空间场**：演播室/客厅房间脉冲响应混响、驻波、环境底噪与麦克风指向性；<br>  • **光学空间场**：混合光源照明频谱、杂散光与视窗漫反射。 | `core` / `connector` / `audio` 层 + `env_chamber` 节点 | ✅ **已完成** (`103/103 passed`)<br>**方向 A+B 桥梁（让设备真正处于物理世界中）** |
| **扩展单元 S** | **网络视听传播、平台转码与流媒体协议系统**<br>(`network_streaming_era`) | **1. 早期互联网流媒体与现代自适应流媒体 (`RealVideo / WMV / H.264 HLS/DASH`)**：建模码率控制、GOP 帧间预测、关键帧呼吸脉冲与缓冲区动态；<br>**2. 分组网络传输与客户端行为**：带宽波动、RTT 延迟、抖动、UDP/RTP 丢包切片撕裂 vs TCP/HLS 预缓冲卡顿转圈 (`Rebuffering Stall`)、ABR 自适应分辨率阶跃切换 (`720p → 240p → 1080p`) 与音画同步追赶；<br>**3. 平台多次转码与跨时代全链路对照**：打通 `广播电视 → 录像带 → 数字采集 → 平台上传二次转码 (色度抽样+CRF压缩+平台水印) → 自适应流媒体 → 用户二次录屏` 的跨时代传播痕迹对比。 | `codec` / `connector` 层 + `stream_cdn_client` 节点 | ✅ **已完成** (`113/113 passed`)<br>**方向 B+C（从模拟广播进入互联网视频时代）** |
| **扩展单元 T** | **人类视觉感知与机房操作者行为闭环系统**<br>(`perception_and_operator`) | **1. 观众感知心理物理模型 (Viewer Perception)**：基于显示器绝对亮度、环境照度与观看距离（视角 CPD），建模人眼对比敏感度函数 (CSF)、闪烁融合与视觉暂留，量化「物理实测误差 vs 人眼主观清晰度/年代感/舒适度评分」的差异；<br>**2. 机房操作者认知与操控行为模型 (Operator Behavior)**：操作者根据波形监视器、预监画面与报警做出手动跟踪微调、CCU 配平、重试或绕过故障决策，以及高时间压力下的操作延迟、过调震荡与误切事故。 | `display` / `overlay` 层 + `human_operator` 节点 | ✅ **已完成** (`113/113 passed`)<br>**方向 A+C（设备输出 ≠ 观众体验与有人值守生产）** |
| **扩展单元 U** | **媒介技术生态演化与历史情景模拟器**<br>(`media_ecosystem_evolution`) | **1. 七维宏观媒介生态动力学 (`MediaEcosystemSimulator`)**：联动追踪 `设备存量`、`介质存量`、`跨代技术兼容性矩阵`、`备件与技师维护能力`、`格式迁移转换成本`、`制度约束` 与 `机构工作习惯惯性`；<br>**2. 历史反事实情景推演与瓶颈预警**：推演「当最后一台可用录像机/磁头备件失效时，存量档案可访问性断崖」或「某项格式提前/推迟十年普及」的长期演化曲线，并将生态年代状态直接投射为当前信号链的设备可用率与老化分布。 | `core` / `overlay` 层 + `ecosystem_simulator` 节点 | ✅ **已完成** (`113/113 passed`)<br>**方向 C（宏观视听技术史与生态演化模拟器）** |
