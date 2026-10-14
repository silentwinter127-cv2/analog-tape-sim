# 下一步 / NEXT

**这份文件的用途**：让一个全新的上下文窗口只看这一页就能接着干活，而不必重读整段对话。
**每次收尾时更新它**，并保持在一屏以内。`tools/` 里的诊断脚本是可复用的仪器，不要重写。

---

## 当前任务

色度校准后的观感问题是**对比度被压缩**：任何输入进去都只剩约 **59%** 的对比度。

## 已确认的事实（都实测过，不必重测）

**仪器**：`tools/stage_trace.py` —— 包住每个阶段的 `apply`，测每步的对比度。
这是本仓库最有用的诊断工具：**它能指出"哪一步"，而单纯的参数中性化只能"排除参数"。**

**测试极简**：一个黑到白的水平斜坡（对比度 0.2896），或者 0.25/0.75 分场（0.2500）。
`tools/player_probe.py` 里都有。

**两处固定压缩，都不响应任何参数**：

| 阶段 | 斜坡通过后 | 说明 |
|---|---|---|
| 源 | 1.000 | |
| `camera` | **0.785** | 固定通路，无条件 |
| `format` `codec` `connector` `tape` | 1.000 | 贡献精确 0.0000 |
| `player` | **0.588** | 固定通路，无条件 |
| `display` | 0.588 | 贡献精确 0.0000 |

**已排除（每一项都实测过）**：
- 显示阶段：`enabled` / `gamma` / `saturation` —— 全部无效
- 十四个相机项：`halation` `ccd_smear` `sensor_noise` `noise_grain` `dynamic_range` `sharpness`
  `bloom_threshold` `fixed_pattern` `spike_spread` `flare_defocus` `motion_blur` `stuck_fraction`
  `chroma_aberration` `overflow_width` —— 单独或全部中性化都无效
- AE 环路：`ae_limit_stops=0` / `ae_target=0.5` / `ae_tau_s=1000` —— 全部 0.785，精确到 0.0000
- `format`/`player` 带宽：3 / 16 / 40 MHz —— 逐位相同
- `codec.dv_enabled` 开或关 —— 相同
- `connector`、`tape` —— 贡献 0.0000
- `player` 全部参数逐个中性化 —— 最大变化 0.0002

## 下一步动作 —— 反色精确到 `_separate` 的 `luma = comp - chroma` 一行

### 完整的极性链（每一步都用与【自身输入】的相关测量）

```
源照片                                   （基准）
进入 player 的 composite          corr with source 【+0.923】  ✓
_chroma_phase_error 返回的色度     corr 【+1.000】              ✓   ← 不是它
时基位移 disp                     最大 2.4 px / 533 宽（0.45%）  ✓   ← 不是回绕
重建出的画面（= _separate 的产物）  corr with source 【−0.358】  ← 【反色在这里】
display / overlay                +1.000                        ✓
```

**而 `_separate` 里唯一能产生负相关的语句就是最后一行：`y_band = comp − chroma`。**

### 已排除的（全部实测，不必重测）

| 嫌疑 | 结论 |
|---|---|
| `split_composite` 的色度除法 | **频带分离本身正确**：干净复合信号上 chroma 0.00009 对输入 0.28958，luma 1.000 |
| `_chroma_phase_error` | chroma 相关 **+1.000** |
| `_frequency_response` | 只贡献 spread 的 **0.0006** |
| 时基位移 / `horizontal_shift` 回绕 | disp 最大 **2.4 px**（0.45% 宽度），太小 |
| 增益项 | `output_gain` / `output_bias` 默认 0 |
| AE / AWB 环路未收敛 | 渲染 frame 1/10/40/100 全部 0.64 倍，完全平稳 |
| display 阶段 | `enabled` / `gamma` / `saturation` 全部无效 |
| `player` 全部参数逐个中性化 | 最大 0.0002 |

### 四次失败留档（**不要重复这些**）

| 尝试 | 极性 | 对比度 | 套件 |
|---|---|---|---|
| 原状 | −0.358 | 0.1678 | 163 通过 |
| 按实测 RMS 增益归一化 | +0.551 | 0.1027 | 1 项失败 |
| 按相位相关门控 | +0.589 | 0.1035 | 1 项失败 |
| 结果上界裁剪 | −0.314 | 0.1559 | 1 项失败 |
| 有效深度上界 0.5 | −0.288 | 0.1499 | 163 通过 |

**全部回退。前两次修好极性却弄坏对比度，说明那个除法不是原因。**

### 下一步（唯一还没做的一步）

**在 `_separate` 里测 `chroma` 与 `comp` 的关系**：

```python
_sep_probe("comp", comp)                              # composite 自身
_sep_probe("chroma_band after the division", chroma)  # 被减掉的那一项
```

`_sep_probe` 已存在且受 `ATSIM_SEP_PROBE=1` 控制（不设则完全静默，已验证）。

**判据**：如果 `chroma` 的 spread **大于或接近** `comp` 的，那 `luma = comp − chroma` 必然是负相关
——**即 `fft_bandpass` 在这个信号上取走的能量超过应有的**，尽管它在【干净】复合信号上正确。
**两次测量的差别在信号**：干净的是纯 luma（未调制），真实的是**已调制**的复合信号。

**若确实如此，正确修法是让 `chroma` 的幅度受 `comp` 约束**（按 `min(1, rms(comp)/rms(chroma))` 缩放），
**而不是动那个除法。**

**验收标准（两条必须同时过）**：极性 > **+0.5**；套件 **163 全过**且真实照片对比度不低于 0.1297。

### （以下为上一轮的记录，保留作背景）

### 确定的部分

**极性测量**（`tools/polarity.py`，本会话唯一能看见符号的工具）：

```
链路每阶段输出与其【自身输入】的相关：
   camera +0.999   format/codec/connector/tape +1.000
   player 【−0.352】 ← 【反色在这里】
   display +1.000

夜视模块单独作用在源图上：ir_highlights +0.993   monochrome_green +1.000   add_illuminator +1.000
```

**均值与标准差看不见反色**（图与其负片共享两者）——所以三个会话的标量测量都只能说"有点平"。

### 而我【已排除】的机制

**不是 `split_composite` 的色度除法**。这是本轮新测的，而且它推翻了我上一版的判断：

```
干净复合信号上的频带分离（逐列常数画面）：
  chroma spread 0.00009（输入的 0.000）   luma spread 0.28957（1.000）   ← 正确
  comb_depth 0.55 时 luma 仍为 1.000
彩条（应当带色度）：chroma 0.31331（0.702）  ← 也正确
```

所以我上一版记的"`fft_bandpass` 从逐列常数斜坡取走大量能量"是**错的**——那是别的信号上的读数。

### 四次尝试，四次失败（表格留档）

| 尝试 | 极性 | 对比度 | 套件 |
|---|---|---|---|
| 原状 | −0.358 | 0.1678 | 163 通过 |
| 按实测 RMS 增益归一化 | **+0.551** | 0.1027 | 1 项失败 |
| 按相位相关门控 | **+0.589** | 0.1035 | 1 项失败 |
| 结果上界裁剪 | −0.314 | 0.1559 | 1 项失败 |
| 有效深度上界 0.5 | −0.288 | 0.1499 | 163 通过 |

**全部回退。** 前两次修好了极性但弄坏对比度；后两次两个都没修好。**所以那个除法不是原因，而我的四次编辑都作用在错误的点上。**

### 下一步（缩小到 `player.apply` 的其它语句）

`player.apply` 里除 `_separate` 外还有：

```python
i, q = self._chroma_phase_error(i, q, buf)      # 嫌疑 1：色度相位，可能旋转到反相
y = self._frequency_response(y)                 # 已测：只贡献 0.0006
y = y * (1.0 + 0.25 * self.output_gain) + 0.08 * self.output_bias   # 默认 0
sig = SignalFrame(y, i, q); rgb_out = sig.to_rgb()
```

**而 `_separate` 内部还有 `disp = self._timebase_offset(buf, ...)` 与三处 `horizontal_shift`** ——
**`horizontal_shift` 是【回绕】位移，大位移会把画面在水平方向错位，那会让相关变负。**

**所以下一步：在 `_separate` 里把 `disp` 打出来。** 用已有的受控探针（`ATSIM_SEP_PROBE=1`）：

```python
_sep_probe("disp", disp)
```

**如果 `disp` 的量级接近半幅宽度，回绕位移就是反色的来源**——而那是一个真正的缺陷：
时基误差应当是几行的抖动，不是半幅的错位。

**验收标准（两条必须同时过）**：
- 极性 > **+0.5**（现在 −0.358）
- 套件 **163 全过**，且真实照片对比度不低于现在的 0.1297（真实帧 0.2032）

## 不要重复的弯路（每条都花过一整轮）

- **尺度陷阱**：源图在 2364×1773 上量、渲染在 533×400 上量，两者不可比。**一律在渲染器自己的尺寸上量。**
- **跨版本取数**：同一条链先后读出 0.1953 / 0.5552 / 0.4167 / 0.3888 / 0.8198，来自文件的不同版本。
  **一次归因必须在一个版本上跑完。**
- **PowerShell**：`python -c` 的引号会被吃掉，heredoc 会剥掉引号。**改动一律写成脚本文件。**
- **锚点**：注释不属于锚点（最常被改）；多行块可能实际是单行；缩进可能是 4 或 8。**用行号或从文件读缩进。**
- **正对照先行**：任何关于真实数据的数字之前，先在一个【已知答案】的合成案例上验证仪器。
  （本会话有一次三个错误诊断，都是因为只对着模型验证仪器。）

## 本会话已修好并验证的（不要再动）

- `atsim/codecs/nightvision.py`
  - 绿显示权重 = 实测的 `0.2168:0.4294:0.3538`，按【和】归一化（不是按 luma）
  - `chroma_residue`（默认 0.35）：保亮度的原图色度透传
  - `chroma_noise`（默认 0.15）：逐通道噪点，减三通道均值，实测 spread 1.23（真实 1.27）
  - `illuminator_field`：以光束原点为中心的径向衰减，按【均值】归一化，参考刻度 0.062
  - 模式预设**不含** `illuminator`（它曾覆盖用户设置，导致参数到不了函数）
- `atsim/codecs/operator.py`：`HAND` 频率 ×0.60
- `atsim/web/session.py`：`resolved_params` 解析夜视模式（渲染器与 API 单一真相来源）

## 验证

```
python tools/verify.py     # 163 项，必须全过
python tools/selftest.py   # ALL CHECKS PASSED
```

**改动后两个都要跑。** 统计缓存：`tools/calib/scan30.json`（3111 帧逐帧统计）、`tools/calib/cuts.json`。
