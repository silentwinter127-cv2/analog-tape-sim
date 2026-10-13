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

## 下一步动作（具体到文件）

**相机阶段【不是】压缩，是放大 —— 上一版这里记错了。** 干净配置（schema 默认，`dynamic_range` **不碰**，保持 8.0）
下的实测：

| 阶段 | 斜坡 | 分场 | 说明 |
|---|---|---|---|
| 源 | 1.000 | 1.000 | 斜坡自身对比度 0.2896，分场 0.2500 |
| `camera` | **1.082** | **1.167** | **S 曲线：两端收、中间放 → 提高对比度** |
| `format`→`tape` | 1.082 | 1.167 | `signal` 通道逐位不变（0.3136 / 0.2905） |
| `player` | **0.800** | **0.870** | **唯一压缩的一级** |

**两个输入的比值【不同】（1.082 vs 1.167）→ 所以相机是 S 曲线，不是标量增益。**
`_roll_off`（`camera.py:380`）把 `lin=0.6` 送到 `comp=0.6667`，**大于线性值，所以它是抬高中间调**——
我上一版把 0.785 记成"压缩"是因为那次用了 `dynamic_range=14.0`，**跨配置取数，正是第 71 行那条弯路。**

**所以只剩一个目标：`atsim/stages/player.py` 里那个无条件压缩。**

**已排除（在【同一配置】下实测）**：
- YIQ 往返还**无损**：斜坡和分场都是 **1.000**，`y` 跨满 [0,1]。`rgb_to_yiq`/`yiq_to_rgb` 在 `atsim.core.signal`
- `player` 全部参数逐个中性化 —— 最大变化 0.0002
- `output_gain` / `output_bias` 默认就是 0
- 带宽 3 / 16 / 40 MHz —— 逐位相同
- `head_wear`、`color_noise`、`blue_screen` —— 中性化无效

**`player.apply` 的实际通路**（已读，见 `player.py` 的 `apply`）：
```python
y, i, q = self._separate(comp, buf)      # 嫌疑 1
i, q = self._chroma_phase_error(i, q, buf)
y = self._frequency_response(y)          # 嫌疑 2（带宽扫过无效，但函数本身可能是固定的）
y = y * (1.0 + 0.25 * self.output_gain) + 0.08 * self.output_bias   # 默认 0
sig = SignalFrame(y, i, q); rgb_out = sig.to_rgb()
```
**下一步：直接对斜坡的 `y` 调用 `_separate` 和 `_frequency_response`，看哪一个把对比度乘了 0.8。**
方法：绕开渲染器，把这两个方法当作纯函数测——**这是唯一还没做过的隔离。**

**验收标准**：黑到白斜坡通过链路后对比度 **接近 1.0**（相机现在给 1.082，所以链路应落在 1.0 附近）。

## 仪器本身修过两次（都是盲区，务必用修好的版本）

1. `tools/stage_trace.py` 只记 `.rgb`，但**链路实际传递的是 `meta["signal"]`**——所以"format 贡献 0.0000"
   是量错了缓冲区。**修好的版本在 `tools/composite_trace.py`，它同时记 `rgb` / `composite` / `signal` 三栏。**
2. **`buf.composite` 全程是 `None`**（实测），所以 `player.apply` 开头的 `comp = buf.composite` 不是它拿到
   信号的途径——**别再去追 composite。**

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
