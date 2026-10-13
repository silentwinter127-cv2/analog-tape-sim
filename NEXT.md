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

**相机阶段那 0.785 已定位到 `atsim/stages/camera.py:380` 的 `_roll_off()`——高光肩部曲线。**

```python
dr    = 8.0                      # dynamic_range 默认
knee  = 0.5 ** (dr/8)   = 0.500
white = 2 ** ((dr-8)/8) = 1.000
span  = white - knee    = 0.500
t     = clip((lin - knee) / span, 0, 1)
rise  = 2t / (1 + t)
comp  = where(lin <= knee, lin, knee + (1-knee) * rise)
```

**两端精确（`comp(knee)==knee`、`comp(white)==1.0`），但中间被【拉向 1.0】**：
`t=0.5` 时输出 `0.833`，而线性到 white 应为 `0.750`。**钉死两端 + 上抬中间 = 压缩对比度。**

**而 `dr` 越大 `knee` 越低、受影响范围越大**——这解释了为什么把 `dynamic_range` 从 8 提到 14 反而让对比度
从 0.711 掉到 0.589。

**所以下一步是**：
1. **改 `rise`**，让肩部在 `dynamic_range` 宽时更接近线性（真实的宽容度传感器不该压得这么狠）。
   约束：`comp(knee)==knee` 和 `comp(white)==1.0` 必须保持精确，否则会重犯"没有纯白像素"那个已修的缺陷。
2. **然后用同一个方法找 `player` 的 0.588**：在 `player.py` 里搜对整个画面做非线性映射的行。
   （已排除：全部参数逐个中性化最大只变 0.0002；带宽 3/16/40 MHz 逐位相同。）
3. **验收标准（极简）**：黑到白斜坡通过链路后对比度 **接近 1.0**。一个斜坡，一个数字。

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
