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

**压缩已定位到 `atsim/stages/player.py` 的 `_separate()`，边界缩到它末尾的 6 行。**

逐层实测（全部在【同一配置】下：schema 默认，`dynamic_range` 保持 8.0）：

| 接缝 | 斜坡 |
|---|---|
| 源的对比度 | 0.2896（1.000） |
| `camera` → **1.082** | **S 曲线，放大**（不是压缩，上一版记错了） |
| `format`→`tape` | 不变（signal 通道 0.3136 → 逐位相同） |
| 到达 `player` 的 signal | **1.083** |
| **`_separate` 返回** | **0.802** ← 【压缩 26% 在这里】 |
| `_frequency_response` | 只贡献 **0.0006** |
| 离开 player | 0.800 |

**进 `_separate` 内部再测一层**（用 `tools/separate_inside.py`，它包住 `yc_separate` / `demod_chroma` /
`remod_chroma` 这三个真实调用）：

```
真实 composite                std 0.3121   (1.078)
after yc_separate #1          std 0.3121   (1.078)   ← 无损
after demod_chroma (i)        std 0.0030            ← chroma 很小，正确
after remod_chroma            std 0.0018
after yc_separate #2          std 0.3570   (1.233)   ← 分离本身【放大】到 1.233
而 _separate 返回              0.802
```

**所以压缩发生在【第二次 `yc_separate` 之后、函数 return 之前】的那一段**，即 `cross_color` / `cross_luma`
残余项与增益：

```python
comp_shifted = (y_band + remod_chroma(i, q, self.std, chroma_gain=gain))     # gain = 0.63
y_band, chroma_band = yc_separate(comp_shifted, ..., comb_depth=self.comb_depth, ...)   # → 1.233
chroma_band = chroma_band / gain
if self.cross_color > 1e-4:
    detail = highpass(y_band, ...)                       # 默认 cross_color = 0.45，【会执行】
    chroma_band = chroma_band + self.cross_color * 0.55 * detail * mod
if self.cross_luma > 1e-4:
    resid = chroma_band - remod_chroma(i, q, self.std, chroma_gain=1.0)
    y_band = y_band + self.cross_luma * 0.5 * resid * mod     # 默认 cross_luma = 0.3，【会执行】
return y_band.astype(np.float32), i.astype(np.float32), q.astype(np.float32)
```

**下一步：把 `cross_luma` 和 `cross_color` 设 0 再测 `_separate` 的返回。** 预期 `cross_luma`
那一行是嫌疑（它**加**一个残差到 `y_band`，而残差含 `y_band` 自己的高频——那可能是一个【自反馈】项）。

**注意**：之前"player 全部参数逐个中性化最大只变 0.0002"的结论与此**不矛盾**——那次是在【整条链的最终输出】上量的，
而这次是在 `_separate` 的**返回值**上量的。**验收标准：斜坡通过链路后接近 1.0。**

**下一步：在【真实函数内部】插探针，不用任何重建。**

**这一轮把两个猜想都证伪了，方法上的教训比结论更值钱。**

**猜想 A（cross_luma 自反馈）——已证伪。** 一个"逐语句复现 `_separate` 尾部"的脚本显示
`y_band = y_band + cross_luma*0.5*resid` 把 1.234 变成 0.777（−37%）。**但那个脚本是自己重建尾部、自己算 `resid` 的**，
它模拟的是【旧代码】。把真实函数改成读 `chroma_band_pre` 之后，**真实函数的返回值【一位不变】，仍是 0.802**
——所以 0.777 是那个重建的产物，不是真实调用。**改动已回退，`git diff` 为空。**

**猜想 B（读 `chroma_band` 而非 `chroma_band_pre` 才是原因）——同一实验否证。** 在真实调用里两者相等。

**所以现在的确定事实只有边界，没有机制**：

```
到达 _separate 的 composite   1.078   （真实，用 patch 抓的）
_separate 的返回值            0.802   （真实，用 patch 抓的）
其内部 yc_separate #1         1.078   （真实调用，hook 抓的）
内部 yc_separate #2           1.234   （真实调用，hook 抓的）
内部 demod/remod               很小    （正确，斜坡无彩度）
```

**而 1.234 → 0.802 这一段横跨【交叉项那几行】，可是把每一项单独或一起关掉都测不出变化。**
**唯一能解释的是：hook 记录的 `yc_separate #2` 返回值与真实函数在该点拿到的 `y_band` 不是同一个东西**
（`horizontal_shift` 用的 `disp` 在重建里被我强制成 0，而真实值来自 `_timebase_offset`）。

**下一步（唯一没做过的一步）：在【真实函数内部】把 `y_band` 的标准差打出来**——不是重建，
而是临时在 `player.py` 的 `_separate` 里加 `print(np.clip(y_band,0,1).std())` 于每个语句之后，跑一次斜坡，然后删掉。
**这不会有重建误差。**

**验收标准**：黑到白斜坡通过链路后接近 1.0（现在 0.800）。

## 方法上的铁律（这一轮又验证了一条）

**不要重建，要调用。** 我两次栽在这上面：
1. 手工用 YIQ 往返重建 composite → 得到 1.000/1.000/1.000/1.165，而真实是 1.078/1.078/…/1.233
2. 逐语句复现 `_separate` 尾部 → 得到 0.777，而真实函数返回 0.802（改前改后都一样）

**凡是"我照着代码写一遍再测"，结论都不成立。** 要么包真实调用（patch/hook），要么临时插入探针再删。

1. **链路实际传递的是 `meta["signal"]`，不是 `.rgb`**。`tools/composite_trace.py` 同时记三栏。
2. **`buf.composite` 全程是 `None`**（实测）——别追 composite。
3. **不要手工重建 composite**：我用 YIQ 往返重建后跑出的数是 1.000/1.000/1.000/1.165，而真实的是
   1.078/1.078/…/1.233。**重建描述的是另一个实验。** 用 `tools/separate_inside.py` 包真实调用。

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
