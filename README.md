# analog-tape-sim — 模拟磁带信号仿真工具

一个**以真实磁带记录、信号传输与解码过程为基础**的模拟信号仿真引擎，而不是一组 VHS 滤镜。

设计思路借鉴 3D 引擎：不是给图像叠一层「磁带感」，而是让画面真正通过一条虚拟的
**拍摄 → 编码 → 记录 → 读取 → 解码 → 显示** 信号链。VHS / Hi8 / MiniDV / Digital8 / VCD
是这条链上不同参数、不同记录机制与不同编码的系统，而不是几个预设。

```bash
python server.py start --open                # 启动 WebUI（默认 8947）并打开浏览器
python server.py start --clip 我的录像.mp4    # 启动时载入素材
python tools/verify.py                       # 全量验证（64 个检查函数 / 205 条断言）
```

---

## Rule one: suspect the instrument first

Before changing a mechanism because it "has no effect", validate the measurement. This is
written down because it has been learned repeatedly and at cost:

| What was concluded | What was true |
|---|---|
| `row_timebase` is inert, not wired | the harness dropped the parameter via `merge_values` |
| `binder_loss` changes saturation the wrong way | the check measured in RGB, where clipping ate the evidence |
| `luma_pedestal` points the wrong way | the target frame was fed back in, so the measurement compared damage with damage |
| `chroma dies first` fails, tune the mechanism | the mixing law is a step; no partial population exists to observe |
| the sweep says `row_chroma_error` does nothing | the sweep harness was wrong; it moves the metric by 0.0252 |
| broadcast noise is inert | the tracker's own floor was above the effect |
| the lens chain is gone (`0 calls`) | the regex matched `self.lens`, the code passes `eff_lens` |
| `App` has no audio API | it did — the check had been written before the method existed |

Every one of those was a measurement fault first and a mechanism fault only sometimes. The
costs are asymmetric:

* **concluding "mechanism broken" when the measurement is broken** leads to editing working
  code, and the real fault is never found because it is never looked at;
* **concluding "measurement broken" when the mechanism is broken** costs one round.

So the default is to suspect the instrument. Concretely, before concluding a mechanism does
nothing:

1. **assert the parameter arrived** where it is used, rather than trusting the call;
2. **validate the instrument against a case whose answer is known** — this is what resolved
   the hue-spread contradiction and what showed the displacement tracker was non-linear;
3. **differencing against the mechanism switched off**, never against the original signal,
   which mixes every other mechanism in;
4. **check the metric can move at all** — a metric that saturates reports "no effect" for
   everything.

The order that works is: **prove the instrument, then measure, then change code.**

---

## 一、为什么它和「VHS 滤镜」不是一回事

### 1. 效果产生于信号机制，不是叠加

管线被拆成**八个阶段**，每个阶段处理**不同种类的信号**：

| 阶段 | 处理对象 | 意味着什么 |
| --- | --- | --- |
| `camera` 01 成像 | RGB（光） | 「当年是什么设备拍的」，发生在信号存在之前 |
| `format` 02 编码 | RGB → **复合信号** | 亮度限带 + 色度欠载 + 真正把两者**相加** |
| `codec` 02b 数字编码 | RGB → **8×8 DCT 码流** | DV / VCD(MPEG-1)：块状、振铃、码率天花板 |
| `connector` 02c 接口 | 信号 + 接口损伤 | HDMI / AV / S-Video / RF 的**机械与电气**劣化 |
| `tape` 03 介质 | 复合信号 / 基带 | 时基误差、失落、折痕、磁头切换、颗粒噪声 |
| `player` 04 读取 | 复合 → YIQ | **试图把加在一起的重新分开**，失败的部分就是串色 |
| `display` 05 显示 | YIQ → RGB → 光 | CRT 扫描、束斑、荧光粉、玻壳、会聚 |
| `overlay` 06 叠加 | OSD 字形 | 字符发生器的固定格子、ROM 字体 |

关键点：**交叉色彩（彩边）与交叉亮度（爬行点）不是在 04 阶段"加"上去的，而是因为在 02
阶段真的把亮度和色度相加过，04 阶段分离不干净才出现的。** 把 02 阶段的串扰参数调到 0，
彩边就会自己消失——因为它本来就是分离残差。

### 2. 时基误差是逐行采样相位误差，不是画面抖动

水平位移被建模为**逐扫描行的采样相位误差**，由一条连续过程生成：

- 相位误差是一个「速度类过程」的**积分**，所以相邻行的差值是平滑的；
  相邻两行在磁带上的读取间隔只有几十微秒，不可能各自独立随机。
- 结果是竖线边缘的锯齿状波动与低频摆动，而不是整幅画面在抖。

### 3. 介质磨损是离散事件，不是全画面加噪

失落、折痕都是**有开始时间、持续时间、位置和尺寸的事件**，由种子确定性地生成一次，
并作为**缺陷时间线**发布到界面。同一工程重复渲染得到完全相同的影像。

### 4. 噪声不会每帧重新随机

磁带上有一个固定的颗粒纹理，叠加缓慢漂移的时间分量；折痕连续移动；一次失落会持续它
整个时长。自检里有一项专门验证「帧间变化平滑而不是逐帧乱跳」。

### 5. 磁带、播放机、显示器是三个独立环节

同一盒带子在带 TBC 的机器上表现完全不同。各层可以单独开关，用来回答
「这个特征是介质造成的，还是显示器造成的」。

### 6. 数字编码不是"降质滤镜"，是另一条支路

DV 用 4:1:1 / 4:2:0、8×8 DCT、固定码率；VCD 用 MPEG-1。它们的伪影形状（块状、振铃、
色度块）与模拟链路完全不同，因此是**并联的分支**而不是叠加在模拟链上。
`atsim/codecs/dv.py` 与 `mpeg1.py` 各自是独立实现，`codec` 层决定走哪条。

---

## 二、十层、342 个参数

| 层 | 参数 | 内容 |
| --- | --- | --- |
| `camera` | **100** | 传感器噪声/动态范围/曝光/白平衡/镜头锐度/像差/运动模糊/夜视/自动控制回路 |
| `format` | 16 | 亮度带宽(MHz)/色度带宽(MHz)/色度垂直平均/色度时间残留 C(t-1)/梳状滤波/串色 |
| `codec` | 11 | DV 与 VCD(MPEG-1)：量化步长、色度格式、码率、8×8 块 |
| `rf` | 26 | RF 调制 / 调谐器链路：频道、伴音、同步、AGC、弱信号、邻频 |
| `tape` | 59 | 走带稳定性/行抖动/低频漂移/垂直不稳定/磁头切换/失落/磨损/折痕/噪声本底/粘滞 |
| `connector` | 17 | HDMI / AV / S-Video / RF：受潮、腐蚀、接地环、阻抗失配 |
| `player` | 24 | 机芯抖动/TBC/残余抖动/行延迟/频响/磁头磨损/输出电平 |
| `display` | 36 | 扫描线/束斑/荧光粉/玻壳光晕/会聚/EHV 下垂/伽马/色温/过扫描/暗角/反光 |
| `overlay` | 36 | OSD：字体、格子、位置、阴影、行距、时间码 |
| `audio` | 17 | 磁带音频：走带耦合、抖晃、粘滞、磁头、信噪比、A/V 同步 |

**五层里只有一条参数是关于时间的**：`format.chroma_lag`（色度时间残留）。色度通道的带宽只有亮度的
几分之一，窄带通道的冲激响应本来就横跨若干帧——所以颜色比亮度晚到、并在物体移走后留下拖影。
它按帧递归（等效一阶 IIR，尾巴按系数衰减），**因此要求按顺序推进**：第一帧、以及任何参数改动后
的第一帧没有历史，此时该机制**精确等于关闭**而不是近似。

每个参数都带：单位、**物理合理区间**、默认值来源。界面在数值超出物理区间时会把它标成
琥珀色——**物理模型是底线，艺术化偏离是允许的，但要让你知道自己在偏离。**

带宽参数以 **MHz** 为单位（VHS 亮度约 3 MHz / S-VHS 约 5 MHz / VHS 色度约 0.7 MHz），
由视频标准的行频与有效采样数换算成像素域滤波，见 `atsim/core/standards.py`。
**一个含义随分辨率的像素数而变的参数不是参数**——所以带宽一律以 MHz 表达，材质格子一律以像素表达。

---

## 三、机制模块：物理不在某一层里的部分

`atsim/codecs/` 有 **18 个模块、约 471 KB**，承载那些无法归入单一 stage 的物理：

| 模块 | 内容 |
| --- | --- |
| `nightvision.py` | 夜视模式：IR 截止滤除 → CCD 近红外响应 → 单色 → 绿显示矩阵；含材质反射率表 |
| `lens.py` | 镜头与像差：色散、场曲、纵向色差、coma、像散、球差、镀膜鬼影 |
| `rf.py` | RF 调制/调谐器：残留边带、同步分离、AGC、弱信号噪底、邻频干扰 |
| `connectors.py` | 接口劣化：受潮与腐蚀的阻抗变化、数字链路的"要么对要么错" |
| `crt.py` | CRT 物理：束斑发散、荧光粉分色、玻壳光晕、会聚误差、EHV 下垂、余辉 |
| `osd.py` + `fonts.py` | 字符发生器与字体注册表（**28 种字体**、格子制、脚本覆盖规则） |
| `audio.py` | 磁带音频：与视频链**共享同一份** `TapePlan`/`RFLink` 状态，保证音画同步 |
| `dv.py` / `mpeg1.py` | DV 与 MPEG-1 编码支路 |
| `autocontrol.py` | 摄像机的 AE / AWB / AF 回路（第二条时间机制） |
| `operator.py` | 操作者：手持、变焦电机、使用习惯 |
| `stiction.py` / `tape_speed.py` | 走带粘滞（间歇性故障）、SP/LP/EP 物理 |
| `material.py` | 近红外材质标注（本地视觉模型） |
| `analog_iir.py` / `optics.py` | 模拟 IIR 滤波器、光学与传感器机制 |

---

## 四、WebUI

- **节点接线画布**：把「素材 → 成像 → 编码 → 码流/接口 → 介质 → 读取 → 显示 → 叠加」
  画成可连线的节点图，**13 种节点**；未连接的转换器会得到可读的错误，而不是静默失效
- **左侧**：整套配置（41 个 factory preset）与 13 个设备档案（Digital8 / Hi8 / VHS / CCTV / VCD…）
- **中间**：预览 + **A/B 对比滑块**；下方是信号链缓存状态与缺陷时间线
- **右侧**：按层分组的参数面板，带单位、物理区间、来源说明
- **音频**：试听、导出 WAV、A/V 同步包络
- **离线渲染**：完整分辨率逐帧渲染并用 ffmpeg 编码，保留音频

预览使用降分辨率代理 + **分阶段缓存**：只改显示层参数时，前面各层的结果会被复用，
因此调 CRT 参数几乎不用重算重活。

**预览与导出走不同分辨率的渲染器**，所以两者出现差异时，先分离分辨率效应再判断是链路问题。

---

## 五、Start / stop

Use `server.py` rather than starting the server by hand. It records the PID it
created and `stop` signals only that PID. That matters: stopping the server by name
(`Get-Process python | Stop-Process`) has already killed unrelated Python processes
on this machine, including a server the user was running themselves.

```bash
python server.py start          # start (default port 8947)
python server.py start --open   # start and open a browser
python server.py status         # process, port, whether it answers
python server.py stop           # stop only the server this tool started
python server.py restart
python server.py logs           # tail the server log

python server.py start --port 9000 --data-dir runs
```

On Windows, double-clicking **`server.bat`** is equivalent to `start --open`;
`server.bat stop` and `server.bat status` also work.

PID and log live in `.server/`:

| file | contents |
|---|---|
| `.server/pid.json` | recorded PID, port, start time |
| `.server/server.log` | server output (read by `logs`) |

`status` also names any port that answers HTTP but was *not* started by this tool,
so another server is not mistaken for this one.

---

## 六、几个实现中真正重要（也踩过坑）的点

这些是工程上最容易做错、且错了会让整个仿真「看起来对但物理上不对」的地方：

1. **色度副载波不能做空间位移。** 一个像素是 0.42 个副载波周期，对载波做亚像素位移等于
   给每一行旋转色相（0.06 px ≈ 15°）。实测直接把色度相关性从 0.99 打到 -0.14，而亮度
   毫无影响。因此介质模型内部一律在**基带**（亮度/I/Q 平面）施加位移，最后再重新调制
   一次载波——这也正是色度欠载记录 + APC 锁相的真实行为。

2. **色度提取带通必须在载波处增益为 1。** 最初的带通把高通拐点放在 0.45×fsc，正好压在
   3.58 MHz 载波上，静悄悄吃掉了约 80% 的彩色。现在用**零相位 FFT 带通**并把载波处的
   增益标定为 1.0 再测量验证。

3. **梳状滤波器是匹配滤波器。** 同样的响应既抑制载波也损失载波幅度，恰好是 (1-2·depth)；
   不归一化时 depth=0.55 只剩 45% 的彩色。

4. **色度解调前的限带是必需的。** 副载波在有效行内不是整数个半周期（NTSC 227.5 周期/行），
   所以 naive 解调（×2cos、低通、除 2）只能恢复约 53% 的 I 并且会串到 Q。

5. **宽带噪声会被当成彩色。** 磁头切换和失落的噪声如果不过亮度带限，会落进色度通道并
   被重新调制为假彩色，实测把色度相关性从 0.98 打到 0.63。

6. **输出必须是重建后的画面。** 一个隐藏很深的 bug：显示阶段被关闭时，输出曾经返回
   从摄像机一路传下来的**原始 RGB 缓冲**，于是格式层所有带宽/串色改动全部被静默丢弃。

7. **计划要惰性增长。** 早期版本用「帧数无上限」的渲染上下文去预分配整条磁带的时基
   过程数组，直接申请了 19 GB。现在 `_ensure(frame)` 按块增长，且因为底层过程是
   有状态的，增长出来的计划与一次性线性渲染完全一致。

8. **一条关于时间的机制，不能用「糊了没有」来验证。** 它产生的每一个量——颜色横向扩散、
   边缘变软、彩色光晕——**一条水平低通也照样产生**。能区分的只有：**把源图在某一帧单独
   改变一次，然后看后面的帧**。一帧的颜色脉冲对空间滤波器等于零，所以闪光之后任一帧里
   出现这个颜色，**只可能来自跨帧记忆**。
   **而做出这条判据的仪器本身错了四次**，每次都是同一个原因：两个场景的差，只有在「闪光」
   是唯一差异时才是在测闪光。**仪器出错时给出的数字是有量纲、有小数位、看起来很正常的。**

9. **一个参数被在两层同时绑定，改一层就毫无变化。** 这个缺陷在本仓库出现过七次：
   schema 没声明（`resolve_all` 丢弃该键）、模式预设用 `dataclasses.replace` 无条件覆盖、
   stage 的 `cfg.get(k, 默认)` 与 schema 默认值不一致。**改一个参数没反应时，先查它绑了几处。**

10. **绿色显示不是滤镜，是投影。** NightShot 把单色信号送到绿色荧光粉，输出的通道比例
    **就是**显示三元组，与输入无关；而 `green_mix = 0` 时**原样返回彩色画面**——VHS/MiniDV
    这类彩色格式的色度副载波承载的是真实彩色，不该被单色化。

---

## 七、目录结构

```
atsim/
  core/
    standards.py   NTSC/PAL 时间基（行频、副载波、有效采样数）——MHz↔像素的换算依据
    params.py      参数系统：单位、物理区间、艺术区间、来源
    profiles.py    342 个参数的定义 + 41 个 factory preset + 设备档案
    rng.py         确定性随机：种子派生、OU 过程、时空相关值噪声
    signal.py      YIQ、复合编码/分离/解调、FFT 带通、逐行位移、滤波器
    pipeline.py    Stage / Chain / FrameBuffer / RenderContext
  stages/          八个阶段，每个处理不同种类的信号
    camera.py format.py codec.py connector.py tape.py player.py display.py overlay.py
  codecs/          18 个机制模块（见第三节）
  graph/           节点图：13 种节点、连线、执行器
  io/              ffmpeg 封装、素材源（ClipSource / SyntheticSource）
  render/          渲染器：分阶段哈希与缓存失效
  web/             纯标准库 HTTP 服务、会话、静态资源（app.js 由 /api/schema 驱动）
tools/
  verify.py        全量验证：64 个检查函数 / 205 条断言（唯一的验收标准）
  selftest.py      引擎自检（确定性、时域相关性、缓存失效、性能）
  audit_params.py  每个参数的区间两端是否都改变画面
  audit_reads.py   schema 的键到底有没有被读
  audit_temporal.py 帧间行为
  repro_render.py  在服务器外复现 /api/render，异常全文可见
  （共 172 个脚本：标定、测量、出图、原型）
profiles/          13 个设备档案（JSON）
docs/              设计记录
```

---

## 八、环境

仅需 **Python 3.11+、numpy、scipy、Pillow**，以及 PATH 上的 **ffmpeg/ffprobe**。
WebUI 不需要任何 Web 框架。

可选：本地视觉语言模型（用于近红外材质标注）；**没有它时模型会退化为按亮度估计**，
功能仍然可用。

**字体**：OSD 使用系统字体，缺失时按脚本能力降级到 `%WINDIR%\Fonts`（Windows）或
`/usr/share/fonts`（Linux）。拉丁字体不会拿到 CJK 回退——因为 `osd` 不能画中文是**被断言的行为**。

---

## 九、当前状态

`tools/verify.py` 是唯一的验收标准。写这份 README 时：

```
183 passed, 6 failed
```

六项失败**全部集中在正在开发的新功能里**（CRT 新参数、VCD 色彩评分、RF 噪底标定、
一处被删掉却仍声明的参数），模拟磁带链路本身的那部分检查是绿的。**这些失败现在会明确
报错，而不会让整个套件崩掉**——这是最近修复的重点：此前的三次崩溃（`FileNotFoundError`、
两处 `AttributeError`）让套件**没有任何汇总行**，把后面的失败全遮住了。

**已知的系统性风险**：

- 参数"存在于一层而不存在于另一层"是本仓库最常见的缺陷形态，`audit_params.py` /
  `audit_reads.py` 是为此写的。当前 `16:9 aspect squeeze` 与 `stage params exist` 两条失败
  就属于这一类：**参数还在 schema 里，读它的代码没了**；
- `tools/` 有 172 个脚本，其中相当一部分是一次性的标定/原型，不是维护对象；
- 预览与导出使用不同分辨率的渲染器，跨分辨率比较前必须先统一尺寸；
- 源码行尾不统一（`HEAD` 中部分文件是 CRLF、部分是 LF），并行编辑时容易把整文件读成
  "全部重写"。加一个 `.gitattributes` 可以消除这类噪音。

交接细节、待办清单与每一处判断的依据见 **`HANDOFF.md`**；历史缺陷与复核请求见 **`REVIEW.md`**。
