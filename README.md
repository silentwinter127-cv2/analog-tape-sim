# analog-tape-sim — 模拟磁带信号仿真工具

一个**以真实磁带记录、信号传输与解码过程为基础**的模拟信号仿真引擎，而不是一组 VHS 滤镜。

设计思路借鉴 3D 引擎：不是给图像叠一层「磁带感」，而是让画面真正通过一条虚拟的
**拍摄 → 编码 → 记录 → 读取 → 解码 → 显示** 信号链，VHS / Hi8 是这条链上不同参数与
不同记录机制的系统，而不是几个预设。


```bash
python -m atsim.web.server --port 8931 --open          # 启动 WebUI
python -m atsim.web.server --clip 我的录像.mp4          # 启动时载入素材
python tools/selftest.py                               # 引擎自检
python tools/measure_response.py                       # 频响标定测量
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

The order that works is: prove the instrument, then measure, then change code.
## 一、为什么它和「VHS 滤镜」不是一回事

### 1. 效果产生于信号机制，不是叠加

管线被拆成五个阶段，每个阶段处理的是**不同种类的信号**：

| 阶段 | 处理对象 | 意味着什么 |
| --- | --- | --- |
| 01 成像 | RGB（光） | 「当年是什么设备拍的」，发生在信号存在之前 |
| 02 编码 | RGB → **复合信号** | 亮度限带 + 色度欠载记录 + 真正把两者**相加** |
| 03 介质 | 复合信号 / 基带 | 时基误差、失落、折痕、磁头切换、颗粒噪声 |
| 04 读取 | 复合 → YIQ | **试图把加在一起的重新分开**，失败的部分就是串色 |
| 05 显示 | YIQ → RGB | CRT 扫描、伽马、过扫描、反光 |

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

同一盒带子在带 TBC 的机器上表现完全不同。五个层可以单独开关，用来回答
「这个特征是介质造成的，还是显示器造成的」。

---

## 二、五层参数

```
A 成像系统   传感器噪声/动态范围/曝光/白平衡/镜头锐度/色差/运动模糊
B 信号格式   亮度带宽(MHz)/色度带宽(MHz)/色度垂直平均/梳状滤波深度/串色/色度噪声
C 记录介质   走带稳定性/行抖动/低频漂移/垂直不稳定/磁头切换/失落密度与长度/磨损/折痕/噪声本底
D 播放设备   机芯抖动/TBC 开关与强度/残余抖动/行延迟/频响/磁头磨损/输出电平
E 显示设备   扫描线/隔行闪烁/辉光/失焦/几何畸变/伽马/色温/对比度/过扫描/暗角/反光
```

每个参数都带：单位、**物理合理区间**、默认值来源。界面在数值超出物理区间时会把它标成
琥珀色——**物理模型是底线，艺术化偏离是允许的，但要让你知道自己在偏离。**

带宽参数以 **MHz** 为单位（VHS 亮度约 3 MHz / S-VHS 约 5 MHz / VHS 色度约 0.7 MHz），
由视频标准的行频与有效采样数换算成像素域滤波，见 `atsim/core/standards.py`。

---

## 三、WebUI

- **左侧**：整套配置（摄像机+格式+介质+播放+显示 的组合）与单个设备档案
- **中间**：预览 + **A/B 对比滑块**，可拖动分隔线与原始素材对比；下方是信号链缓存状态
  与失落密度时间线
- **右侧**：按五层分组的参数面板，带单位、物理区间、来源说明
- **离线渲染**：完整分辨率逐帧渲染并用 ffmpeg 编码，保留音频

预览使用降分辨率代理 + **分阶段缓存**：只改显示层参数时，成像/编码/介质/读取的结果会被
复用，因此调 CRT 参数几乎不用重算重活。

---

### Start / stop

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

## 四、几个实现中真正重要（也踩过坑）的点

这些是工程上最容易做错、且错了会让整个仿真「看起来对但物理上不对」的地方：

1. **色度副载波不能做空间位移。** 一个像素是 0.42 个副载波周期，对载波做亚像素位移等于
   给每一行旋转色相（0.06 px ≈ 15°）。实测直接把色度相关性从 0.99 打到 -0.14，而亮度
   毫无影响。因此介质模型内部一律在**基带**（亮度/I/Q 平面）施加位移，最后再重新调制
   一次载波——这也正是色度欠载记录 + APC 锁相的真实行为。

2. **色度提取带通必须在载波处增益为 1。** 最初的带通把高通拐点放在 0.45×fsc，正好压在
   3.58 MHz 载波上，静悄悄吃掉了约 80% 的彩色。现在用**零相位 FFT 带通**并把载波处的
   增益标定为 1.0 再测量验证。

3. **梳状滤波器是匹配滤波器。** 同样的响应既抑制载波也损失载波幅度，恰好是 (1-2·depth)；
   不归一化时 depth=0.55 只剩 45% 的彩色。归一化之后，梳状深度才成为「爬行点 ↔ 垂直
   亮度细节」的纯粹权衡。

4. **色度解调前的限带是必需的。** 副载波在有效行内不是整数个半周期（NTSC 227.5 周期/行），
   所以 naive 解调（×2cos、低通、除 2）只能恢复约 53% 的 I 并且会串到 Q。先限带再解调，
   2f 项自然消失，恢复变成精确的。

5. **宽带噪声会被当成彩色。** 磁头切换和失落的噪声如果不过亮度带限，会落进色度通道并
   被重新调制为假彩色，实测把色度相关性从 0.98 打到 0.63。真实接收机在切换后会限带亮度。

6. **合成信号只作为传输格式。** 介质模型内部所有机制都在基带施加，载波只重建一次。

7. **输出必须是重建后的画面。** 一个隐藏很深的 bug：显示阶段被关闭时，输出曾经返回
   从摄像机一路传下来的**原始 RGB 缓冲**，于是格式层所有带宽/串色改动全部被静默丢弃。
   现在每一阶段的 RGB 输出都是它自己重建出来的画面。

8. **计划要惰性增长。** 早期版本用「帧数无上限」的渲染上下文去预分配整条磁带的时基
   过程数组，直接申请了 19 GB。现在 `_ensure(frame)` 按块增长，且因为底层过程是
   有状态的，增长出来的计划与一次性线性渲染完全一致。

---

## 五、目录结构

```
atsim/
  core/
    standards.py   NTSC/PAL 时间基（行频、副载波、有效采样数）——MHz↔像素的换算依据
    params.py      参数系统：单位、物理区间、艺术区间、来源
    profiles.py    五层参数表 + 媒介/播放机/显示/摄像机档案库 + 整套配置
    rng.py         确定性随机：种子派生、OU 过程、时空相关值噪声
    signal.py      YIQ、复合编码/分离/解调、FFT 带通、逐行位移、滤波器
    pipeline.py    Stage / Chain / FrameBuffer / RenderContext
  stages/
    camera.py      01 成像：噪声、动态范围、CCD 拖尾、色差、运动模糊
    format.py      02 编码：亮度带宽、色度欠载、边缘增强、色度噪声
    tape.py        03 介质：计划生成、时基、失落、折痕、磁头切换、噪声本底
    player.py      04 读取：TBC、梳状分离、串色、色同步误差、频响
    display.py     05 显示：扫描线、辉光、伽马、过扫描、几何、反光
  io/
    ffmpeg.py      ffmpeg/ffprobe 封装：探测、逐帧读取、编码、JPEG
    source.py      素材源：ClipSource（带 LRU）与 SyntheticSource（内置测试图）
  render/renderer.py  渲染器：分阶段哈希与缓存失效
  web/
    server.py     纯标准库 HTTP 服务：API、作业表、静态资源
    session.py    工程与渲染会话
    static/       WebUI（app.js 由 /api/schema 驱动生成全部控件）
tools/
  selftest.py         引擎自检（确定性、时域相关性、缓存失效、性能）
  testimage.py        合成「照片式」测试场景 + 预设对比图
  measure_response.py 频响标定测量
```

## 六、环境

仅需 **Python 3.11+、numpy、scipy、Pillow**，以及 PATH 上的 **ffmpeg/ffprobe**。
WebUI 不需要任何 Web 框架。
