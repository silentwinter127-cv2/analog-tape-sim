"""
The node graph.

The linear chain this project started with (`camera -> format -> tape -> player
-> display`) is a special case of a graph, and it is the wrong shape for the
thing users actually want to describe: an *apparatus*.  A real signal path is a
sequence of devices wired together, and the interesting questions are about the
wiring:

    camera -> encoder -> tape -> deck -> TV
    HDMI source -> AV converter -> RF modulator -> tuner -> TV
    camera -> encoder -> (two tapes, spliced) -> deck

So the pipeline here is a directed acyclic graph of typed nodes.  Three design
decisions matter more than the UI:

1. **Ports are typed, and the type is the format.**
   An edge carries a specific kind of signal (`rgb`, `composite`, `rf`,
   `dv`, `yiq`, `hdmi`) and a connection is only legal when the source's output
   type matches the sink's input type.  This is what makes the graph
   self-documenting and prevents nonsense wiring such as feeding a composite
   carrier into a DV encoder.  It is also why "HDMI-AV-RF-HDMI" can be expressed
   as a chain: each converter node translates one type into the next.

2. **The cache key is derived from the node AND everything upstream of it.**
   With a linear chain, hashing "the parameters of this stage" was enough.  In a
   graph it is not: changing a node in the middle must invalidate every node
   downstream *of that node*, and must not invalidate anything else -- including
   a sibling branch that does not depend on it.  `_node_hash` walks the ancestor
   set, so a graph with two parallel branches only re-renders the branch that
   was touched.  Getting this wrong is the classic node-editor bug where a
   parameter change appears not to take effect.

3. **Execution is topological and per-node cached.**
   Frames are memoised per node, so a 6-node path costs 6 passes the first time
   and only the downstream passes afterwards.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# signal types carried by edges
# ---------------------------------------------------------------------------
TYPES: dict[str, dict] = {
    "rgb": {
        "label": "RGB 画面",
        "note": "成像系统的输出：光，尚无任何视频编码",
    },
    "composite": {
        "label": "复合视频",
        "note": "亮度＋色度已调制的单一信号。只有复合格式的损伤机制对它有效",
    },
    "rf": {
        "label": "射频",
        "note": "经调制的 RF 载波（决定频道与邻频干扰），与复合基带不同",
    },
    "dv": {
        "label": "DV 数据流",
        "note": "数字分量：4:1:1/4:2:0 子采样＋DCT 量化后的码流",
    },
    "yiq": {
        "label": "YIQ 分量",
        "note": "分离后的亮度与色度，显示前的形态",
    },
    "hdmi": {
        "label": "HDMI",
        "note": "未压缩数字视频",
    },
    "audio": {"label": "音频", "note": "旁路用途"},
}


# ---------------------------------------------------------------------------
@dataclass
class Port:
    name: str
    type: str
    label: str = ""


@dataclass
class NodeSpec:
    """A node type: what it accepts, what it produces, what it can be told."""
    kind: str
    label: str
    group: str
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    device: str = ""            # physical device class, for provenance in the UI
    note: str = ""
    params_layer: str | None = None   # which schema layer configures it

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "label": self.label, "group": self.group,
            "device": self.device, "note": self.note,
            "params_layer": self.params_layer,
            "inputs": [{"name": p.name, "type": p.type, "label": p.label} for p in self.inputs],
            "outputs": [{"name": p.name, "type": p.type, "label": p.label} for p in self.outputs],
        }


def _p(name: str, type_: str, label: str = "") -> Port:
    return Port(name, type_, label or name)


# ---------------------------------------------------------------------------
# The node catalogue.
#
# This is a *device* catalogue, not a filter list: every node is something you
# could put on a bench and wire up.  Converters exist as first-class nodes
# precisely so that a consumer signal path (HDMI -> AV -> RF -> TV) can be
# expressed, which the old fixed chain could not do at all.
# ---------------------------------------------------------------------------
CATALOG: tuple[NodeSpec, ...] = (
    NodeSpec(
        kind="source", label="素材源", group="输入",
        inputs=(), outputs=(_p("out", "rgb", "画面"),),
        device="数字文件", note="把解码后的画面接入仿真链路",
        params_layer=None,
    ),
    NodeSpec(
        kind="camera", label="成像系统", group="设备",
        inputs=(_p("in", "rgb"),), outputs=(_p("out", "rgb"),),
        device="摄像机/镜头/传感器",
        note="镜头、传感器、曝光与色彩科学。发生在任何视频信号存在之前",
        params_layer="camera",
    ),
    NodeSpec(
        kind="encoder", label="信号编码", group="设备",
        inputs=(_p("in", "rgb"),), outputs=(_p("out", "composite"), _p("dv", "dv")),
        device="编码器 / 摄像机输出级",
        note="把画面变成可记录的信号：复合调制的限带与串扰，或 DV 的量化",
        params_layer="format",
    ),
    NodeSpec(
        kind="dv_codec", label="DV 编解码", group="设备",
        inputs=(_p("in", "dv"),), outputs=(_p("out", "dv"),),
        device="MiniDV / DVCAM 记录器",
        note="4:1:1 或 4:2:0 子采样、8x8 DCT、按宏块分配码率的量化",
        params_layer="codec",
    ),
    NodeSpec(
        kind="tape", label="记录介质", group="设备",
        inputs=(_p("in", "composite"), _p("dv", "dv")),
        # The medium is transparent to *format*: it damages whatever was
        # recorded on it and hands the same kind of stream to the player.  A
        # MiniDV cassette therefore outputs `dv`, and it is the deck that turns
        # that into a viewable signal.
        outputs=(_p("out", "composite"), _p("dv", "dv")),
        device="磁带 / 磁头",
        note="时基误差、失落、折痕、磁头切换、堵塞与划伤。对复合信号与 DV 码流施加不同的损伤机制",
        params_layer="tape",
    ),
    NodeSpec(
        kind="deck", label="播放设备", group="设备",
        inputs=(_p("in", "composite"), _p("dv", "dv")),
        # A deck reads whichever medium it is given and outputs a decoded
        # component signal, so its output type is always YIQ regardless of
        # branch.  This is what lets one node serve both the analog and the
        # digital path without the graph needing format-specific plumbing.
        outputs=(_p("out", "yiq"),),
        device="录像机 / 摄像机回放 / TBC",
        note="读取、时基校正、解码与输出。复合分支做梳状分离，DV 分支做解码",
        params_layer="player",
    ),
    NodeSpec(
        kind="display", label="显示设备", group="设备",
        inputs=(_p("in", "yiq"),), outputs=(_p("out", "rgb"),),
        device="CRT / 监视器",
        note="扫描、光栅、伽马、过扫描、几何、屏幕反光",
        params_layer="display",
    ),
    # --- converters: how a consumer signal path is expressed -------------
    NodeSpec(
        kind="hdmi_out", label="HDMI 输出", group="转换",
        inputs=(_p("in", "rgb"),), outputs=(_p("out", "hdmi"),),
        device="播放器/电脑 HDMI 口",
        note="未压缩数字输出。到这里为止没有任何模拟损伤",
        params_layer=None,
    ),
    NodeSpec(
        kind="hdmi_to_av", label="HDMI→AV 转换器", group="转换",
        inputs=(_p("in", "hdmi"),), outputs=(_p("out", "composite"),),
        device="消费级转换盒",
        note="把数字画面编码成复合基带。廉价转换盒自己的限带与色度处理就在这里引入",
        params_layer="format",
    ),
    NodeSpec(
        kind="rf_modulator", label="RF 调制器", group="转换",
        inputs=(_p("in", "composite"),), outputs=(_p("out", "rf"),),
        device="射频调制器",
        note="把复合基带搬到 VHF/UHF 载波上：频道宽度、图像载波位置、邻频差拍都在这里",
        params_layer="rf",
    ),
    NodeSpec(
        kind="rf_tuner", label="RF 调谐器", group="转换",
        inputs=(_p("in", "rf"),), outputs=(_p("out", "composite"),),
        device="电视/录像机高频头",
        # The RF layer belongs to the tuner as much as to the modulator: received
        # level only exists at the receiving end.  Both nodes expose it, so the
        # parameters are reachable from either one on the canvas.  Previously
        # neither did, which left the entire RF layer unreachable from the UI --
        # the physics worked, but there was no way to dial it in.
        note="解调回基带：载噪比、调谐器噪声系数、行 AFC 失锁与邻频干扰",
        params_layer="rf",
    ),
)

CATALOG_BY_KIND: dict[str, NodeSpec] = {n.kind: n for n in CATALOG}


# ---------------------------------------------------------------------------
@dataclass
class Node:
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)
    enabled: bool = True
    # True once the position has been set in pixels (by a drag) rather than in
    # grid units (by a preset).  Without this the canvas cannot tell the two
    # apart on reload and re-multiplies a dragged coordinate.
    px_position: bool = False

    @property
    def spec(self) -> NodeSpec:
        return CATALOG_BY_KIND[self.kind]


@dataclass
class Edge:
    src: str
    src_port: str
    dst: str
    dst_port: str


@dataclass
class Issue:
    severity: str          # "error" | "warning"
    message: str
    node: str | None = None

    def to_dict(self) -> dict:
        return {"severity": self.severity, "message": self.message, "node": self.node}


# ---------------------------------------------------------------------------
class Graph:
    """A validated DAG of device nodes."""

    def __init__(self, nodes: Iterable[Node] = (), edges: Iterable[Edge] = (),
                 seed: int = 20240101):
        self.nodes: dict[str, Node] = {n.name: n for n in nodes}
        self.edges: list[Edge] = list(edges)
        self.seed = int(seed)

    # -- construction ---------------------------------------------------
    def add(self, name: str, kind: str, params: dict | None = None,
            position: tuple[float, float] | None = None) -> Node:
        if kind not in CATALOG_BY_KIND:
            raise KeyError(f"unknown node kind {kind!r}")
        n = Node(name=name, kind=kind, params=dict(params or {}),
                 position=position or (0.0, 0.0))
        self.nodes[name] = n
        return n

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> Edge:
        e = Edge(src, src_port, dst, dst_port)
        self.edges.append(e)
        return e

    def disconnect(self, dst: str, dst_port: str) -> None:
        self.edges = [e for e in self.edges
                      if not (e.dst == dst and e.dst_port == dst_port)]

    def remove(self, name: str) -> None:
        self.nodes.pop(name, None)
        self.edges = [e for e in self.edges if e.src != name and e.dst != name]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "nodes": [
                {"name": n.name, "kind": n.kind, "params": n.params,
                 "position": list(n.position), "enabled": n.enabled,
                 # once a node has been dragged its position is in pixels, not
                 # grid units; the canvas needs to know which, or it would
                 # re-scale an already-scaled coordinate on every reload
                 "px": bool(getattr(n, "px_position", False))}
                for n in self.nodes.values()
            ],
            "edges": [
                {"src": e.src, "src_port": e.src_port,
                 "dst": e.dst, "dst_port": e.dst_port}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls(seed=int(d.get("seed", 20240101)))
        for nd in d.get("nodes", []):
            g.add(nd["name"], nd["kind"], nd.get("params"),
                  tuple(nd.get("position", (0.0, 0.0))))
            node = g.nodes[nd["name"]]
            node.enabled = bool(nd.get("enabled", True))
            node.px_position = bool(nd.get("px", False))
        for ed in d.get("edges", []):
            g.connect(ed["src"], ed["src_port"], ed["dst"], ed["dst_port"])
        return g

    # -- queries --------------------------------------------------------
    def upstream(self, name: str) -> list[str]:
        """Direct predecessors of a node."""
        return [e.src for e in self.edges if e.dst == name]

    def ancestors(self, name: str) -> set[str]:
        """Every node that the given node's output depends on."""
        seen: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            for up in self.upstream(cur):
                if up not in seen:
                    seen.add(up)
                    stack.append(up)
        return seen

    def downstream(self, name: str) -> set[str]:
        out: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            for e in self.edges:
                if e.src == cur and e.dst not in out:
                    out.add(e.dst)
                    stack.append(e.dst)
        return out

    def inputs_of(self, name: str) -> dict[str, Edge]:
        return {e.dst_port: e for e in self.edges if e.dst == name}

    def order(self) -> list[str]:
        """Topological order (Kahn).  Raises if the graph has a cycle."""
        indeg: dict[str, int] = {n: 0 for n in self.nodes}
        for e in self.edges:
            if e.dst in indeg:
                indeg[e.dst] += 1
        queue = [n for n in self.nodes if indeg[n] == 0]
        out: list[str] = []
        while queue:
            cur = queue.pop(0)
            out.append(cur)
            for e in self.edges:
                if e.src != cur or e.dst not in indeg:
                    continue
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    queue.append(e.dst)
        if len(out) != len(self.nodes):
            raise ValueError("graph has a cycle")
        return out

    # -- validation -----------------------------------------------------
    def validate(self) -> list[Issue]:
        issues: list[Issue] = []
        try:
            self.order()
        except ValueError as exc:
            issues.append(Issue("error", str(exc)))
            return issues

        for name, node in self.nodes.items():
            spec = node.spec
            ins = self.inputs_of(name)
            # every required input must be filled, and with the right type
            allowed = {p.name: p.type for p in spec.inputs}
            for port_name in ins:
                if port_name not in allowed:
                    issues.append(Issue("error", f"节点没有输入端口 {port_name!r}", name))
            for p in spec.inputs:
                e = ins.get(p.name)
                if e is None:
                    # a missing optional input is only fatal when nothing else
                    # is connected at all
                    if not ins:
                        issues.append(Issue("warning", f"输入 {p.label} 未连接", name))
                    continue
                src_spec = self.nodes[e.src].spec if e.src in self.nodes else None
                if src_spec is None:
                    issues.append(Issue("error", f"上游节点 {e.src!r} 不存在", name))
                    continue
                outs = {q.name: q.type for q in src_spec.outputs}
                if e.src_port not in outs:
                    issues.append(Issue("error",
                                        f"{e.src} 没有输出端口 {e.src_port!r}", name))
                    continue
                src_type = outs[e.src_port]
                if src_type != p.type:
                    issues.append(Issue(
                        "error",
                        f"类型不匹配：{e.src}.{e.src_port} 是 {TYPES[src_type]['label']}，"
                        f"而 {name}.{p.name} 需要 {TYPES[p.type]['label']}",
                        name))
            # a disabled node still needs its inputs to exist structurally,
            # but contributes nothing downstream -- that is a warning, not error
            if not node.enabled and self.downstream(name):
                issues.append(Issue("warning", "该节点已关闭，下游将收到未处理的信号", name))

        # a graph with no sink produces nothing
        sinks = [n for n in self.nodes if not self.downstream(n)]
        if not sinks:
            issues.append(Issue("error", "图里没有输出节点"))
        return issues

    def output_nodes(self) -> list[str]:
        return [n for n in self.nodes if not self.downstream(n)]

    # -- cache keys -----------------------------------------------------
    def _params_of(self, name: str) -> dict:
        node = self.nodes[name]
        return {"kind": node.kind, "params": node.params, "enabled": node.enabled}

    def node_hashes(self) -> dict[str, str]:
        """
        A stable hash per node over itself AND all of its ancestors.

        This is the property that makes a node editor usable: touching a node
        invalidates exactly its own subtree and nothing else.  Sibling branches
        keep their caches.
        """
        hashes: dict[str, str] = {}
        for name in self.order():
            payload: dict[str, Any] = {"seed": self.seed}
            payload["self"] = self._params_of(name)
            for up in sorted(self.ancestors(name)):
                payload[f"up:{up}"] = self._params_of(up)
                payload[f"uph:{up}"] = hashes.get(up, "")
            hashes[name] = hashlib.blake2s(
                json.dumps(payload, sort_keys=True, default=str).encode(),
                digest_size=10).hexdigest()
        return hashes

    def describe(self) -> dict:
        h = self.node_hashes()
        return {
            "order": self.order(),
            "nodes": [
                {**n.spec.to_dict(),
                 "name": n.name,
                 "enabled": n.enabled,
                 # position and per-node overrides must be carried here: the
                 # canvas draws from `describe()` alone, and an earlier version
                 # omitted position, so every edge failed to resolve its
                 # endpoints while the API still looked perfectly healthy.
                 "position": [float(n.position[0]), float(n.position[1])],
                 "px": bool(getattr(n, "px_position", False)),
                 "params": dict(n.params),
                 "hash": h.get(n.name, "")[:8],
                 "upstream": sorted(self.upstream(n.name))}
                for n in self.nodes.values()
            ],
            "edges": [
                {**e.__dict__, "type": self._edge_type(e)} for e in self.edges
            ],
            "outputs": self.output_nodes(),
            "types": TYPES,
        }

    def _edge_type(self, e: Edge) -> str:
        spec = self.nodes.get(e.src)
        if spec is None:
            return "?"
        for p in spec.spec.outputs:
            if p.name == e.src_port:
                return p.type
        return "?"


# ---------------------------------------------------------------------------
def default_graph() -> Graph:
    """The classic consumer camcorder -> tape -> TV path, as a graph."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", position=(3, 0))
    g.add("deck", "deck", position=(4, 0))
    g.add("display", "display", position=(5, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


def minidv_graph() -> Graph:
    """A MiniDV camcorder: the digital branch, no composite carrier anywhere."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("codec", "dv_codec", position=(3, 0))
    g.add("tape", "tape", position=(4, 0))
    g.add("deck", "deck", position=(5, 0))
    g.add("display", "display", position=(6, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    # the DV branch leaves the encoder through its `dv` output, not `out`,
    # and the deck reads it from its matching `dv` input
    g.connect("encoder", "dv", "codec", "in")
    g.connect("codec", "out", "tape", "dv")
    g.connect("tape", "dv", "deck", "dv")
    g.connect("deck", "out", "display", "in")
    return g


def rf_loop_graph() -> Graph:
    """
    A consumer signal path: HDMI -> AV -> RF -> tuner -> TV.

    This is the wiring the old fixed chain could not express at all, and it is
    also the case where the *converters themselves* matter: each box adds its
    own encoding loss, so the picture degrades in steps that belong to specific
    devices rather than to "the effect".
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("hdmi", "hdmi_out", position=(2, 0))
    g.add("av", "hdmi_to_av", position=(3, 0))
    g.add("mod", "rf_modulator", position=(4, 0))
    g.add("tuner", "rf_tuner", position=(5, 0))
    g.add("deck", "deck", position=(6, 0))
    g.add("display", "display", position=(7, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "hdmi", "in")
    g.connect("hdmi", "out", "av", "in")
    g.connect("av", "out", "mod", "in")
    g.connect("mod", "out", "tuner", "in")
    # the tuner recovers composite baseband; a deck (or the TV's own decoder)
    # is what separates it into a component signal for the tube
    g.connect("tuner", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


GRAPH_PRESETS: dict[str, dict] = {
    "vhs_consumer": {
        "label": "VHS 消费级链路",
        "note": "摄像机 → 编码 → 磁带 → 录像机 → CRT",
        "build": default_graph,
    },
    "minidv": {
        "label": "MiniDV 便携机链路",
        "note": "走 DV 数字分支，全程没有复合载波",
        "build": minidv_graph,
    },
    "minidv_shuttle": {
        "label": "MiniDV 快进/倒带搜索",
        "note": "磁头掠过磁迹的速度与记录速率不符，只能取到部分宏块，"
                "解码器用缓冲区旧帧补齐——成片的、带锯齿边缘的方块残留。"
                "把「走带倍速」调回 0 即恢复正常播放",
        "build": minidv_graph,
        "params": {"codec": {"dv_enabled": 1.0},
                   "tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                            "shuttle_cluster": 0.55, "shuttle_carry": 0.45}},
    },
    "rf_loop": {
        "label": "HDMI→AV→RF→电视（信号偏弱）",
        "note": "消费级信号回路：每一级转换器各引入一次损失。默认接收电平偏低，"
                "便于看到 RF 特有的撕裂条纹；把「接收电平」调到 −55 dBm 以上即变干净",
        "build": rf_loop_graph,
        "params": {"rf": {"signal_dbm": -74.0, "adjacent_channel": 0.15}},
    },
}
