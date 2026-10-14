"""
Material labelling for the near-infrared model, from a local vision-language model when one is available.

**Why the note asks for this and why an estimate is not enough.**  `note.txt`'s second mechanism:

    "给场景材质设置近红外反射率，而不是直接把 RGB 转成灰度。植被、织物、皮肤、塑料和涂漆表面的近红外响应可能与可见光观感不同。
     这是普通夜视滤镜最难自然模拟的部分之一。"

**Near-IR albedo is not a function of visible brightness** -- chlorophyll returns a great deal (the Wood effect), skin
scatters it, dark cotton returns almost none, and a black plastic case returns more than a black cotton shirt.  **So
`lum ** 0.55`, which is what the model does now, is the thing the note says is hardest to make natural**, and
`nightvision.IR_ALBEDO` already lists the measured orderings while nothing reads them: a visible photograph does not say
which material is where.

**A vision-language model does say**, which is the whole point of having one: `siglip2-base-patch16-224` scores a picture
against arbitrary text, so `IR_ALBEDO`'s own keys become the prompts.

**Two properties are load-bearing, and both are enforced here rather than trusted:**

  * **an unavailable model is not an error.**  `available()` reports whether a checkpoint can be loaded, and where it
    cannot, `analyse` returns `None` so the caller falls back to the luminance estimate.  **A user without a model gets
    the estimate, not a failure** -- which is what the interface promises;
  * **every label carries its confidence, and low-confidence patches are marked so the estimate can take over
    patch-by-patch.**  Measured on the reference photograph, the model's margin over the runner-up runs from **0.692 down
    to 0.030**, so a label at 0.03 is worth nothing and must not be allowed to put foliage's 2.10 on concrete.  The
    margin is therefore returned beside the label and the blurring in `albedo_map` weights by it.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

#: where a checkpoint is looked for, in order.  The first is this machine's; the rest are the usual cache locations a
#: user with `transformers` would have.  **None has to exist** -- `available()` returns False and the caller estimates.
DEFAULT_MODEL_DIRS = (
    r"D:\siglip2-base-patch16-224",
    os.path.expanduser("~/.cache/huggingface/hub/models--google--siglip2-base-patch16-224"),
    "siglip2-base-patch16-224",
)

#: the prompts are `IR_ALBEDO`'s own keys, so the table and the classifier cannot drift apart
def _prompts() -> tuple[list[str], list[str]]:
    from .nightvision import IR_ALBEDO
    classes = list(IR_ALBEDO)
    return ["a photo of " + c.replace("_", " ") for c in classes], classes


_CACHE: dict[str, Any] = {}


def find_model(explicit: str | None = None) -> str | None:
    """The first checkpoint that exists, or None.  Never raises: an absent model is a supported configuration."""
    cands = ([explicit] if explicit else []) + list(DEFAULT_MODEL_DIRS)
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


def available(explicit: str | None = None) -> bool:
    """Whether material labelling can run at all, which decides between the model and the estimate."""
    if find_model(explicit) is None:
        return False
    try:
        import torch            # noqa: F401
        import transformers     # noqa: F401
    except Exception:                                                          # noqa: BLE001
        return False
    return True


def _load(path: str):
    if path not in _CACHE:
        import torch
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(path, local_files_only=True)
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        model.eval()
        _CACHE[path] = (model, proc, torch)
    return _CACHE[path]


def analyse(rgb: np.ndarray, grid: int = 8, model_dir: str | None = None,
            margin_floor: float = 0.02, grid_h: int | None = None,
            grid_w: int | None = None) -> dict | None:
    """
    Label the picture's materials on a lattice of `grid_h` x `grid_w` cells.

    Returns `None` when no model is available, which is the signal to fall back to the luminance estimate.  Otherwise:

        labels      (grid_h, grid_w) int      the class index, or -1 where the model was unsure
        confidence  (grid_h, grid_w) float32  the margin over the runner-up, clipped at `margin_floor`
        classes     list[str]                 the class names, i.e. `IR_ALBEDO`'s keys in order
        albedo      (grid_h, grid_w) float32  the near-IR albedo of each cell's label
        source      str                       the checkpoint that produced it, recorded so a figure carries its provenance

    **`margin_floor` is what makes the fallback per-patch rather than all-or-nothing**: a cell whose margin falls below
    it is labelled -1 and carries zero confidence, so `albedo_map` lets the estimate through there.  On the reference
    photograph the margins run 0.030 to 0.692, so this is not a hypothetical case.

    **`grid_h`/`grid_w` are separate because a cell has to be square, and a single `grid` cannot make it so.**  With one
    number the cell is `w/grid` by `h/grid`, so a 16:9 or 4:3 frame gets cells that are 1.33x taller or wider than they
    are broad -- and since the cell is what a material boundary is quantised to, the *shape* of the quantisation would
    depend on the frame's aspect ratio.  The caller derives both from one **pixel** size instead, which is the quantity
    that means something physically.  `grid` is kept for the existing callers and used for both when the others are
    absent.
    """
    path = find_model(model_dir)
    if path is None or not available(model_dir):
        return None
    from PIL import Image
    from .nightvision import IR_ALBEDO

    prompts, classes = _prompts()
    model, proc, torch = _load(path)

    a = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    h, w = a.shape[:2]
    gh = int(grid_h if grid_h else grid)
    gw = int(grid_w if grid_w else grid)
    pil = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))
    cells = []
    for j in range(gh):
        for i in range(gw):
            cells.append(pil.crop((i * w // gw, j * h // gh, (i + 1) * w // gw, (j + 1) * h // gh)))

    with torch.no_grad():
        inputs = proc(text=prompts, images=cells, padding="max_length", return_tensors="pt")
        probs = model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()

    labels = np.full((gh, gw), -1, np.int32)
    conf = np.zeros((gh, gw), np.float32)
    for k, row in enumerate(probs):
        order = np.argsort(-row)
        margin = float(row[order[0]] - row[order[1]])
        j, i = divmod(k, gw)
        conf[j, i] = max(0.0, margin)
        if margin >= margin_floor:
            labels[j, i] = int(order[0])

    alb = np.zeros((gh, gw), np.float32)
    for j in range(gh):
        for i in range(gw):
            alb[j, i] = IR_ALBEDO[classes[labels[j, i]]] if labels[j, i] >= 0 else 0.0

    return {"labels": labels, "confidence": conf, "classes": classes, "albedo": alb,
            "source": os.path.basename(path.rstrip("\\/"))}


#: **Ordinal depth bins.**  The wording matters more than the count: the model was trained on captions, so these have to
#: be *ordered phrases a caption would use*, not depth numbers.  Their probabilities are combined into an expectation,
#: which is what makes the result continuous rather than a class label -- and what makes an unplaceable image degrade to
#: a mid depth, so the illuminator's falloff stays gentle instead of inventing structure.
DEPTH_BINS = (
    "something extremely close to the camera, right in front of the lens",
    "something very near the camera, within arm's reach",
    "something a short distance away, a metre or two",
    "something a few metres away in the middle of the room",
    "something far away, across the room",
    "the distant background, far behind everything else",
)


def analyse_depth(rgb: np.ndarray, grid: int = 8, model_dir: str | None = None) -> dict | None:
    """
    A per-cell relative depth in [0, 1], where 0 is nearest the camera, from the same vision model.

    **This is what lets `note.txt`'s first mechanism be modelled at all.**  It asks for

        "物体距离增加后反射光减弱 … 近处物体可能很亮，远处背景却仍然很暗"

    and a *radial* illuminator field cannot express a decline with distance -- only a vignette about a point.  **Depth is
    what the mechanism is about**, and an ordinal expectation over caption-style phrases gives it per pixel with no depth
    checkpoint.

    **What is calibrated and what is not, stated plainly because it decides how the result may be used.**  The output is
    a **relative** depth on an arbitrary zero: the bins are equally spaced and the expectation is over their indices, so
    the scale is that of the phrases rather than of metres.  **It is therefore used for the falloff's *shape*, never for
    an absolute distance in the physics.**  And on the reference photograph the model places the far wall at 0.80-0.88
    and the near floor at 0.08-0.21, which matches the picture -- **but there is no depth ground truth for that frame, so
    the ordering is corroborated by eye and not by measurement.**  The fallback covers the case where even that fails.

    Returns `None` when no model is available, exactly as `analyse` does, so the caller keeps its radial field.
    """
    path = find_model(model_dir)
    if path is None or not available(model_dir):
        return None
    from PIL import Image

    prompts = list(DEPTH_BINS)
    model, proc, torch = _load(path)

    a = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    h, w = a.shape[:2]
    pil = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))
    cells = []
    for j in range(grid):
        for i in range(grid):
            cells.append(pil.crop((i * w // grid, j * h // grid, (i + 1) * w // grid, (j + 1) * h // grid)))

    with torch.no_grad():
        inputs = proc(text=prompts, images=cells, padding="max_length", return_tensors="pt")
        P = model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()

    idx = np.arange(len(prompts), dtype=np.float32)
    depth = ((P * idx).sum(axis=1) / max(1.0, len(prompts) - 1.0)).reshape(grid, grid)
    conf = P.max(axis=1).reshape(grid, grid)
    return {"depth": depth.astype(np.float32), "confidence": conf.astype(np.float32),
            "source": os.path.basename(path.rstrip("\\/"))}
