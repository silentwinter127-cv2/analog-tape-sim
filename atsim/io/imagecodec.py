"""
Image encoding and decoding without Pillow.

Pillow is a convenient dependency and an unnecessary one.  The tool only ever
needs three things from it -- write a PNG, write a JPEG, read a PNG/JPEG -- and
each of those has a path that needs nothing beyond the standard library plus the
FFmpeg binary the project already requires for video.  On a machine without
Pillow the previous code failed at *the first preview frame*, which is the worst
possible place for a dependency to bite: the program starts, accepts a clip, and
then dies deep inside a request handler.

What is implemented here:

    PNG   encode + decode, in pure Python with `zlib`
    JPEG  encode + decode, through FFmpeg

PNG needs no external tool because the format is genuinely simple once the
container is understood: a signature, then length-prefixed chunks, with the pixel
data in zlib-compressed IDAT chunks and one filter byte per scanline.  That is
enough to satisfy both the encoder and a decoder for the non-interlaced 8-bit RGB
and RGBA files that cameras, screenshots and image editors produce.

The public functions mirror what the rest of the code used to call, so a caller
does not care which implementation is behind them.
"""
from __future__ import annotations

import io
import os
import struct
import zlib

import numpy as np

PNG_SIG = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------
def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def png_encode(rgb: np.ndarray, level: int = 1) -> bytes:
    """
    Encode an (H, W, 3) float or uint8 array as an 8-bit RGB PNG.

    `level=1` mirrors the fast setting the preview used: these frames are written
    to a pipe and thrown away, so spending CPU on compression is wasted.
    """
    arr = rgb
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    h, w, ch = arr.shape
    if ch == 4:
        colour_type = 6
    else:
        colour_type = 2
        arr = arr[:, :, :3]
    # one filter byte (0 = None) per scanline.  Filtering would shrink the output,
    # but the frames are transient and the images are already noisy, so the
    # simpler path is also the faster one.
    stride = w * (4 if colour_type == 6 else 3)
    raw = np.empty((h, stride + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = arr.reshape(h, stride)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0)
    return (PNG_SIG
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
            + _chunk(b"IEND", b""))


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def png_decode(data: bytes) -> np.ndarray:
    """
    Decode a PNG into an (H, W, 3) uint8 array.

    Supports 8-bit non-interlaced greyscale, RGB and RGBA, which covers what
    cameras, screenshots and image editors emit.  Palette, 16-bit and interlaced
    files raise rather than returning something wrong -- a silently mis-decoded
    reference image would be far more confusing than a clear error.
    """
    if data[:8] != PNG_SIG:
        raise ValueError("not a PNG")
    pos = 8
    width = height = 0
    depth = colour = interlace = 0
    idat = bytearray()
    palette = b""
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, colour, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body)
        elif tag == b"PLTE":
            palette = body
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if depth != 8:
        raise ValueError(f"unsupported PNG bit depth {depth}")
    if interlace:
        raise ValueError("interlaced PNG is not supported")
    raw = zlib.decompress(bytes(idat))
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colour)
    if channels is None:
        raise ValueError(f"unsupported PNG colour type {colour}")
    stride = width * channels
    out = np.empty((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=pos).astype(np.int32)
        pos += stride
        if ftype == 0:
            cur = line
        elif ftype == 1:
            cur = line.copy()
            for x in range(channels, stride):
                cur[x] = (cur[x] + cur[x - channels]) & 0xFF
        elif ftype == 2:
            cur = (line + prev) & 0xFF
        elif ftype == 3:
            cur = line.copy()
            for x in range(stride):
                left = cur[x - channels] if x >= channels else 0
                cur[x] = (cur[x] + ((left + prev[x]) >> 1)) & 0xFF
        elif ftype == 4:
            cur = line.copy()
            for x in range(stride):
                left = int(cur[x - channels]) if x >= channels else 0
                upleft = int(prev[x - channels]) if x >= channels else 0
                cur[x] = (cur[x] + _paeth(left, int(prev[x]), upleft)) & 0xFF
        else:
            raise ValueError(f"bad PNG filter {ftype}")
        out[y] = cur.astype(np.uint8)
        prev = cur
    img = out.reshape(height, width, channels)
    if colour == 0:
        return np.repeat(img, 3, axis=2)
    if colour == 4:
        return np.repeat(img[:, :, :1], 3, axis=2)
    if colour == 6:
        return img[:, :, :3].copy()
    if colour == 3:
        if not palette:
            raise ValueError("palette PNG without PLTE")
        pal = np.frombuffer(palette, dtype=np.uint8).reshape(-1, 3)
        return pal[img[:, :, 0]]
    return img


# ---------------------------------------------------------------------------
# JPEG, via FFmpeg
# ---------------------------------------------------------------------------
def _ffmpeg() -> str:
    from ..io.ffmpeg import resolve_tool, FFMPEG
    return resolve_tool("ffmpeg", FFMPEG)


def _pillow():
    """Pillow if it is installed, else None.  Optional, never required."""
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


def jpeg_encode(rgb: np.ndarray, quality: int = 88) -> bytes:
    """
    Encode to JPEG, using Pillow when available and FFmpeg otherwise.

    Pillow is used first purely for speed: it encodes in-process, whereas the
    FFmpeg path spawns a subprocess per call (measured ~50 ms against ~6 ms for a
    360x240 PNG).  Neither is required -- the PNG encoder is pure standard
    library, and FFmpeg is already a hard requirement for video -- so a machine
    without Pillow loses milliseconds, not functionality.
    """
    Image = _pillow()
    if Image is not None:
        arr = rgb
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(arr[:, :, :3])).save(
            buf, format="JPEG", quality=int(quality))
        return buf.getvalue()
    import subprocess
    from ..io.ffmpeg import _creationflags, _tool_error
    arr = rgb
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    arr = np.ascontiguousarray(arr[:, :, :3])
    h, w = arr.shape[:2]
    q = max(2, min(31, int(round(31 - (quality / 100.0) * 29))))
    cmd = [_ffmpeg(), "-y", "-v", "error", "-nostdin",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-i", "-", "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg",
           "-q:v", str(q), "-"]
    try:
        proc = subprocess.run(cmd, input=arr.tobytes(), capture_output=True,
                              creationflags=_creationflags())
    except FileNotFoundError:
        raise _tool_error("ffmpeg") from None
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg jpeg encode failed: "
                           + proc.stderr.decode("utf-8", "replace")[:300])
    return proc.stdout


def jpeg_decode(data: bytes) -> np.ndarray:
    """
    Decode a JPEG (or any image FFmpeg understands) to an (H, W, 3) uint8 array.

    Routed through PNG rather than raw rgb24 because a raw buffer carries no
    dimensions.  Asking FFmpeg for PNG in one pass both frame-accurately decodes
    and tells us the size, since PNG stores it in the header -- one subprocess
    instead of a probe plus a decode that could disagree.
    """
    return decode_image_bytes(data)


def decode_image_file(path: str) -> np.ndarray:
    """Decode one image file through FFmpeg at its native size."""
    import subprocess
    from ..io.ffmpeg import _creationflags, _tool_error
    cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-i", str(path),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True,
                              creationflags=_creationflags())
    except FileNotFoundError:
        raise _tool_error("ffmpeg") from None
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg image decode failed: "
                           + proc.stderr.decode("utf-8", "replace")[:300])
    return decode_image_bytes(proc.stdout)


def decode_image_bytes(buf: bytes) -> np.ndarray:
    """
    Decode image bytes of unknown format, returning (H, W, 3) uint8.

    Routed through PNG so the dimensions come for free from the header, instead
    of needing a separate probe that could disagree with the decode.

    Note there is no `-f` before `-i`: the container has to be probed, because
    these bytes may be PNG, JPEG, BMP or WebP.  Declaring an input format made
    FFmpeg skip the probe and then reject perfectly valid data.
    """
    import subprocess
    from ..io.ffmpeg import _creationflags, _tool_error
    cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-i", "-",
           "-frames:v", "1", "-c:v", "png", "-f", "image2pipe", "-"]
    try:
        proc = subprocess.run(cmd, input=buf, capture_output=True,
                              creationflags=_creationflags())
    except FileNotFoundError:
        raise _tool_error("ffmpeg") from None
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg conversion failed: "
                           + proc.stderr.decode("utf-8", "replace")[:300])
    return png_decode(proc.stdout)


def resize_nearest(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resample to an exact size with nearest-neighbour sampling.

    Deliberately not a smooth filter: this exists so the analysis and contact
    sheet tools need no imaging library, and for sheets and crops it is also the
    honest choice -- it never invents detail, so a metric computed from a resized
    sheet is a statement about the actual pixels.
    """
    arr = rgb
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = arr.shape[:2]
    if (w, h) == (int(width), int(height)):
        return arr
    ys = np.clip((np.arange(int(height)) + 0.5) * h / max(1, int(height)),
                 0, h - 1).astype(np.int32)
    xs = np.clip((np.arange(int(width)) + 0.5) * w / max(1, int(width)),
                 0, w - 1).astype(np.int32)
    return arr[ys][:, xs]


def fit_into(rgb: np.ndarray, width: int, height: int,
             pad: int = 0) -> np.ndarray:
    """Scale-to-fill and centre-crop into exactly (height, width) uint8."""
    arr = rgb
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = arr.shape[:2]
    tw, th = max(1, int(width) - 2 * pad), max(1, int(height) - 2 * pad)
    scale = max(tw / max(1, w), th / max(1, h))
    rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    big = resize_nearest(arr, rw, rh)
    x0 = max(0, (rw - tw) // 2)
    y0 = max(0, (rh - th) // 2)
    crop = big[y0:y0 + th, x0:x0 + tw]
    out = np.zeros((int(height), int(width), 3), np.uint8)
    out[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop[:th, :tw]
    return out


def image_size(path: str | os.PathLike) -> tuple[int, int] | None:
    """(width, height) of a still, or None if it cannot be determined."""
    try:
        arr = load_image(path)
        return int(arr.shape[1]), int(arr.shape[0])
    except Exception:
        return None



# ---------------------------------------------------------------------------
# on-screen display: a 5x7 bitmap font
#
# Enough to draw the short strings a deck puts on screen ("SEARCH", "NO SIGNAL",
# chapter numbers).  Each glyph is seven rows of five bits, most significant bit
# leftmost, which keeps the table readable as art in the source.
# ---------------------------------------------------------------------------
GLYPH5X7: dict[str, tuple[int, ...]] = {
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
}


def draw_text(rgb: np.ndarray, text: str, x: int, y: int,
              scale: int = 3, colour: tuple[float, float, float] = (0.92, 0.92, 0.92)
              ) -> np.ndarray:
    """
    Draw `text` into an RGB float array, returning a new array.

    Characters outside the table are skipped rather than raising: an on-screen
    display is decoration, and a missing glyph must not take down a render.
    """
    out = np.array(rgb, dtype=np.float32, copy=True)
    h, w = out.shape[:2]
    col = np.array(colour, np.float32)
    cx = int(x)
    for ch in str(text).upper():
        glyph = GLYPH5X7.get(ch)
        if glyph is None:
            cx += 6 * scale
            continue
        for ry, bits in enumerate(glyph):
            for rx in range(5):
                if not (bits >> (4 - rx)) & 1:
                    continue
                y0, x0 = y + ry * scale, cx + rx * scale
                y1, x1 = min(h, y0 + scale), min(w, x0 + scale)
                if y0 < h and x0 < w and y1 > 0 and x1 > 0:
                    out[max(0, y0):y1, max(0, x0):x1] = col
        cx += 6 * scale
    return out


def deck_blue_screen(height: int, width: int,
                     lines: tuple[str, ...] = ("NO SIGNAL",),
                     phase: float = 0.0) -> np.ndarray:
    """
    The graphic a deck substitutes when it cannot use the tape at all.

    Not a degraded picture: the machine stops trying to display the tape and puts
    its own output on screen.  Measured on the reference, that output is a flat
    blue field (saturation ~0.94 because a pure blue field is nearly all chroma)
    with white text -- which is why a "blue screen with writing" is a different
    failure from any amount of noise on a real picture.
    """
    out = np.zeros((int(height), int(width), 3), np.float32)
    # slightly non-uniform, as a real character generator's field is
    ramp = np.linspace(-0.04, 0.04, int(width), dtype=np.float32)[None, :]
    out[:, :, 2] = 0.72 + ramp
    out[:, :, 0] = 0.06
    out[:, :, 1] = 0.10
    scale = max(1, int(round(height / 240.0 * 3)))
    y = int(height * 0.16)
    for line in lines:
        out = draw_text(out, line, int(width * 0.12), y, scale=scale)
        y += 9 * scale
    return np.clip(out, 0.0, 1.0)

# ---------------------------------------------------------------------------
# format dispatch
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_image_path(path: str | os.PathLike) -> bool:
    """True for a still image, as opposed to a video clip."""
    return str(path).lower().endswith(IMAGE_EXTENSIONS)


def load_image(path: str | os.PathLike) -> np.ndarray:
    """
    Read a still image, or a clip's first frame, as float RGB in 0..1.

    PNG is tried with the built-in decoder first, but any failure falls back to
    FFmpeg rather than propagating: the built-in decoder deliberately supports
    only a subset of the format (8-bit non-interlaced), and a file it cannot read
    is not a broken file.  Failing over costs one subprocess; refusing to open the
    image would be the actual bug.
    """
    p = str(path)
    try:
        with open(p, "rb") as fh:
            head = fh.read(8)
    except OSError:
        head = b""
    if head == PNG_SIG:
        try:
            with open(p, "rb") as fh:
                return png_decode(fh.read()).astype(np.float32) / 255.0
        except Exception:
            pass    # 16-bit, interlaced, palette oddities: let FFmpeg handle it
    if head[:2] == b"\xff\xd8":
        return jpeg_decode(open(p, "rb").read()).astype(np.float32) / 255.0
    return decode_image_file(p).astype(np.float32) / 255.0


def save_image(path: str | os.PathLike, rgb: np.ndarray, quality: int = 92) -> None:
    """Write a still, choosing the encoder from the extension."""
    p = str(path)
    arr = rgb
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    low = p.lower()
    if low.endswith(".png"):
        data = png_encode(arr)
    elif low.endswith((".jpg", ".jpeg")):
        data = jpeg_encode(arr, quality=quality)
    else:
        raise ValueError(f"unsupported image extension: {p}")
    with open(p, "wb") as fh:
        fh.write(data)
