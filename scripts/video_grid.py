"""同屏对比视频：每格带中文标签，标签写清变量值（"λ=0.1"而不是"实验 2"）。

    from scripts.video_grid import grid
    grid([frames_a, frames_b], ["lang_cls", "lang_bow"], "videos/cmp.mp4")
"""
from __future__ import annotations

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def label(img, text, height=26, fs=15, fg=(20, 20, 20), bg=(245, 245, 245)):
    h, w = img.shape[:2]
    bar = Image.new("RGB", (w, height), bg)
    d = ImageDraw.Draw(bar)
    d.text((6, height // 2), text, font=ImageFont.truetype(FONT_PATH, fs), fill=fg, anchor="lm")
    return np.concatenate([np.array(bar), img], axis=0)


def grid(seqs, labels, out, fps=12, cols=None, gif=True, gif_stride=2):
    """seqs: list of 帧序列（每个是 (T,H,W,3) 或 list）。短的用最后一帧补齐。"""
    T = max(len(s) for s in seqs)
    cols = cols or len(seqs)
    rows = int(np.ceil(len(seqs) / cols))
    frames = []
    for t in range(T):
        tiles = [label(np.asarray(s[min(t, len(s) - 1)]), lab) for s, lab in zip(seqs, labels)]
        h, w = tiles[0].shape[:2]
        blank = np.full((h, w, 3), 255, np.uint8)
        while len(tiles) < rows * cols:
            tiles.append(blank)
        frames.append(np.concatenate(
            [np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows)], axis=0))
    imageio.mimsave(out, frames, fps=fps, quality=7)
    print("→", out)
    if gif:
        g = out.rsplit(".", 1)[0] + ".gif"
        imageio.mimsave(g, frames[::gif_stride], fps=max(1, fps // gif_stride), loop=0)
        print("→", g)
    return frames
