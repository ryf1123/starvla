"""成功率的演进史：每一步改了什么、涨了多少、以及哪些归因后来被推翻。

    python -m scripts.explain_progress  →  docs/figs/progress.png

这张图的重点不是"涨了多少"，而是**中间那两次箭头是虚线**——
那两步我一次改了好几处，事后无法归因；直到补做对照实验才知道真正起作用的是什么。
"""
from __future__ import annotations

import json, os
import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C
from scripts.summary import wilson

STEPS = [
    ("第一版\n（旧任务）", 34, 50, "8×8 特征图 + 600 条演示\n12000 步 + 时间集成 k=4", C["warn"], None),
    ("bc_v3", 72.5, 40, "同时改了：16×16 特征图、\n初始姿态抖动、恢复数据、\n800 条演示、16000 步", C["front"],
     "一次改了 5 处 → 无法归因"),
    ("vis_relu16", 87.5, 40, "只把空间 softmax 改回\n默认写法（ReLU + 温度 1.0）", C["act"],
     "对照实验：我之前的「修复」\n其实是负收益"),
]


def main():
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    xs = np.arange(len(STEPS))
    for i, (name, sr, n, what, col, note) in enumerate(STEPS):
        lo, hi = wilson(round(sr / 100 * n), n)
        ax.bar(i, sr, 0.5, color=col, alpha=0.9)
        ax.plot([i, i], [lo * 100, hi * 100], color="k", lw=1.4)
        ax.text(i, sr + 4, f"{sr:.1f}%", ha="center", fontsize=13, color=col)
        ax.text(i, 4, what, ha="center", va="bottom", fontsize=8.2, color="#2d3748")
        if note:
            ax.annotate(note, xy=(i - 0.5, (STEPS[i - 1][1] + sr) / 2),
                        xytext=(i - 0.5, 128 - (i - 1) * 14), fontsize=8.8, color=C["warn"],
                        ha="center", va="top",
                        arrowprops=dict(arrowstyle="->", color=C["warn"], lw=1.1,
                                        connectionstyle="arc3,rad=0.2"))
    for i in range(len(STEPS) - 1):
        ax.annotate("", xy=(i + 1 - 0.26, STEPS[i + 1][1]), xytext=(i + 0.26, STEPS[i][1]),
                    arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.6, ls="--"))
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in STEPS], fontsize=10)
    ax.set_ylabel("闭环成功率 (%)"); ax.set_ylim(0, 132)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("成功率的演进史（误差棒为 95% Wilson 区间）\n"
                 "虚线箭头 = 这一步同时改了多处，当时无法归因", fontsize=11)
    ax.text(0.5, -0.165,
            "真正的教训不在涨了多少，而在：中间那一步我一次改了五处，事后花了一整轮对照实验才知道，"
            "其中一处（我自称的「修复」）其实是负收益。",
            transform=ax.transAxes, ha="center", fontsize=9, color=C["warn"])
    save(fig, "docs/figs/progress.png")


if __name__ == "__main__":
    main()
