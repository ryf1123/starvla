"""观测新鲜度 vs 成功率：ACT 的两个招式在这里为什么都是负收益。

    python -m scripts.explain_freshness  →  docs/figs/freshness.png

两个机制分开画，不硬凑成一个横轴——它们让动作"变旧"的方式不一样：
  - 时间集成：把旧预测**混进**当前动作（每一步都掺一点旧的）
  - 分块开环执行：用旧预测**顶替**当前动作（连续若干步完全不看新观测）
"""
from __future__ import annotations

import json, os
import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C
from scripts.summary import wilson

ENSEMBLE = [(1, 80, 0.838), (2, 40, 0.750), (4, 80, 0.750), (8, 40, 0.650)]
STRIDE_FILE = "runs/ablate_stride.json"


def main():
    stride = []
    if os.path.exists(STRIDE_FILE):
        stride = [(r["stride"], r["n"], r["sr"]) for r in json.load(open(STRIDE_FILE))]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, data, xlabel, title, col in (
            (axes[0], ENSEMBLE, "时间集成窗口 k",
             "把旧预测**混进**当前动作", C["front"]),
            (axes[1], stride, "重规划间隔 stride（步）",
             "用旧预测**顶替**当前动作", C["warn"])):
        if not data:
            continue
        xs = [d[0] for d in data]
        ys = [d[2] * 100 for d in data]
        ax.plot(xs, ys, "-o", color=col, ms=7, lw=1.8)
        for x, n, sr in data:
            lo, hi = wilson(round(sr * n), n)
            ax.plot([x, x], [lo * 100, hi * 100], color="k", lw=1.2)
            ax.text(x, sr * 100 + 4, f"{sr:.0%}", ha="center", fontsize=9.5, color=col)
        ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
        ax.set_title(title.replace("**", ""), fontsize=10.5)
    axes[0].set_ylabel("闭环成功率 (%)"); axes[0].set_ylim(0, 100)
    fig.suptitle("ACT 的两个招式在这个任务上都是负收益，而且都随「用了多旧的信息」单调下降\n"
                 "抓取的容错只有厘米级，0.4 秒足够让末端偏出去——这个任务对闭环的依赖极强",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "docs/figs/freshness.png")


if __name__ == "__main__":
    main()
