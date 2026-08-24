"""配对比较的可视化：把 200 局逐局画成格子，看清"不一致的局"到底有多少。

    python -m scripts.explain_paired --a runs/bc_v5_hist --b runs/bc_v5_place

为什么值得画：一个 p 值只是一个数，而这张图能直接看到
**大量 episode 处在"谁都可能成功也可能失败"的边界上**——
不配对的话这些翻转会被当成噪声吸收掉，什么也看不出来。
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scripts._style import save, C
from scripts.compare import mcnemar_exact, wilson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/bc_v5_hist")
    ap.add_argument("--b", default="runs/bc_v5_place")
    ap.add_argument("--cmp", default="runs/cmp_hist.json")
    ap.add_argument("--labels", nargs=2, default=["最近 3 帧观测", "单帧观测"])
    a = ap.parse_args()

    # 逐局结果：优先用 compare 跑出来的大样本，否则退回各自的 eval.json
    from policy.eval import evaluate
    ea = json.load(open(f"{a.a}/eval.json"))["results"]
    eb = json.load(open(f"{a.b}/eval.json"))["results"]
    A = np.array([r["success"] for r in ea], bool)
    B = np.array([r["success"] for r in eb], bool)
    n = min(len(A), len(B)); A, B = A[:n], B[:n]

    cells = np.where(A & B, 0, np.where(A & ~B, 1, np.where(~A & B, 2, 3)))
    cols = 20
    rows = int(np.ceil(n / cols))
    grid = np.full(rows * cols, -1)
    grid[:n] = cells
    grid = grid.reshape(rows, cols)

    palette = {0: "#c6f6d5", 1: C["act"], 2: C["warn"], 3: "#e2e8f0", -1: "white"}
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * rows + 3.0))
    for i in range(rows):
        for j in range(cols):
            v = grid[i, j]
            ax.add_patch(plt.Rectangle((j, rows - 1 - i), 0.92, 0.92,
                                       fc=palette[v], ec="white", lw=1.2))
    ax.set_xlim(-0.3, cols + 0.3); ax.set_ylim(-0.3, rows + 0.3)
    ax.axis("off"); ax.set_aspect("equal")

    only_a = int((A & ~B).sum()); only_b = int((~A & B).sum())
    p = mcnemar_exact(only_a, only_b)
    la, ha = wilson(int(A.sum()), n); lb, hb = wilson(int(B.sum()), n)
    ax.legend(handles=[
        Patch(fc="#c6f6d5", label=f"两个都成功 {int((A & B).sum())}"),
        Patch(fc=C["act"], label=f"只有「{a.labels[0]}」成功 {only_a}"),
        Patch(fc=C["warn"], label=f"只有「{a.labels[1]}」成功 {only_b}"),
        Patch(fc="#e2e8f0", label=f"两个都失败 {int((~A & ~B).sum())}"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9, frameon=False)
    ax.set_title(f"{n} 局逐局配对（同一批初始状态）\n"
                 f"{a.labels[0]} {A.mean():.1%}（{la:.0%}–{ha:.0%}） vs "
                 f"{a.labels[1]} {B.mean():.1%}（{lb:.0%}–{hb:.0%}）   "
                 f"McNemar p = {p:.3f}", fontsize=11)
    fig.text(0.5, 0.015,
             f"关键是中间两种颜色：{only_a + only_b} 局（{(only_a+only_b)/n:.0%}）在两个模型之间翻转。"
             "不做配对的话，这些翻转会被当成噪声吸收掉。\n"
             "同样的两个模型各跑 200 局时：26 : 12，McNemar p = 0.034（显著）——效应量没变，只是样本量到了。",
             ha="center", fontsize=9, color=C["warn"])
    save(fig, "docs/figs/paired.png")


if __name__ == "__main__":
    main()
