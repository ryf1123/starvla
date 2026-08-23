"""视觉分支健康度对比：softmax 接在哪、特征图多大，训练全程发生了什么。

    python -m scripts.explain_vision

产出 docs/figs/vision_health.png：
    左：每通道 softmax 的最大概率 / 均匀概率（=1 表示分布就是均匀的，视觉分支等于瞎的）
    中：关键点在一个 batch 的不同场景之间的标准差（=0 表示不管看到什么都输出同一个位置）
    右：验证 L1（说明前两项的差异在 loss 上几乎看不出来）
"""
from __future__ import annotations

import glob, json, os
import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C

# bc_v3 训练时还没加视觉健康指标，用同样是"原始 logits + 16×16"的 lang_seq 当参照
RUNS = [("runs/lang_seq", "原始 logits + 16×16（=基线的视觉配置）", C["act"]),
        ("runs/vis_relu16", "ReLU 后 + 16×16", C["warn"]),
        ("runs/vis_raw8", "原始 logits + 8×8", C["front"]),
        ("runs/vis_relu8", "ReLU 后 + 8×8（第一版）", C["grey"])]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for d, lab, col in RUNS:
        f = f"{d}/log.jsonl"
        if not os.path.exists(f):
            continue
        log = [json.loads(l) for l in open(f)]
        va = [r for r in log if "front_max_p" in r]
        if not va:
            continue
        cfg = json.loads(json.dumps(log[0]))          # 占位
        hw = 8 if "8×8" in lab else 16
        uni = 1.0 / (hw * hw)
        steps = [r["step"] for r in va]
        axes[0].plot(steps, [r["front_max_p"] / uni for r in va], "-o", ms=3, color=col, label=lab)
        axes[1].plot(steps, [r["front_kp_std"] for r in va], "-o", ms=3, color=col, label=lab)
        vl = [r for r in log if "val_l1" in r]
        axes[2].plot([r["step"] for r in vl], [r["val_l1"] for r in vl], "-o", ms=3, color=col, label=lab)

    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].text(0.02, 0.06, "=1 就是均匀分布，视觉分支等于瞎的",
                 transform=axes[0].transAxes, fontsize=8, color="k")
    axes[0].set_yscale("log")
    axes[0].set_title("softmax 最大概率 ÷ 均匀概率（越高越尖）", fontsize=9.5)
    axes[1].set_title("关键点跨场景标准差（越高越说明它在看东西）", fontsize=9.5)
    axes[2].set_title("验证 L1（几乎看不出差别）", fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("训练步"); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
    fig.suptitle("空间 softmax 接在 ReLU 之后会静悄悄失灵：分布全程保持均匀，而 loss 照样下降", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "docs/figs/vision_health.png")


def fig_grid():
    """2×2 表：softmax 接法 × 特征图分辨率，各自的闭环成功率。"""
    import json, os
    import matplotlib.pyplot as plt
    from scripts.summary import wilson
    cells = {("原始 logits", "16×16"): "runs/bc_v3",
             ("ReLU 之后", "16×16"): "runs/vis_relu16",
             ("原始 logits", "8×8"): "runs/vis_raw8",
             ("ReLU 之后", "8×8"): "runs/vis_relu8"}
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axis("off")
    rows = ["原始 logits", "ReLU 之后"]
    cols = ["16×16", "8×8"]
    ax.text(0.5, 0.95, "空间 softmax 接法 × 特征图分辨率（每格 40 局闭环）",
            ha="center", fontsize=11)
    for j, c in enumerate(cols):
        ax.text(0.38 + j * 0.30, 0.80, f"特征图 {c}", ha="center", fontsize=10, color=C["grey"])
    for i, r in enumerate(rows):
        ax.text(0.16, 0.62 - i * 0.26, r, ha="right", va="center", fontsize=10, color=C["grey"])
        for j, c in enumerate(cols):
            d = cells[(r, c)]
            f = f"{d}/eval.json"
            if not os.path.exists(f):
                txt, col = "（未完成）", "#a0aec0"
            else:
                e = json.load(open(f))
                k = sum(1 for x in e["results"] if x["outcome"] == "success")
                n = len(e["results"])
                lo, hi = wilson(k, n)
                txt = f"{e['success_rate']:.0%}\n{lo:.0%}–{hi:.0%}"
                col = C["act"] if e["success_rate"] > 0.6 else C["warn"]
            ax.text(0.38 + j * 0.30, 0.62 - i * 0.26, txt, ha="center", va="center",
                    fontsize=15, color=col,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=col, lw=1.6))
    ax.text(0.5, 0.06,
            "读法：决定成败的是特征图分辨率，不是 softmax 前面接不接 ReLU。\n"
            "我一开始把因果安在了后者身上——见 notes/02 末尾的更正。",
            ha="center", fontsize=9.5, color=C["warn"])
    save(fig, "docs/figs/vision_grid.png")


if __name__ == "__main__":
    main()
    fig_grid()
