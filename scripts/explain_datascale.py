"""数据量曲线：同样"数据 ×3"，收益差 12 倍，差别在每种条件的样本密度。

    python -m scripts.explain_datascale  →  docs/figs/datascale.png
"""
from __future__ import annotations

import json, os
import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C
from scripts.summary import wilson

# (演示条数, run 目录)
REL = [(300, "runs/rel_n300"), (600, "runs/rel_n600"), (900, "runs/rel_cls_hist"),
       (1800, "runs/rel_n1800"), (2700, "runs/rel_cls_big")]
ORIG = [(900, "runs/bc_v5_hist"), (2700, "runs/bc_v6_big")]


def pts(items):
    out = []
    for n, d in items:
        f = f"{d}/eval.json"
        if not os.path.exists(f):
            continue
        res = json.load(open(f))["results"]
        k, m = sum(r["success"] for r in res), len(res)
        lo, hi = wilson(k, m)
        out.append((n, k / m, lo, hi, m))
    return out


def main():
    rel, orig = pts(REL), pts(ORIG)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for data, lab, col, n_instr in ((rel, "36 条指令（同色方块 + 左边的/右边的）", C["warn"], 36),
                                    (orig, "12 条指令（原任务）", C["act"], 12)):
        if not data:
            continue
        x = [d[0] for d in data]; y = [d[1] for d in data]
        err = [[d[1] - d[2] for d in data], [d[3] - d[1] for d in data]]
        ax.errorbar(x, y, yerr=err, marker="o", ms=6, lw=1.8, capsize=4,
                    color=col, label=lab)
        for d in data:
            ax.annotate(f"{d[1]:.0%}\n每条指令 {d[0]//n_instr} 局",
                        (d[0], d[1]), textcoords="offset points", xytext=(6, -16),
                        fontsize=7.5, color=col)
    ax.set_xscale("log")
    ax.set_xticks([300, 600, 900, 1800, 2700])
    ax.set_xticklabels(["300", "600", "900", "1800", "2700"])
    ax.minorticks_off()          # 对数轴的次刻度标签会和自定义标签打架
    ax.set_xlabel("演示条数（同一份数据的嵌套子集）")
    ax.set_ylabel("闭环成功率（80 局，95% Wilson 区间）")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("同样是「数据 ×3」，收益差 12 倍\n"
                 "36 条指令：900→2700 是 +30 个百分点（p<1e-5）；"
                 "12 条指令：900→2700 是 +2.5 且不显著（p=0.38）", fontsize=10.5)
    fig.tight_layout()
    save(fig, "docs/figs/datascale.png")
    print(f"{'演示条数':>8}{'每条指令':>10}{'成功率':>9}{'95% 区间':>16}")
    for n, sr, lo, hi, m in rel:
        print(f"{n:>8}{n//36:>10}{sr:>9.1%}   [{lo:.0%}, {hi:.0%}]  n={m}")


if __name__ == "__main__":
    main()
