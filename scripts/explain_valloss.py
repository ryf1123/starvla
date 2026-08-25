"""跨全部实验：验证损失到底能不能预测闭环成功率。

本项目反复遇到"val L1 差不多、闭环差很多"，但一直是轶事。
这个脚本把 runs/ 下**所有**有 eval.json 的实验拿出来，画 val L1 vs 闭环成功率，
并算 Spearman 秩相关——把主张变成一个数字。

    python -m scripts.explain_valloss  →  docs/figs/valloss.png
"""
from __future__ import annotations

import glob, json, os
import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C


def spearman(x, y):
    def rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v))
        return r
    a, b = rank(np.asarray(x)), rank(np.asarray(y))
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


def main():
    rows = []
    for d in sorted(glob.glob("runs/*")):
        f, g = f"{d}/log.jsonl", f"{d}/eval.json"
        if not (os.path.exists(f) and os.path.exists(g)):
            continue
        L = [json.loads(l) for l in open(f)]
        va = [r for r in L if "val_l1" in r]
        if not va:
            continue
        e = json.load(open(g))
        rows.append((os.path.basename(d), va[-1]["val_l1"], e["success_rate"], len(e["results"])))
    if len(rows) < 5:
        print("有效实验太少"); return

    names = [r[0] for r in rows]
    x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    rho = spearman(x, y)

    # 关键的一段：val L1 落在"都学会了"的窄带里时，它还有没有分辨力
    BAND = (0.080, 0.095)
    band = [i for i in range(len(rows)) if BAND[0] <= x[i] <= BAND[1]]
    rho_band = spearman(x[band], y[band])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, idx, t, r in (
            (axes[0], range(len(rows)), f"全部 {len(rows)} 个实验", rho),
            (axes[1], band,
             f"只看 val L1 ∈ [{BAND[0]}, {BAND[1]}] 的 {len(band)} 个\n"
             "（「都学会了」的那一段——我们做的每个选择都在这里）", rho_band)):
        idx = list(idx)
        ax.scatter(x[idx], y[idx], s=42, color=C["front"], zorder=3)
        for i in idx:
            ax.annotate(names[i], (x[i], y[i]), fontsize=6.5, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel("验证集 L1（训练时能看到的指标）")
        ax.set_ylabel("闭环成功率（真正关心的指标）")
        ax.set_title(f"{t}\nSpearman 秩相关 ρ = {r:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].axvspan(*BAND, color=C["warn"], alpha=0.12, zorder=0)
    axes[0].text(BAND[1] + 0.002, 0.06, "右图放大\n这一段", fontsize=7.5, color=C["warn"])
    fig.suptitle("验证损失能不能预测闭环成功率？"
                 "（每个点是一个训练好的模型，横轴越小 = 验证损失越好）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "docs/figs/valloss.png")

    print(f"{'实验':<22}{'val L1':>9}{'成功率':>9}{'局数':>6}")
    for n, a, b, c in sorted(rows, key=lambda r: r[1]):
        print(f"{n:<22}{a:>9.4f}{b:>9.1%}{c:>6}")
    print(f"\n全部 {len(rows)} 个实验：Spearman ρ = {rho:+.2f}")
    print(f"只看 val L1 ∈ [{BAND[0]}, {BAND[1]}] 的 {len(band)} 个：ρ = {rho_band:+.2f}"
          f"，而成功率横跨 {y[band].min():.0%}–{y[band].max():.0%}")
    print("（ρ = −1 表示 val L1 越小成功率越高，完美预测；ρ = 0 表示毫无关系）")


if __name__ == "__main__":
    main()
