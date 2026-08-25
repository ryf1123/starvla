"""所有"改进尝试"的总览：哪些有效、哪些无效、哪些有害。

    python -m scripts.explain_improvements  →  docs/figs/improvements.png

重点不是"涨了多少"，而是**每一条都配了配对检验的 p 值**——
没有 p 值的柱状图在这个样本量下基本没有信息量。
"""
from __future__ import annotations

import glob, json, os
import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C
from scripts.compare import mcnemar_exact
from scripts.summary import wilson

# (名字, run 目录, 对照 run 目录, 一句话)
ITEMS = [
    ("多步历史观测", "runs/bc_v5_hist", "runs/bc_v5_place", "观测带最近 3 帧状态+动作"),
    ("放置精度修正", "runs/bc_v5_place", "runs/bc_v4", "专家松手高度 3.5→1.2 cm"),
    ("大动作噪声", "runs/bc_v5_noise", "runs/bc_v5_place", "采集噪声 σ 0.05→0.15"),
    ("抓取时刻过采样", "runs/bc_v5_grasp", "runs/bc_v5_place", "夹爪开合前后样本 ×3"),
    ("160×160 分辨率", "runs/bc_v5_res160", "runs/bc_v5_place", "观测 128→160"),
    ("辅助定位监督", "runs/bc_v5_aux", "runs/bc_v5_place", "预测目标像素坐标，weight=1.0"),
    ("离散 token 动作头", "runs/head_discrete_h", "runs/bc_v5_hist", "每维 41 格 + 交叉熵"),
    ("扩散动作头", "runs/head_diffusion_h", "runs/bc_v5_hist", "10 步 DDIM 去噪"),
    ("预训练 ResNet18 骨干", "runs/bb_r18", "runs/bc_v5_hist", "ImageNet 权重，同一套学习率"),
    ("数据 ×3（36 条指令）", "runs/rel_cls_big", "runs/rel_cls_hist", "900 → 2700 条演示"),
    # 域随机化不放进这张图：它的 eval 是在**随机化分布**上跑的，和基线的固定场景
    # 不是同一个测试集，放进来会是苹果比橘子。它的结果单独在 notes/13 里讲。
]

# 大样本配对结果优先（scripts/compare.py 的输出），比各自 80 局的 eval.json 可信
def large_sample(a, b):
    import glob
    for f in glob.glob("runs/cmp_*.json"):
        d = json.load(open(f))
        if d["a"] == a and d["b"] == b:
            return d
        if d["a"] == b and d["b"] == a:      # 方向反了，翻过来
            return dict(a=a, b=b, n=d["n"], sr_a=d["sr_b"], sr_b=d["sr_a"],
                        only_a=d["only_b"], only_b=d["only_a"], p=d["p"])
    return None


def load(p):
    if not os.path.exists(f"{p}/eval.json"):
        return None
    return np.array([r["success"] for r in json.load(open(f"{p}/eval.json"))["results"]], bool)


def main():
    rows = []
    for name, a, b, why in ITEMS:
        big = large_sample(a, b)
        if big:
            rows.append(dict(name=name, why=why, sr=big["sr_a"], base=big["sr_b"], n=big["n"],
                             d=big["sr_a"] - big["sr_b"], p=big["p"],
                             oa=big["only_a"], ob=big["only_b"], lo=0, hi=0))
            continue
        A, B = load(a), load(b)
        if A is None or B is None:
            continue
        n = min(len(A), len(B)); A, B = A[:n], B[:n]
        oa = int((A & ~B).sum()); ob = int((~A & B).sum())
        rows.append(dict(name=name, why=why, sr=A.mean(), base=B.mean(), n=n,
                         d=A.mean() - B.mean(), p=mcnemar_exact(oa, ob), oa=oa, ob=ob,
                         lo=wilson(int(A.sum()), n)[0], hi=wilson(int(A.sum()), n)[1]))
    if not rows:
        print("还没有结果"); return
    rows.sort(key=lambda r: -r["d"])

    fig, ax = plt.subplots(figsize=(11.5, 0.72 * len(rows) + 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        col = C["act"] if (r["p"] < 0.05 and r["d"] > 0) else (
            C["warn"] if (r["p"] < 0.05 and r["d"] < 0) else C["grey"])
        ax.barh(yi, r["d"] * 100, 0.55, color=col, alpha=0.9)
        sig = "显著" if r["p"] < 0.05 else "不显著"
        txt = (f"{r['d']*100:+.1f} 个百分点   p={r['p']:.3f} {sig}   "
               f"({r['sr']:.0%} vs {r['base']:.0%}, n={r['n']})")
        if r["d"] * 100 < -20:                      # 长的负向柱：字写在柱子里面，靠右对齐
            ax.text(-2, yi, txt, va="center", ha="right", fontsize=8.6, color="white")
        else:
            ax.text(r["d"] * 100 + (1.5 if r["d"] >= 0 else -1.5), yi, txt, va="center",
                    ha="left" if r["d"] >= 0 else "right", fontsize=8.6, color=col)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['name']}\n{r['why']}" for r in rows], fontsize=9)
    ax.set_xlabel("相对各自对照组的成功率变化（百分点，配对）")
    ax.set_xlim(-95, 62)
    ax.grid(axis="x", alpha=0.3)
    npos = sum(1 for r in rows if r["p"] < 0.05 and r["d"] > 0)
    nneg = sum(1 for r in rows if r["p"] < 0.05 and r["d"] < 0)
    ax.set_title(f"{len(rows)} 条改进尝试：{npos} 条是统计显著的正收益，{nneg} 条显著有害\n"
                 "每条都用同一批评测种子做配对 McNemar 检验——"
                 "没有 p 值的柱状图在这个样本量下没有信息量", fontsize=11)
    fig.tight_layout()
    save(fig, "docs/figs/improvements.png")
    for r in rows:
        print(f"{r['name']:<16}{r['d']*100:+6.1f} pp  p={r['p']:.3f}  n={r['n']}")


if __name__ == "__main__":
    main()
