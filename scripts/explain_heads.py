"""动作表示三条路线的对照图（notes/08 的预注册预测在这里被检验）。

    python -m scripts.explain_heads  →  docs/figs/heads.png

三个臂共用同一套视觉/语言骨干和同一份数据，只换最后一步怎么表示动作：
    regress    直接回归 (H,5)，L1                    —— ACT
    discrete   每维 41 个格子，交叉熵                 —— RT-1 / RT-2 / OpenVLA
    diffusion  100 步训练噪声表，推理 10 步 DDIM      —— Diffusion Policy / π0
"""
from __future__ import annotations

import json, os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C
from scripts.compare import mcnemar_exact
from scripts.summary import wilson

ARMS = [
    ("回归 (ACT)", "runs/bc_v5_hist", "直接回归 8×5，L1 损失"),
    ("离散 argmax\n(RT-1)", "runs/head_discrete_h", "每维 41 格，交叉熵"),
    ("离散 期望解码", "runs/head_discrete_h|eval_expect.json", "同一个模型，换解码方式"),
    ("扩散 (DP/π0)", "runs/head_diffusion_h", "10 步 DDIM 去噪"),
]
# 真正多峰的任务：指令不指定盘子，放进任意一个都算成功（notes/18）
MM_ARMS = [
    ("回归", "runs/any_regress"),
    ("离散\nargmax", "runs/any_discrete"),
    ("离散\n期望", "runs/any_discrete|eval_expect.json"),
    ("扩散", "runs/any_diffusion"),
]
BUCKETS = [("success", "成功"), ("wrong_cube", "抓错方块"), ("no_grasp", "没碰到"),
           ("pushed", "推走了"), ("dropped", "掉了"), ("off_plate", "没进盘子")]


def load(run, prefer=("eval80.json", "eval.json")):
    if "|" in run:                     # "目录|文件名"：同一个模型的另一种推理设置
        run, f = run.split("|")
        prefer = (f,)
    for f in prefer:
        p = f"{run}/{f}"
        if os.path.exists(p):
            e = json.load(open(p))
            return e["results"]
    return None


def bars(ax, names, results, title):
    xs = np.arange(len(names))
    for i, (nm, res) in enumerate(zip(names, results)):
        if res is None:
            ax.text(i, 0.05, "还没跑完", ha="center", fontsize=8, color=C["grey"])
            continue
        k, n = sum(r["success"] for r in res), len(res)
        lo, hi = wilson(k, n)
        ax.bar(i, k / n, color=C["act"] if i == 0 else C["front"], width=0.6)
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color="k", capsize=4, lw=1.2)
        ax.text(i, k / n + (hi - k / n) + 0.03, f"{k/n:.1%}\nn={n}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 1.12); ax.set_ylabel("闭环成功率")
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.3, axis="y")


def main():
    res = [load(r) for _, r, _ in ARMS]
    mm = [load(r) for _, r in MM_ARMS]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    bars(axes[0], [a[0] for a in ARMS], res, "有指令（任务几乎单峰）")
    bars(axes[1], [a[0] for a in MM_ARMS], mm, "多峰任务：放进任意盘子都算成功")
    ax = axes[1]
    ax.axhline(0.938, color=C["act"], ls="--", lw=1.2)
    ax.text(len(MM_ARMS) - 0.4, 0.955, "同配置在单峰任务上 93.8%", fontsize=7.5,
            ha="right", color=C["act"])

    # 阶段分解：多峰只发生在放置阶段，所以要把两个阶段分开看
    ax = axes[2]
    PAIRS = [("回归", "runs/bc_v5_hist", "runs/any_regress"),
             ("离散\nargmax", "runs/head_discrete_h", "runs/any_discrete"),
             ("离散\n期望", "runs/head_discrete_h|eval_expect.json",
              "runs/any_discrete|eval_expect.json")]

    def cond_place(run):
        res = load(run)
        if not res:
            return None
        c = Counter(r["outcome"] for r in res)
        lifted = c["success"] + c["off_plate"] + c["dropped"]
        return (c["success"] / lifted if lifted else np.nan, lifted, len(res))

    xs = np.arange(len(PAIRS))
    w = 0.36
    for off, key, lab, col in ((-w / 2, 1, "单峰任务", C["front"]), (w / 2, 2, "多峰任务", C["warn"])):
        vals = []
        for row in PAIRS:
            r = cond_place(row[key])
            vals.append(r[0] if r else np.nan)
        ax.bar(xs + off, vals, w, label=lab, color=col)
        for x, v, row in zip(xs, vals, PAIRS):
            r = cond_place(row[key])
            if r:
                ax.text(x + off, v + 0.02, f"{v:.0%}\n{r[1]}局", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in PAIRS], fontsize=8.5)
    ax.set_ylim(0, 1.15); ax.set_ylabel("抓起来之后放进盘子的比例")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("只看放置阶段（多峰只发生在这里）\n期望解码在单峰任务上无害，在多峰任务上崩掉", fontsize=9.5)

    fig.suptitle("动作表示三条路线：同一套骨干，只换最后一步怎么表示动作", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "docs/figs/heads.png")

    # 配对检验
    print(f"\n{'对比':<34}{'A':>8}{'B':>8}{'只A成':>7}{'只B成':>7}{'McNemar p':>11}")
    base = res[0]
    for (nm, run, _), r in zip(ARMS[1:], res[1:]):
        if base is None or r is None:
            continue
        n = min(len(base), len(r))
        b = sum(1 for i in range(n) if base[i]["success"] and not r[i]["success"])
        c = sum(1 for i in range(n) if r[i]["success"] and not base[i]["success"])
        p = mcnemar_exact(b, c)
        print(f"{'回归 vs ' + nm:<34}{sum(x['success'] for x in base[:n])/n:>7.1%}"
              f"{sum(x['success'] for x in r[:n])/n:>8.1%}{b:>7}{c:>7}{p:>11.3f}")


if __name__ == "__main__":
    main()
