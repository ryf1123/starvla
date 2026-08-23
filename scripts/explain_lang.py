"""语言到底有没有起作用：同一个场景，只改指令。

这是整个项目最直观的一张图/一段视频：**冻结初始状态，只把指令里的颜色换掉，
看机械臂走向哪个方块、放进哪个盘子。** 如果两次轨迹一样，语言就没接上。

产出（docs/figs/ 和 videos/）：
    lang_swap.png / lang_swap.gif   同一场景 × 4 条指令的同屏对比（一行一条指令）
    lang_embed.png                  各模式下指令编码的相似度矩阵
    lang_ablation.png               五组消融的成功率和失败归因柱状图

    python -m scripts.explain_lang --run runs/lang_cls --compare runs/lang_bow
"""
from __future__ import annotations

import argparse, json, os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from scripts._style import save, C
from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv
from sim.tasks import sample_task


def instructions_for(spec):
    """同一场景下所有"说得通"的指令：方块颜色 × 盘子颜色（颜色不同）。"""
    out = []
    for ci, cc in enumerate(spec.cube_colors):
        for pi, pc in enumerate(spec.plate_colors):
            if pc != cc:
                out.append((ci, pi, f"把{cc}色方块放进{pc}色盘子"))
    return out


def run_one(runner, env, spec, instr, img_hw=192, record=True):
    obs = env.reset(spec=spec)
    runner.reset(instr)
    frames = []
    done = False
    while not done:
        if record:
            frames.append(obs["front"])
        obs, r, done, info = env.step(runner.act(obs))
    return frames, info, classify(env), env


def fig_swap(run, seed=321, n_instr=4, img_hw=192):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, smean, sstd = load_policy(run, dev)
    runner = Runner(model, vocab, smean, sstd, dev, k=cfg["eval"]["ensemble_k"])
    env = TabletopEnv(seed=seed, img_hw=img_hw)
    env.reset(seed=seed)
    spec = env.spec
    cands = instructions_for(spec)[:n_instr]

    rows, labels = [], []
    for ci, pi, instr in cands:
        spec2 = type(spec)(**{**spec.__dict__, "target_cube": ci, "target_plate": pi,
                              "instruction": instr})
        frames, info, outcome, _ = run_one(runner, env, spec2, instr, img_hw)
        rows.append(frames)
        labels.append(f"「{instr}」→ {outcome}")

    from scripts.video_grid import grid
    grid(rows, labels, "videos/lang_swap.mp4", fps=12)
    os.replace("videos/lang_swap.gif", "docs/figs/lang_swap.gif")
    T = max(len(r) for r in rows)

    picks = [0, int(T * 0.35), int(T * 0.6), min(T - 1, int(T * 0.95))]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(2.6 * len(picks), 2.7 * len(rows)))
    for i, (r, lab) in enumerate(zip(rows, labels)):
        for j, t in enumerate(picks):
            ax = axes[i, j]
            ax.imshow(r[min(t, len(r) - 1)]); ax.axis("off")
            if j == 0:
                ax.set_title(lab, fontsize=9, loc="left", color=C["lang"])
            else:
                ax.set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"完全相同的初始场景（方块 {spec.cube_colors}，盘子 {spec.plate_colors}），"
                 f"只改指令\n模型：{run}", fontsize=10)
    save(fig, "docs/figs/lang_swap.png")
    return labels


def fig_ablation(suite="lang"):
    rows = json.load(open(f"runs/ablate_{suite}.json"))
    keys = ["success", "wrong_cube", "no_grasp", "off_plate"]
    names = ["成功", "抓错方块", "没抓起来", "没进盘子"]
    cols = [C["act"], C["warn"], C["grey"], C["wrist"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    total = sum(rows[0][k] for k in keys)
    for k, nm, col in zip(keys, names, cols):
        v = np.array([r[k] for r in rows]) / total * 100
        ax.bar(x, v, bottom=bottom, label=nm, color=col, alpha=0.9)
        for xi, (vi, bi) in enumerate(zip(v, bottom)):
            if vi > 4:
                ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center",
                        fontsize=8, color="white")
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in rows], fontsize=9)
    ax.set_ylabel("占比 (%)"); ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.set_title(f"语言消融：{total} 局闭环评测的结果分解\n"
                 f"「抓错方块」这一栏就是语言没接上的直接证据", fontsize=10)
    save(fig, f"docs/figs/{suite}_ablation.png")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/lang_cls")
    ap.add_argument("--seed", type=int, default=321)
    ap.add_argument("--ablation", action="store_true", help="只画消融汇总图")
    a = ap.parse_args()
    if a.ablation:
        fig_ablation()
    else:
        for lab in fig_swap(a.run, a.seed):
            print(" ", lab)
