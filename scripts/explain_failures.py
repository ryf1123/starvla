"""失败画廊：把闭环评测里失败的局重跑一遍，看它们各自错在哪。

成功率只是一个数；真正有用的是"失败长什么样"。本脚本按失败归因分组，
每类挑几局，拼成带标签的对比图和视频。

    python -m scripts.explain_failures --run runs/bc_v3 --per-kind 3
"""
from __future__ import annotations

import argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts._style import save, C
from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv

KIND_CN = {"wrong_cube": "抓错方块（语言没接上）",
           "no_grasp": "没抓起来（定位/抓取失败）",
           "off_plate": "没进盘子（放置精度不够）",
           "success": "成功"}


def replay(run, seeds, device):
    model, vocab, cfg, sm, ss = load_policy(run, device)
    env = TabletopEnv(seed=1000)
    runner = Runner(model, vocab, sm, ss, device, k=cfg["eval"]["ensemble_k"])
    out = []
    for s in seeds:
        obs = env.reset(seed=s)
        runner.reset(obs["instruction"])
        F, D = [], []
        tc = env.spec.target_cube
        done = False
        while not done:
            F.append(obs["front"])
            D.append(np.linalg.norm(env.tcp()[:2] - env.cube_pos(tc)[:2]) * 1000)
            obs, r, done, info = env.step(runner.act(obs))
        out.append(dict(seed=s, frames=F, d=D, outcome=classify(env),
                        instr=env.spec.instruction, T=env.t,
                        cubes=env.spec.cube_colors, plates=env.spec.plate_colors))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bc_v3")
    ap.add_argument("--per-kind", type=int, default=3)
    a = ap.parse_args()
    ev = json.load(open(f"{a.run}/eval.json"))["results"]
    by = {}
    for i, r in enumerate(ev):
        by.setdefault(r["outcome"], []).append(1000 + i)
    print({k: len(v) for k, v in by.items()})

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kinds = [k for k in ("wrong_cube", "no_grasp", "off_plate") if k in by]
    # 多取几局备用：约 5% 的局在重跑时会因 MPS 浮点执行历史不同而翻转结果
    # （见 notes/05-评测方法.md），所以按**重跑时的实际结果**筛选，而不是存档里的标签。
    seeds = [s for k in kinds for s in by[k][:a.per_kind + 2]]
    if not seeds:
        print("没有失败局 🎉"); return
    eps = replay(a.run, seeds, dev)
    kept, cnt = [], {}
    for e in eps:
        if e["outcome"] == "success":
            continue
        cnt[e["outcome"]] = cnt.get(e["outcome"], 0) + 1
        if cnt[e["outcome"]] <= a.per_kind:
            kept.append(e)
    eps = kept
    print("重跑后仍失败的局：", [(e["seed"], e["outcome"]) for e in eps])

    rows = len(eps)
    fig, axes = plt.subplots(rows, 5, figsize=(11.5, 2.35 * rows))
    axes = np.atleast_2d(axes)
    for i, e in enumerate(eps):
        T = len(e["frames"])
        picks = [0, int(T * .2), int(T * .45), int(T * .7), T - 1]
        for j, t in enumerate(picks):
            ax = axes[i, j]
            ax.imshow(e["frames"][min(t, T - 1)]); ax.axis("off")
            if j == 0:
                ax.set_title(f"{KIND_CN[e['outcome']]}\n「{e['instr']}」 {T} 步",
                             fontsize=8, loc="left", color=C["warn"])
            else:
                ax.set_title(f"t={t}  离目标 {e['d'][min(t, T-1)]:.0f} mm", fontsize=7.5)
    fig.suptitle(f"{a.run} 的失败画廊：每行一局，标题里是失败归因和当时离目标方块的距离", fontsize=10)
    save(fig, "docs/figs/failures.png")

    from scripts.video_grid import grid
    grid([e["frames"] for e in eps],
         [f"{KIND_CN[e['outcome']].split('（')[0]}：{e['instr']}" for e in eps],
         "videos/failures.mp4", fps=12, cols=min(3, len(eps)))


if __name__ == "__main__":
    main()
