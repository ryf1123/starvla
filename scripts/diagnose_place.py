"""放置失败诊断：方块最终落在盘子的哪个方位、离盘心多远。

成功率里"没进盘子"是最大的失败桶。这个脚本回答两个问题：
    1. 是**系统性偏移**（总往某个方向偏 → 可以直接修）还是**随机散布**（精度不够）？
    2. 松手时方块离盘心多远、离盘面多高？

    python -m scripts.diagnose_place --run runs/bc_v4 --episodes 60
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts._style import save, C
from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv
from sim.assets import TABLE_TOP, CUBE_HALF, PLATE_R
from expert.scripted import ScriptedExpert


def run(run_dir, episodes, seed0, device, expert=False):
    if expert:
        env = TabletopEnv(seed=seed0)
        agent = None
    else:
        model, vocab, cfg, sm, ss = load_policy(run_dir, device)
        env = TabletopEnv(seed=seed0, img_hw=cfg["eval"].get("img_hw", 128),
                          same_color_prob=cfg["eval"].get("same_color_prob", 0.0))
        agent = Runner(model, vocab, sm, ss, device, k=cfg["eval"].get("ensemble_k", 1),
                       state_history=cfg["model"].get("state_history", 1))
    out = []
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep)
        if expert:
            ex = ScriptedExpert(env, rng=np.random.default_rng(ep))
        else:
            agent.reset(obs["instruction"])
        rel_at_release, done = None, False
        prev_grip = 1.0
        while not done:
            a = ex.act() if expert else agent.act(obs)
            if prev_grip < 0 and a[4] > 0 and rel_at_release is None:   # 夹爪从闭合变张开 = 松手
                c = env.cube_pos(env.spec.target_cube)
                p = env.plate_pos(env.spec.target_plate)
                rel_at_release = (c - p)
            prev_grip = a[4]
            obs, _, done, _ = env.step(a)
        c = env.cube_pos(env.spec.target_cube)
        p = env.plate_pos(env.spec.target_plate)
        out.append(dict(outcome=classify(env), final=(c - p), release=rel_at_release))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bc_v4")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pol = run(a.run, a.episodes, a.seed, dev)
    exp = run(None, min(a.episodes, 30), a.seed, dev, expert=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, data, name in ((axes[0], exp, "脚本专家"), (axes[1], pol, a.run)):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(PLATE_R * np.cos(th) * 1000, PLATE_R * np.sin(th) * 1000, "-", color=C["grey"], lw=1.2)
        ax.plot((PLATE_R - 0.015) * np.cos(th) * 1000, (PLATE_R - 0.015) * np.sin(th) * 1000,
                "--", color=C["act"], lw=1.0)
        ok = np.array([d["final"][:2] for d in data if d["outcome"] == "success"]) * 1000
        bad = np.array([d["final"][:2] for d in data if d["outcome"] == "off_plate"]) * 1000
        if len(ok):
            ax.scatter(ok[:, 0], ok[:, 1], s=26, color=C["act"], label=f"成功 {len(ok)}", alpha=0.85)
        if len(bad):
            ax.scatter(bad[:, 0], bad[:, 1], s=32, color=C["warn"], marker="x",
                       label=f"没进盘子 {len(bad)}")
        allp = np.array([d["final"][:2] for d in data]) * 1000
        mu = allp.mean(0)
        ax.plot(mu[0], mu[1], "*", ms=18, mfc="#00e5ff", mec="k", mew=0.8)
        ax.set_title(f"{name}\n落点均值 ({mu[0]:+.0f}, {mu[1]:+.0f}) mm，"
                     f"散布 {allp.std(0).mean():.0f} mm", fontsize=10)
        ax.set_xlabel("离盘心 x (mm)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    axes[0].set_ylabel("离盘心 y (mm)")
    fig.suptitle("方块最终落点相对盘心（灰圈 = 盘沿 65 mm，绿虚线 = 判定成功的 50 mm，★ = 落点均值）\n"
                 "均值明显偏离原点 = 系统性偏差（可以直接修）；只是散得开 = 精度不够", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "docs/figs/place_scatter.png")

    for data, name in ((exp, "专家"), (pol, "策略")):
        allp = np.array([d["final"][:2] for d in data]) * 1000
        rel = np.array([d["release"][:2] for d in data if d["release"] is not None]) * 1000
        relz = np.array([d["release"][2] for d in data if d["release"] is not None]) * 1000
        print(f"{name}: n={len(data)}  落点均值 ({allp.mean(0)[0]:+.1f}, {allp.mean(0)[1]:+.1f}) mm  "
              f"半径中位数 {np.median(np.linalg.norm(allp, axis=1)):.1f} mm")
        if len(rel):
            print(f"      松手时：离盘心 {np.median(np.linalg.norm(rel, axis=1)):.1f} mm，"
                  f"方块中心离盘面 {np.median(relz):.1f} mm")


if __name__ == "__main__":
    main()
