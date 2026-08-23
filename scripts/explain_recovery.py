"""恢复行为可视化：策略抓空之后会不会重来。

    python -m scripts.explain_recovery --run runs/bc_v3 --ep 14

产出 docs/figs/recovery.png（胶片条 + 夹爪/距离曲线）和 docs/figs/recovery.gif。
挑的是闭环评测里步数明显偏长但最终成功的那几局——长出来的步数就是"重试"。
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from scripts._style import save, C
from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bc_v3")
    ap.add_argument("--ep", type=int, default=14, help="闭环评测里的第几局（seed = 1000 + ep）")
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, sm, ss = load_policy(a.run, dev)
    env = TabletopEnv(seed=a.seed)
    runner = Runner(model, vocab, sm, ss, dev, k=cfg["eval"]["ensemble_k"])

    obs = env.reset(seed=a.seed + a.ep)
    runner.reset(obs["instruction"])
    tc = env.spec.target_cube
    F, W, D, G, Z = [], [], [], [], []
    done = False
    while not done:
        F.append(obs["front"]); W.append(obs["wrist"])
        D.append(np.linalg.norm(env.tcp()[:2] - env.cube_pos(tc)[:2]) * 1000)
        G.append(env.grip_width() * 1000)
        Z.append((env.cube_pos(tc)[2] - 0.30) * 1000)
        obs, r, done, info = env.step(runner.act(obs))
    T = len(F)
    print(f"「{env.spec.instruction}」 {classify(env)}  {T} 步")

    grip_close = np.where(np.diff((np.array(G) < 50).astype(int)) > 0)[0]
    picks = sorted(set([0] + list(grip_close[:3]) + [int(T * .5), int(T * .8), T - 1]))[:7]
    fig = plt.figure(figsize=(2.05 * len(picks), 5.6))
    gs = fig.add_gridspec(2, len(picks), height_ratios=[2.1, 1.3], hspace=0.35)
    for i, t in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.concatenate([F[t], W[t]], axis=0)); ax.axis("off")
        ax.set_title(f"t={t}", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(D, color=C["front"], label="TCP–目标方块 水平距离 (mm)")
    ax.plot(G, color=C["act"], label="夹爪开度 (mm)")
    ax.plot(Z, color=C["lang"], label="方块离桌面高度 (mm)")
    for t in grip_close:
        ax.axvline(t, color=C["warn"], ls="--", lw=1)
    ax.text(0.01, 0.92, "红色虚线 = 发出闭合指令的时刻；出现多次 = 抓空之后重试",
            transform=ax.transAxes, fontsize=8, color=C["warn"])
    ax.set_xlabel("控制步"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"「{env.spec.instruction}」 共 {T} 步，{classify(env)}"
                 f"（专家只要 33 步；多出来的都是重试）", fontsize=10)
    save(fig, "docs/figs/recovery.png")
    imageio.mimsave("docs/figs/recovery.gif",
                    [np.concatenate([f, w], axis=1) for f, w in zip(F[::2], W[::2])], fps=8, loop=0)
    print("→ docs/figs/recovery.gif")


if __name__ == "__main__":
    main()
