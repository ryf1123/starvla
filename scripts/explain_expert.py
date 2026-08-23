"""脚本专家讲解图。

产出（docs/figs/）：
    expert_phases.png   一局演示的胶片条 + 阶段时间线 + TCP 高度/夹爪曲线（同一时间轴）
    expert_noise.png    动作噪声 0 vs 0.05 时，演示覆盖到的状态有多不一样
    expert_demo.gif     三局连续演示（前视 + 腕视）

    python -m scripts.explain_expert
"""
from __future__ import annotations

import numpy as np
import imageio
import matplotlib.pyplot as plt

from scripts._style import save, C
from sim.tabletop_env import TabletopEnv
from sim.assets import TABLE_TOP
from expert.scripted import ScriptedExpert

PHASE_NAME = ["0 悬停", "1 下降", "2 闭合", "3 抬起", "4 平移", "5 放下", "6 松手"]


def rollout(seed, noise=0.0, img_hw=192, record=True):
    env = TabletopEnv(seed=seed, img_hw=img_hw)
    obs = env.reset(seed=seed)
    ex = ScriptedExpert(env, rng=np.random.default_rng(seed), noise=noise)
    rec = dict(front=[], wrist=[], phase=[], z=[], grip=[], tcp=[], action=[])
    done = False
    while not done:
        a = ex.act()
        if record:
            rec["front"].append(obs["front"]); rec["wrist"].append(obs["wrist"])
        rec["phase"].append(ex.phase); rec["z"].append(env.tcp()[2])
        rec["grip"].append(env.grip_width()); rec["tcp"].append(env.tcp().copy())
        rec["action"].append(a.copy())
        obs, r, done, info = env.step(a)
    rec["success"] = info["success"]
    rec["instruction"] = env.spec.instruction
    return rec


def fig_phases(seed=3):
    r = rollout(seed)
    T = len(r["phase"])
    picks = [0] + [int(np.argmax(np.array(r["phase"]) == p)) for p in range(1, 7)]
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(3, len(picks), height_ratios=[2.0, 1.0, 1.0], hspace=0.45)
    for i, t in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.concatenate([r["front"][t], r["wrist"][t]], axis=0))
        ax.set_title(f"{PHASE_NAME[i]}\nt={t}", fontsize=8)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(np.array(r["z"]) - TABLE_TOP, color=C["front"], label="TCP 离桌面高度 (m)")
    ax.plot(np.array(r["grip"]), color=C["act"], label="夹爪开度 (m)")
    for t in picks[1:]:
        ax.axvline(t, color=C["grey"], ls=":", lw=0.8)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    ax.set_xlim(0, T - 1); ax.set_xticklabels([])
    ax.set_title(f"「{r['instruction']}」  共 {T} 步 = {T/20:.1f} 秒   成功={r['success']}", fontsize=10)

    ax = fig.add_subplot(gs[2, :])
    A = np.array(r["action"])
    for i, (lab, col) in enumerate(zip(["dx", "dy", "dz", "夹爪"],
                                       [C["front"], C["wrist"], C["lang"], C["warn"]])):
        ax.plot(A[:, i if i < 3 else 4], color=col, label=lab, lw=1.2)
    for t in picks[1:]:
        ax.axvline(t, color=C["grey"], ls=":", lw=0.8)
    ax.set_xlim(0, T - 1); ax.set_ylim(-1.15, 1.15); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.set_xlabel("控制步（20 Hz）")
    ax.set_title("专家动作（策略要学的就是这条曲线）", fontsize=9)
    save(fig, "docs/figs/expert_phases.png")


def fig_noise(n=25):
    """噪声让演示覆盖到航点周围的状态，否则策略一偏离就没见过。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharex=True, sharey=True)
    for ax, noise in zip(axes, (0.0, 0.05)):
        for s in range(n):
            r = rollout(1000 + s, noise=noise, img_hw=64, record=False)
            tcp = np.array(r["tcp"])
            ax.plot(tcp[:, 0], tcp[:, 2] - TABLE_TOP, lw=0.8, alpha=0.6)
        ax.set_title(f"动作噪声 σ={noise}", fontsize=10)
        ax.set_xlabel("TCP x (m)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("TCP 离桌面高度 (m)")
    fig.suptitle(f"{n} 局演示的末端轨迹：噪声把数据从「一条细线」摊成「一条带子」\n"
                 "带子的宽度就是策略在闭环里偏离之后还认得的范围", fontsize=10)
    save(fig, "docs/figs/expert_noise.png")


def gif_demo(seeds=(3, 11, 27)):
    frames = []
    for s in seeds:
        r = rollout(s, img_hw=192)
        for f, w in zip(r["front"], r["wrist"]):
            frames.append(np.concatenate([f, w], axis=1))
    imageio.mimsave("videos/expert_demo.mp4", frames, fps=20, quality=7)
    imageio.mimsave("docs/figs/expert_demo.gif", frames[::2], fps=10, loop=0)
    print("→ videos/expert_demo.mp4 / docs/figs/expert_demo.gif", len(frames), "帧")


if __name__ == "__main__":
    fig_phases()
    fig_noise()
    gif_demo()
