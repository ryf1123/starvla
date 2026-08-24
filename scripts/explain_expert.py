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


def rollout(seed, noise=0.0, img_hw=192, record=True, home_jitter=True):
    env = TabletopEnv(seed=seed, img_hw=img_hw, home_jitter=home_jitter)
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
    """数据覆盖了状态空间的多大一块——三个条件对照。

    这张图早先只对比"有无动作噪声"，但后来给初始姿态加了抖动之后，
    抖动带来的散布盖过了噪声，两栏看起来一模一样，图就名不副实了。
    改成三栏，把两种机制分开看。
    """
    conds = [(0.0, False, "σ=0，初始姿态固定\n（最初的版本）"),
             (0.05, False, "σ=0.05，初始姿态固定"),
             (0.05, True, "σ=0.05 + 初始姿态抖动\n（现在用的）")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, (noise, jit, title) in zip(axes, conds):
        pts = []
        for s in range(n):
            r = rollout(1000 + s, noise=noise, img_hw=64, record=False, home_jitter=jit)
            tcp = np.array(r["tcp"])
            ax.plot(tcp[:, 0], tcp[:, 2] - TABLE_TOP, lw=0.7, alpha=0.55)
            pts.append(tcp[:, [0, 2]])
        P = np.concatenate(pts)
        spread = float(np.mean(np.std(P, axis=0)))
        ax.set_title(f"{title}\n轨迹点散布 {spread*1000:.0f} mm", fontsize=9.5)
        ax.set_xlabel("TCP x (m)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("TCP 离桌面高度 (m)")
    fig.suptitle(f"{n} 局演示的末端轨迹：数据覆盖到状态空间的多大一块\n"
                 "带子越宽，策略在闭环里偏离之后越可能还认得——这是模仿学习最核心的一件事", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    save(fig, "docs/figs/expert_noise.png")


def fig_noise_isolated(task_seed=3, n=12):
    """把"动作噪声到底拓宽了多少状态覆盖"单独量出来：固定同一个任务和初始状态，只改噪声。

    结论出乎意料：**σ=0.05 只拓宽了 1.4 mm**（方块半宽是 22 mm）。
    因为专家是**闭环**的——比例控制器下一步就把噪声纠回来了。
    真正拓宽覆盖的是任务随机化、初始姿态抖动、以及中途注入的扰动（那是 1–3 个满幅随机动作）。
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True, sharey=True)
    for ax, noise in zip(axes, (0.0, 0.05, 0.15)):
        env = TabletopEnv(seed=task_seed, img_hw=64, home_jitter=False)
        trs = []
        for r in range(n):
            env.reset(seed=task_seed)                 # 同一个任务、同一个初始状态
            ex = ScriptedExpert(env, rng=np.random.default_rng(1000 + r), noise=noise)
            tcp, done = [], False
            while not done:
                tcp.append(env.tcp().copy())
                _, _, done, _ = env.step(ex.act())
            trs.append(np.array(tcp))
            ax.plot(trs[-1][:, 0], trs[-1][:, 2] - TABLE_TOP, lw=1.0, alpha=0.6)
        T = min(len(x) for x in trs)
        sp = float(np.stack([x[:T] for x in trs]).std(axis=0).mean()) * 1000
        ax.set_title(f"σ={noise}\n各步 TCP 散布 {sp:.1f} mm", fontsize=10)
        ax.set_xlabel("TCP x (m)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("TCP 离桌面高度 (m)")
    fig.suptitle(f"同一个任务、同一个初始状态，只改动作噪声，各重复 {n} 次\n"
                 "σ=0.05 几乎看不出区别——闭环专家下一步就把噪声纠回来了（方块半宽 22 mm 作参照）",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "docs/figs/expert_noise_isolated.png")


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
    fig_noise_isolated()
    gif_demo()
