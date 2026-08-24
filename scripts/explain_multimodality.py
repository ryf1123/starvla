"""多峰任务的教学图：分岔点在哪、条件变异涨了多少、两条下限为什么不一样。

    python -m scripts.explain_multimodality  →  docs/figs/multimodality.png

左：按阶段画条件变异（单峰 vs 多峰），分岔点一目了然。
中：同一个场景、同一条指令，专家 12 次 rollout 的 TCP 轨迹俯视图——多峰任务里分成两束。
右：为什么"均值式下限"和"采样式下限"不一样，以及它们的比值怎么当多峰性探针。
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scripts._style import save, C
from sim.tabletop_env import TabletopEnv
from sim.assets import PLATE_R
from expert.scripted import ScriptedExpert
from scripts.diagnose_multimodality import conditional_actions

PHASES = ["0 悬停\n对准", "1 下降", "2 闭合\n夹爪", "3 抬起", "4 平移到\n盘子上方", "5 下降\n放置", "6 松开\n撤离"]


def measure(any_plate, episodes=8, k=16):
    env = TabletopEnv(seed=0, any_plate=any_plate)
    rng = np.random.default_rng(0)
    per = {}
    for ep in range(episodes):
        env.reset(seed=1000 + ep)
        ex = ScriptedExpert(env, rng=np.random.default_rng(ep), noise=0.05)
        done = False
        while not done:
            A = conditional_actions(env, ex, k, 0.05, rng, resample_target=any_plate)
            m = A.mean(0)
            per.setdefault(ex.phase, []).append(
                (np.abs(A - m).mean(), np.abs(A[:, None] - A[None]).mean()))
            ex.rng = np.random.default_rng(ep); ex.noise = 0.05
            _, _, done, _ = env.step(ex.act())
    return {p: np.array(v).mean(0) for p, v in per.items()}


def trajectories(any_plate, n=12, seed=1007):
    """同一个场景重复跑，记录 TCP 的俯视轨迹。"""
    env = TabletopEnv(seed=0, any_plate=any_plate)
    env.reset(seed=seed)
    spec = env.spec
    out, plates = [], np.array(spec.plate_xy)
    for i in range(n):
        import copy
        sp = copy.deepcopy(spec)
        if any_plate:
            sp.target_plate = int(np.random.default_rng(i).integers(len(sp.plate_colors)))
        env.reset(spec=sp)
        ex = ScriptedExpert(env, rng=np.random.default_rng(100 + i), noise=0.05)
        xy, done = [], False
        while not done:
            xy.append(env.tcp()[:2].copy())
            _, _, done, _ = env.step(ex.act())
        out.append(np.array(xy))
    return out, plates, spec


def main():
    uni, multi = measure(False), measure(True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axes[0]
    xs = np.arange(len(PHASES))
    w = 0.38
    u = [uni.get(p, [np.nan, np.nan])[0] for p in range(7)]
    m = [multi.get(p, [np.nan, np.nan])[0] for p in range(7)]
    ax.bar(xs - w / 2, u, w, label="单峰任务（指令指定盘子）", color=C["front"])
    ax.bar(xs + w / 2, m, w, label="多峰任务（放进任意盘子）", color=C["warn"])
    ax.axvspan(3.5, 6.5, color="0.9", zorder=0)
    ax.text(5, max(m) * 1.02, "抓起来之后才分岔", fontsize=8.5, ha="center")
    for i in (4, 5):
        ax.annotate(f"×{m[i]/u[i]:.1f}", (i + w / 2, m[i]), textcoords="offset points",
                    xytext=(14, -2), ha="center", fontsize=8.5, color=C["warn"], weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(PHASES, fontsize=7.5)
    ax.set_ylabel("条件变异（均值式下限）")
    ax.set_title("同一个观测下，专家动作有多不一样", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3, axis="y")

    for ax, ap, t in ((axes[1], True, "多峰任务：12 次 rollout 分成两束"),):
        trajs, plates, spec = trajectories(ap)
        for tr in trajs:
            ax.plot(tr[:, 1], tr[:, 0], lw=1.1, alpha=0.75, color=C["warn"])
        for i, p in enumerate(plates):
            ax.add_patch(plt.Circle((p[1], p[0]), PLATE_R, fill=False, lw=1.6, color=C["act"]))
            ax.text(p[1], p[0], f"盘{i}", ha="center", va="center", fontsize=8, color=C["act"])
        cx = spec.cube_xy[spec.target_cube]
        ax.plot(cx[1], cx[0], "s", ms=7, color=C["front"])
        ax.text(cx[1], cx[0] - 0.03, "目标方块", ha="center", fontsize=7.5, color=C["front"])
        ax.set_aspect("equal"); ax.invert_xaxis()
        ax.set_xlabel("y（画面左右）"); ax.set_ylabel("x（画面远近）")
        ax.set_title(t + f"\n指令：「{spec.instruction}」", fontsize=9.5)
        ax.grid(alpha=0.3)

    ax = axes[2]
    ax.axis("off")
    lines = [
        ("两条下限为什么不一样", ""),
        ("", ""),
        ("均值式模型（回归）的下限", "E |a − 条件均值|"),
        ("采样式模型（扩散）的下限", "E |a − a′|，两个独立样本"),
        ("", ""),
        ("单峰（高斯型）", "两者之比 ≈ √2 ≈ 1.41"),
        ("双峰（各占一半）", "两者之比 → 1.0"),
        ("", ""),
        ("实测 单峰任务", f"{uni_all(uni)[1]:.4f} / {uni_all(uni)[0]:.4f} = {uni_all(uni)[1]/uni_all(uni)[0]:.2f}"),
        ("实测 多峰任务", f"{uni_all(multi)[1]:.4f} / {uni_all(multi)[0]:.4f} = {uni_all(multi)[1]/uni_all(multi)[0]:.2f}"),
    ]
    y = 1.0
    for a_, b_ in lines:
        ax.text(0.0, y, a_, fontsize=9, color=C["grey"], va="top")  # 中文不能用等宽字体
        ax.text(0.5, y, b_, fontsize=9, va="top")
        y -= 0.082
    ax.text(0.0, y - 0.03,
            "这个比值不用画图就能判断一份数据是不是多峰的。\n"
            "也说明：拿同一个 L1 去比回归和扩散是不公平的——\n"
            "它们的最优解本来就不是同一个东西。",
            fontsize=8.5, color=C["warn"], va="top")
    ax.set_title("为什么必须分开算两条下限", fontsize=10)

    fig.tight_layout()
    save(fig, "docs/figs/multimodality.png")


def uni_all(d):
    v = np.array(list(d.values()))
    return v[:, 0].mean(), v[:, 1].mean()


if __name__ == "__main__":
    main()
