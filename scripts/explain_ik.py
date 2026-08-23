"""IK 讲解图：阻尼最小二乘到底在算什么，λ 改了会怎样。

产出（docs/figs/）：
    ik_jacobian.png    一帧真实的雅可比数值 + 残差 + 解出的 Δq
    ik_lambda.png      λ 从 0.001 扫到 1：残差、Δq 范数、迭代次数
    ik_lambda.gif      同一个目标、不同 λ 下机械臂收敛过程
    ik_converge.png    单帧的收敛曲线（位置残差随迭代下降）

    python -m scripts.explain_ik
"""
from __future__ import annotations

import numpy as np
import mujoco
import imageio
import matplotlib.pyplot as plt

from scripts._style import save, C
from sim.assets import build_scene, HOME_QPOS, TABLE_TOP, ARM_JOINTS
from sim.ik import solve_ik, yaw_to_R, rot_error


def setup():
    m, lay = build_scene(["红", "绿", "蓝"], ["黄", "蓝"])
    d = mujoco.MjData(m)
    d.qpos[lay["arm_qadr"]] = HOME_QPOS
    d.qpos[lay["finger_qadr"]] = 0.04
    mujoco.mj_forward(m, d)
    return m, lay, d


def fig_jacobian(target=(0.45, -0.12, TABLE_TOP + 0.05), yaw=0.3):
    m, lay, d = setup()
    sid = lay["tcp_site"]
    dof = lay["arm_dofadr"]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, sid)
    J = np.vstack([jacp[:, dof], jacr[:, dof]])
    p_cur = d.site_xpos[sid].copy()
    R_cur = d.site_xmat[sid].reshape(3, 3).copy()
    e = np.concatenate([np.array(target) - p_cur, rot_error(R_cur, yaw_to_R(yaw))])
    lam = 0.15
    dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.axis("off")
    ax.set_title("一次 IK 迭代的真实数字（起点：home 位姿）", fontsize=11)
    lines = [
        ("当前 TCP 位置 p", f"[{', '.join(f'{x:+.4f}' for x in p_cur)}]"),
        ("目标位置 p*", f"[{', '.join(f'{x:+.4f}' for x in target)}]"),
        ("位置残差 e_p = p* − p", f"[{', '.join(f'{x:+.4f}' for x in e[:3])}]  ‖·‖={np.linalg.norm(e[:3]):.4f} m"),
        ("姿态残差 e_r（轴角）", f"[{', '.join(f'{x:+.4f}' for x in e[3:])}]  ‖·‖={np.linalg.norm(e[3:]):.4f} rad"),
        ("雅可比 J 的形状", f"6×7（6 个任务维度 × 7 个手臂关节）"),
        ("J 的位置部分（前 3 行）", ""),
    ]
    y = 1.0
    for k, v in lines:
        ax.text(0.0, y, k, fontsize=9, color=C["grey"], va="top")
        ax.text(0.33, y, v, fontsize=9, family="monospace", va="top")
        y -= 0.075
    for r in range(3):
        ax.text(0.33, y, "[" + "  ".join(f"{x:+.3f}" for x in J[r]) + "]",
                fontsize=8.5, family="monospace", va="top")
        y -= 0.055
    y -= 0.02
    for k, v in [
        ("解 (J Jᵀ + λ²I) x = e，λ=0.15", ""),
        ("Δq = Jᵀ x", "[" + "  ".join(f"{x:+.4f}" for x in dq) + "]  rad"),
        ("‖Δq‖", f"{np.linalg.norm(dq):.4f} rad（超过 0.3 会被截断）"),
        ("对应关节", "  ".join(f"{j[-1]}" for j in ARM_JOINTS)),
    ]:
        ax.text(0.0, y, k, fontsize=9, color=C["grey"], va="top")
        ax.text(0.33, y, v, fontsize=9, family="monospace", va="top")
        y -= 0.075
    ax.text(0.0, y - 0.02,
            "读法：J 的每一列是「这个关节转 1 rad，TCP 会往哪动」。"
            "位置部分第 4、6 列（肘和腕）在 z 方向分量最大，抬高末端主要靠它们。",
            fontsize=8.5, color=C["warn"], va="top", wrap=True)
    save(fig, "docs/figs/ik_jacobian.png")


def fig_lambda(target=(0.58, 0.16, TABLE_TOP + 0.10), yaw=0.4):
    """λ 扫描：精确 vs 平滑的取舍。目标选在接近臂展边缘的地方，差别才看得出来。"""
    lams = np.logspace(-3, 0, 13)
    res, dqn, iters = [], [], []
    for lam in lams:
        m, lay, d = setup()
        q_prev = d.qpos[lay["arm_qadr"]].copy()
        q, ep, er = solve_ik(m, d, lay, np.array(target), yaw, HOME_QPOS, iters=30, lam=lam)
        res.append(ep * 1000)
        dqn.append(np.linalg.norm(q - q_prev))
        # 收敛到 1 mm 需要几次
        n = 30
        for k in (1, 2, 3, 5, 8, 12, 20, 30):
            m2, lay2, d2 = setup()
            _, e2, _ = solve_ik(m2, d2, lay2, np.array(target), yaw, HOME_QPOS, iters=k, lam=lam)
            if e2 < 1e-3:
                n = k
                break
        iters.append(n)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, y, lab, col in zip(axes, [res, dqn, iters],
                               ["30 次迭代后的位置残差 (mm)", "关节角总变化 ‖q − q_home‖ (rad)",
                                "收敛到 1 mm 需要的迭代次数"],
                               [C["warn"], C["front"], C["act"]]):
        ax.semilogx(lams, y, "-o", color=col, ms=4)
        ax.axvline(0.15, color="k", ls="--", lw=1)
        ax.text(0.17, ax.get_ylim()[0] + 0.75 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "λ=0.15", fontsize=8)
        ax.set_xlabel("阻尼 λ"); ax.set_title(lab, fontsize=9.5); ax.grid(alpha=0.3)
    fig.suptitle(f"阻尼 λ 的取舍：目标 {np.round(target,3).tolist()}，从 home 位姿出发\n"
                 "λ 小 → 准但每步跳得大、靠近奇异位形会发散；λ 大 → 稳但收敛慢、残差大", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, "docs/figs/ik_lambda.png")


def fig_converge(target=(0.45, -0.12, TABLE_TOP + 0.05), yaw=0.3):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for lam, col in zip([0.01, 0.05, 0.15, 0.5], [C["front"], C["act"], C["warn"], C["grey"]]):
        errs = []
        for k in range(1, 26):
            m, lay, d = setup()
            _, ep, _ = solve_ik(m, d, lay, np.array(target), yaw, HOME_QPOS, iters=k, lam=lam)
            errs.append(ep * 1000)
        ax.semilogy(range(1, 26), errs, "-o", ms=3, color=col, label=f"λ={lam}")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.text(1, 1.15, "1 mm", fontsize=8)
    ax.set_xlabel("迭代次数"); ax.set_ylabel("位置残差 (mm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_title("同一个目标，不同阻尼下的收敛速度\n控制环里每步只跑 12 次——因为上一步的解是很好的初值", fontsize=10)
    save(fig, "docs/figs/ik_converge.png")


def gif_lambda(target=(0.58, 0.16, TABLE_TOP + 0.10), yaw=0.4):
    """同一个目标，四个 λ，把每次迭代的姿态渲染出来。"""
    lams = [0.01, 0.05, 0.15, 0.6]
    seqs = []
    for lam in lams:
        m, lay, d = setup()
        m.vis.global_.offwidth = m.vis.global_.offheight = 240
        r = mujoco.Renderer(m, 240, 240)
        frames = []
        for k in range(1, 21):
            m2, lay2, d2 = setup()
            solve_ik(m2, d2, lay2, np.array(target), yaw, HOME_QPOS, iters=k, lam=lam)
            mujoco.mj_forward(m2, d2)
            r2 = mujoco.Renderer(m2, 240, 240)
            r2.update_scene(d2, camera="front")
            frames.append(r2.render())
        seqs.append(frames)
    from scripts.video_grid import grid
    grid(seqs, [f"λ={l}" for l in lams], "videos/ik_lambda.mp4", fps=4, gif_stride=1)
    import shutil
    shutil.move("videos/ik_lambda.gif", "docs/figs/ik_lambda.gif")
    print("→ docs/figs/ik_lambda.gif")


if __name__ == "__main__":
    fig_jacobian()
    fig_converge()
    fig_lambda()
    gif_lambda()
