"""环境讲解图：这一步到底发生了什么。

产出（docs/figs/）：
    env_annotated.png     前视/腕视图上标出 TCP、每个物体的世界坐标和它在观测里的位置
    env_action_chain.png  一个控制步的完整数值链：动作 → 末端目标 → IK → 关节指令 → 实际 TCP
    env_action_scale.gif  只改动作尺度 ACT_POS，同一串动作走出的轨迹怎么变

    python -m scripts.explain_env
"""
from __future__ import annotations

import numpy as np
import mujoco
import imageio
import matplotlib.pyplot as plt

from scripts._style import save, annotate, C
from sim.assets import TABLE_TOP, CUBE_HALF, WORKSPACE
from sim.tabletop_env import TabletopEnv, ACT_POS, ACT_YAW, EE_BOX
from sim.ik import solve_ik
from sim.tasks import sample_task


def world_to_pixel(model, data, cam_name, p, hw):
    """世界坐标 → 图像像素。相机看自己的 -z，x 向右，y 向上。"""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R = data.cam_xmat[cid].reshape(3, 3)
    t = data.cam_xpos[cid]
    pc = R.T @ (np.asarray(p) - t)
    f = (hw / 2) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
    u = hw / 2 + f * pc[0] / (-pc[2])
    v = hw / 2 - f * pc[1] / (-pc[2])
    return np.array([u, v]), -pc[2]


def fig_annotated(env):
    hw = env.img_hw
    front, wrist = env.render("front"), env.render("wrist")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, img, cam in zip(axes, (front, wrist), ("front", "wrist")):
        ax.imshow(img)
        ax.set_title(f"{cam}  {img.shape}  uint8", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    s = env.spec
    ax = axes[0]
    for i, cube in enumerate(env.layout["cubes"]):
        p = env.cube_pos(i)
        uv, _ = world_to_pixel(env.model, env.data, "front", p, hw)
        tag = "★目标" if i == s.target_cube else ""
        annotate(ax, uv, f"cube{i} {s.cube_colors[i]}{tag}\n({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})",
                 (uv[0] + 24, uv[1] - 70 + 46 * i), color=C["front"])
    p = env.plate_pos(s.target_plate)
    uv, _ = world_to_pixel(env.model, env.data, "front", p, hw)
    annotate(ax, uv, f"plate {s.plate_colors[s.target_plate]}\n({p[0]:.3f}, {p[1]:.3f})",
             (uv[0] - 60, uv[1] + 30), color=C["act"])
    tcp = env.tcp()
    uv, _ = world_to_pixel(env.model, env.data, "front", tcp, hw)
    annotate(ax, uv, f"TCP state[0:3]\n({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f})",
             (uv[0] - 70, uv[1] - 42), color=C["warn"])

    # 工作区（物体采样范围）投影成一个四边形
    corners = [(WORKSPACE["x"][i], WORKSPACE["y"][j], TABLE_TOP)
               for i, j in ((0, 0), (0, 1), (1, 1), (1, 0), (0, 0))]
    uvs = np.array([world_to_pixel(env.model, env.data, "front", c, hw)[0] for c in corners])
    ax.plot(uvs[:, 0], uvs[:, 1], "--", color=C["grey"], lw=1.2)
    ax.text(uvs[:, 0].mean(), uvs[:, 1].max() + 6, "物体采样工作区", color=C["grey"],
            fontsize=8, ha="center")

    tcp_uv, _ = world_to_pixel(env.model, env.data, "wrist", tcp, hw)
    annotate(axes[1], tcp_uv, "TCP（两指中心）", (tcp_uv[0] - 55, tcp_uv[1] - 30), color=C["warn"])

    fig.suptitle(f"观测的两张图 · 指令「{s.instruction}」\n"
                 f"state(7) = [tcp_x, tcp_y, tcp_z, sin yaw, cos yaw, 夹爪开度, 上一步夹爪指令] = "
                 f"[{', '.join(f'{x:.3f}' for x in env.state())}]", fontsize=10)
    save(fig, "docs/figs/env_annotated.png")


def fig_action_chain(env):
    """一个控制步的数值链，全是真实数字。"""
    a = np.array([0.6, -0.35, -0.8, 0.0, 1.0])
    ee_before = env.ee_pos.copy()
    tcp_before = env.tcp().copy()
    q_before = env.data.qpos[env.layout["arm_qadr"]].copy()
    ee_target = np.clip(ee_before + a[:3] * ACT_POS,
                        [EE_BOX[k][0] for k in "xyz"], [EE_BOX[k][1] for k in "xyz"])
    obs, r, done, info = env.step(a)
    q_ctrl = env.data.ctrl[:7].copy()
    tcp_after = env.tcp().copy()

    rows = [
        ("策略输出 a", f"[{', '.join(f'{x:+.2f}' for x in a)}]", "范围 [-1,1]⁵"),
        ("① 位移增量", f"a[0:3] × {ACT_POS} = {np.round(a[:3]*ACT_POS, 4).tolist()}", "米"),
        ("② 末端目标（上一目标 + 增量，再夹到工作箱）",
         f"{np.round(ee_before,4).tolist()} → {np.round(ee_target,4).tolist()}", "米"),
        ("③ IK 解出的关节目标", f"{np.round(q_ctrl, 3).tolist()}", "rad，7 个手臂关节"),
        ("   IK 相对当前关节的变化 Δq", f"{np.round(q_ctrl - q_before, 4).tolist()}", "rad"),
        ("   IK 残差（目标 vs 解出的位姿）", f"{info['ik_pos_err']*1000:.2f} mm", "> 5 mm 就该查奇异位形"),
        ("④ 夹爪指令", f"a[4]={a[4]:+.1f} → ctrl[7]={env.data.ctrl[7]:.0f}", "255 张开 / 0 闭合"),
        ("⑤ 25 个物理子步后的实际 TCP",
         f"{np.round(tcp_before,4).tolist()} → {np.round(tcp_after,4).tolist()}", "米"),
        ("   实际位移 vs 指令位移",
         f"{np.linalg.norm(tcp_after-tcp_before)*1000:.1f} mm vs "
         f"{np.linalg.norm(ee_target-ee_before)*1000:.1f} mm",
         "位置控制器跟不满，这个差值就是跟踪滞后"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    ax.set_title("一个控制步（1/20 秒 = 25 个物理子步）里发生的事", fontsize=11)
    for i, (k, v, note) in enumerate(rows):
        y = 1 - i * 0.108
        ax.text(0.0, y, k, fontsize=9, color=C["grey"], va="top")
        ax.text(0.40, y, v, fontsize=9, family="monospace", va="top")
        ax.text(0.40, y - 0.045, note, fontsize=7.5, color=C["warn"], va="top")
    save(fig, "docs/figs/env_action_chain.png")


def gif_action_scale(seed=7):
    """只改 ACT_POS：同一串"往前走 10 步"的动作，走出的距离和抖动怎么变。"""
    import sim.tabletop_env as T
    frames, curves = [], {}
    scales = [0.01, 0.02, 0.04, 0.08]
    orig = T.ACT_POS
    for sc in scales:
        T.ACT_POS = sc
        env = TabletopEnv(seed=seed, img_hw=192, max_steps=30)
        env.reset(seed=seed)
        traj = [env.tcp().copy()]
        shots = []
        for t in range(20):
            env.step(np.array([0.0, 1.0, -0.6, 0.0, 1.0]))
            traj.append(env.tcp().copy())
            if t % 4 == 0:
                shots.append(env.render("front"))
        curves[sc] = np.array(traj)
        frames.append((sc, shots))
    T.ACT_POS = orig

    n = min(len(s) for _, s in frames)
    gif = []
    for i in range(n):
        row = np.concatenate([s[i] for _, s in frames], axis=1)
        gif.append(row)
    imageio.mimsave("docs/figs/env_action_scale.gif", gif, fps=2, loop=0)
    print("→ docs/figs/env_action_scale.gif  （左到右 ACT_POS =", scales, "）")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for sc, tr in curves.items():
        ax.plot(tr[:, 1], tr[:, 2], "-o", ms=2.5, label=f"ACT_POS={sc}")
    ax.set_xlabel("TCP y (m)"); ax.set_ylabel("TCP z (m)")
    ax.set_title("同一串动作 [0, +1, -0.6] × 20 步，动作尺度决定走多远")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save(fig, "docs/figs/env_action_scale.png")


if __name__ == "__main__":
    env = TabletopEnv(seed=5, img_hw=384)
    env.reset(seed=5)
    for _ in range(6):                      # 先走几步让画面里手臂在工作区上方
        env.step(np.array([0.3, -0.2, -0.5, 0.0, 1.0]))
    fig_annotated(env)
    fig_action_chain(env)
    gif_action_scale()
