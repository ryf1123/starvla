"""阻尼最小二乘 IK：给定 TCP 目标位姿，解 7 个手臂关节角。

解的优化问题（每次迭代）：
    min_Δq  ‖J Δq − e‖² + λ²‖Δq‖²        →  Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
其中 e = [位置误差(3); 姿态误差(3, 轴角)]，J 是 TCP 站点对 7 个手臂关节的 6×7 雅可比。

λ 的作用：靠近奇异位形时 J Jᵀ 接近奇异，λ 把解拉小、拉平滑；λ 越大越平滑越不准。
docs/figs/ik_lambda_sweep.gif 是同一帧不同 λ 的对比。
"""
from __future__ import annotations

import numpy as np
import mujoco

# 手爪竖直向下时 TCP 的姿态（见 notes/：由 HOME_QPOS 前向运动学读出）
R_DOWN = np.array([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])


def yaw_to_R(yaw: float) -> np.ndarray:
    """绕世界 z 轴转 yaw 之后的目标姿态。"""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ R_DOWN


def rot_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """姿态误差，表示成世界系下的轴角向量（3,）。"""
    R_err = R_tgt @ R_cur.T
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R_err.flatten())
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def solve_ik(model, data, layout, p_tgt, yaw_tgt, q_init=None,
             iters=30, lam=0.15, pos_tol=1e-3, rot_tol=1e-2, max_step=0.3):
    """就地在 data 上迭代，返回 (q_arm, 位置残差, 姿态残差)。

    调用者负责保存/恢复 data —— 这个函数会写 data.qpos 并调用 mj_kinematics。
    """
    adr = layout["arm_qadr"]
    sid = layout["tcp_site"]
    if q_init is not None:
        data.qpos[adr] = q_init
    R_tgt = yaw_to_R(yaw_tgt)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    dof = np.array([model.jnt_dofadr[model.body_jntadr[0]]])  # 占位，下面重算
    dof = layout["arm_dofadr"]

    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        p_cur = data.site_xpos[sid].copy()
        R_cur = data.site_xmat[sid].reshape(3, 3).copy()
        e_p = p_tgt - p_cur
        e_r = rot_error(R_cur, R_tgt)
        if np.linalg.norm(e_p) < pos_tol and np.linalg.norm(e_r) < rot_tol:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, dof], jacr[:, dof]])          # 6×7
        e = np.concatenate([e_p, e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q = data.qpos[adr] + dq
        q = np.clip(q, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
        data.qpos[adr] = q

    mujoco.mj_kinematics(model, data)
    p_cur = data.site_xpos[sid].copy()
    R_cur = data.site_xmat[sid].reshape(3, 3).copy()
    return (data.qpos[adr].copy(),
            float(np.linalg.norm(p_tgt - p_cur)),
            float(np.linalg.norm(rot_error(R_cur, R_tgt))))
