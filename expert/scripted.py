"""脚本专家：用特权信息（物体真实位姿）走一套状态机，产生演示数据。

它是"老师"，不是我们要的东西——策略只看图像和指令，看不到这些坐标。
专家存在的意义有两个：给模仿学习提供数据；给闭环评测一个成功率上界。

状态机（place 任务，7 段）：
    0 hover   移到目标方块正上方 0.12 m，夹爪张开，偏航对齐方块朝向
    1 descend 下到抓取高度（TCP 落在方块中心略上）
    2 close   闭合夹爪，保持 4 步等接触稳定
    3 lift    抬到 0.16 m
    4 carry   平移到目标盘子正上方
    5 place   下降到方块底面刚好在盘子上方
    6 open    张开夹爪，抬手离开

每一段都是"给一个航点，用比例控制器走过去"，误差小于阈值就进下一段。
噪声（--noise）加在动作上，让数据覆盖航点附近的状态，否则策略一旦偏离就没见过。
"""
from __future__ import annotations

import numpy as np

from sim.assets import TABLE_TOP, CUBE_HALF
from sim.tabletop_env import ACT_POS, ACT_YAW

HOVER_H = 0.12
LIFT_H = 0.16
GRASP_DZ = 0.004          # TCP 目标相对方块中心的高度偏移
PLACE_DZ = 0.035          # 松手时方块底面离盘面的余量


def _wrap_grasp_yaw(phi):
    """方块绕 z 转 φ，夹爪只需对齐到 90° 的余数，再折到 [-45°, 45°]。"""
    y = np.mod(phi, np.pi / 2)
    return y - np.pi / 2 if y > np.pi / 4 else y


class ScriptedExpert:
    def __init__(self, env, rng=None, noise=0.0):
        self.env, self.rng = env, np.random.default_rng() if rng is None else rng
        self.noise = noise
        self.reset()

    def reset(self):
        self.phase = 0
        self.hold = 0

    def act(self):
        env = self.env
        s = env.spec
        cube = env.cube_pos(s.target_cube)
        plate = env.plate_pos(s.target_plate)
        yaw_t = _wrap_grasp_yaw(s.cube_yaw[s.target_cube])
        p = env.ee_pos                                  # 用"指令位置"而不是实际 TCP，避免 IK 滞后累积

        grip = 1.0                                       # +1 张开
        if self.phase == 0:
            tgt = np.array([cube[0], cube[1], TABLE_TOP + HOVER_H])
            if np.linalg.norm(p - tgt) < 0.012 and abs(env.ee_yaw - yaw_t) < 0.06:
                self.phase = 1
        elif self.phase == 1:
            tgt = np.array([cube[0], cube[1], cube[2] + GRASP_DZ])
            if np.linalg.norm(p - tgt) < 0.008:
                self.phase, self.hold = 2, 0
        elif self.phase == 2:
            tgt = np.array([cube[0], cube[1], cube[2] + GRASP_DZ])
            grip = -1.0
            self.hold += 1
            if self.hold >= 5:
                self.phase = 3
        elif self.phase == 3:
            tgt = np.array([p[0], p[1], TABLE_TOP + LIFT_H])
            grip = -1.0
            if p[2] > TABLE_TOP + LIFT_H - 0.01:
                self.phase = 4
        elif self.phase == 4:
            tgt = np.array([plate[0], plate[1], TABLE_TOP + LIFT_H])
            grip = -1.0
            if np.linalg.norm(p[:2] - plate[:2]) < 0.012:
                self.phase = 5
        elif self.phase == 5:
            tgt = np.array([plate[0], plate[1], TABLE_TOP + 0.01 + 2 * CUBE_HALF + PLACE_DZ])
            grip = -1.0
            if abs(p[2] - tgt[2]) < 0.008:
                self.phase, self.hold = 6, 0
        else:
            tgt = np.array([plate[0], plate[1], TABLE_TOP + LIFT_H])
            grip = 1.0
            self.hold += 1
            if self.hold < 3:
                tgt = p.copy()                            # 先松手，站住别动

        if s.task_type == "lift" and self.phase >= 3:
            tgt = np.array([cube[0], cube[1], TABLE_TOP + 0.20])
            grip = -1.0

        a = np.zeros(5)
        a[:3] = np.clip((tgt - p) / ACT_POS, -1, 1)
        a[3] = np.clip((yaw_t - env.ee_yaw) / ACT_YAW, -1, 1) if self.phase <= 1 else 0.0
        a[4] = grip
        if self.noise > 0:
            a[:4] = np.clip(a[:4] + self.rng.normal(0, self.noise, 4), -1, 1)
        return a
