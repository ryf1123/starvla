"""桌面操作环境：VLA 的"世界"。

一句话：策略每 1/20 秒看两张图 + 一句话 + 自身状态，输出末端的一小步位移和夹爪开合。

时间尺度
    物理 mj timestep = 0.002 s（500 Hz），控制 20 Hz，decimation = 25。
    每个控制步内 IK 只解一次，25 个物理子步用同一组关节位置目标。

动作 a ∈ [-1,1]^5
    a[0:3]  末端位移增量，× ACT_POS = 0.04 m
    a[3]    末端偏航增量，× ACT_YAW = 0.30 rad
    a[4]    夹爪：> 0 张开，≤ 0 闭合（训练时按 ±1 记录）

观测
    front (H,W,3) uint8   桌面前方固定相机
    wrist (H,W,3) uint8   腕部相机
    state (7,) float32    [tcp_x, tcp_y, tcp_z, sin yaw, cos yaw, 夹爪开度, 上一步夹爪指令]
    instruction str       中文指令，例如「把红色方块放进黄色盘子」

设计取舍：动作用末端增量（不是关节角），因为它和相机看到的东西同一个坐标系，
数据效率高得多；代价是需要 IK，而 IK 会在奇异位形附近失败——这一点在 notes 里有例子。
"""
from __future__ import annotations

import numpy as np
import mujoco

from sim.assets import (build_scene, HOME_QPOS, TABLE_TOP, CUBE_HALF, PLATE_R,
                        COLORS, WORKSPACE)
from sim.ik import solve_ik
from sim.tasks import TaskSpec, sample_task

CTRL_HZ = 20
ACT_POS = 0.04
ACT_YAW = 0.30
GRIP_OPEN, GRIP_CLOSE = 255.0, 0.0
EE_BOX = dict(x=(0.32, 0.72), y=(-0.32, 0.32), z=(TABLE_TOP + 0.005, TABLE_TOP + 0.35))
HOME_TCP = np.array([0.50, 0.0, TABLE_TOP + 0.22])


class TabletopEnv:
    def __init__(self, n_cubes=3, n_plates=2, img_hw=128, max_steps=140,
                 task_type="place", seed=0, render_cams=("front", "wrist"),
                 same_color_prob=0.0, dr=None, home_jitter=True):
        self.n_cubes, self.n_plates = n_cubes, n_plates
        self.img_hw, self.max_steps, self.task_type = img_hw, max_steps, task_type
        self.same_color_prob = same_color_prob
        self.dr = dr                                   # DomainRandomizer 或 None
        self.home_jitter = home_jitter                 # 关掉可以复现"没有抖动"时的数据分布
        self.render_cams = render_cams
        self.model, self.layout = build_scene(["红"] * n_cubes, ["黄"] * n_plates, img_hw)
        self.data = mujoco.MjData(self.model)
        self.decimation = int(round(1.0 / (CTRL_HZ * self.model.opt.timestep)))
        self.renderer = mujoco.Renderer(self.model, img_hw, img_hw)
        self.rng = np.random.default_rng(seed)
        self._geom_id = {}
        for kind in ("cubes", "plates"):
            for i, o in enumerate(self.layout[kind]):
                self._geom_id[o["name"]] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{o['name']}_geom")
        self.spec: TaskSpec | None = None

    # ---------------------------------------------------------------- reset
    def reset(self, spec: TaskSpec | None = None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.spec = spec or sample_task(self.rng, self.task_type, self.n_cubes, self.n_plates,
                                        self.same_color_prob)
        s = self.spec
        mujoco.mj_resetData(self.model, self.data)

        # 颜色是运行时改的：模型只编译一次，换任务只改 rgba，省掉重新编译
        for i, c in enumerate(s.cube_colors):
            self.model.geom_rgba[self._geom_id[f"cube{i}"]] = [*COLORS[c], 1.0]
        for i, c in enumerate(s.plate_colors):
            self.model.geom_rgba[self._geom_id[f"plate{i}"]] = [*COLORS[c], 1.0]
        for i, xy in enumerate(s.plate_xy):
            self.model.body_pos[self.layout["plates"][i]["bid"]] = [xy[0], xy[1], TABLE_TOP + 0.005]

        # 域随机化：相机位姿、光照、桌面颜色、方块明暗。放在颜色设置**之后**，
        # 因为方块明暗是在基准颜色上乘一个系数。
        if self.dr is not None:
            self.dr.apply(self.model, self.rng, self.layout)

        self.data.qpos[self.layout["arm_qadr"]] = HOME_QPOS
        self.data.qpos[self.layout["finger_qadr"]] = 0.04
        for i, cube in enumerate(self.layout["cubes"]):
            a = cube["qadr"]
            yaw = s.cube_yaw[i]
            self.data.qpos[a:a + 3] = [s.cube_xy[i][0], s.cube_xy[i][1], TABLE_TOP + CUBE_HALF]
            self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        mujoco.mj_forward(self.model, self.data)
        # 把末端摆到 home 位姿（IK 一次），夹爪张开。
        # home 位姿要抖动：不抖的话每局到达方块的步数几乎一样，
        # 策略会学成"第 10 步闭合夹爪"这个时序，而不是"到位了才闭合"这个条件。
        j = 1.0 if self.home_jitter else 0.0
        tcp0 = HOME_TCP + self.rng.uniform(-1, 1, 3) * np.array([0.06, 0.08, 0.04]) * j
        yaw0 = float(self.rng.uniform(-0.25, 0.25)) * j
        q, _, _ = solve_ik(self.model, self.data, self.layout, tcp0, yaw0, HOME_QPOS, iters=60)
        self.data.qpos[self.layout["arm_qadr"]] = q
        self.data.ctrl[:7] = q
        self.data.ctrl[7] = GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)

        self.ee_pos = self.tcp().copy()
        self.ee_yaw = yaw0
        self.grip_cmd = 1.0
        self.t = 0
        self._success_hold = 0
        return self.obs()

    # ----------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.ee_pos = self.ee_pos + a[:3] * ACT_POS
        for i, k in enumerate("xyz"):
            self.ee_pos[i] = np.clip(self.ee_pos[i], *EE_BOX[k])
        self.ee_yaw = float(np.clip(self.ee_yaw + a[3] * ACT_YAW, -np.pi / 2, np.pi / 2))
        self.grip_cmd = 1.0 if a[4] > 0 else -1.0

        q_now = self.data.qpos[self.layout["arm_qadr"]].copy()
        qpos_backup = self.data.qpos.copy()
        q_tgt, pos_err, _ = solve_ik(self.model, self.data, self.layout,
                                     self.ee_pos, self.ee_yaw, q_now, iters=12)
        self.data.qpos[:] = qpos_backup                    # IK 只借用 data 算，不改真实状态
        self.data.ctrl[:7] = q_tgt
        self.data.ctrl[7] = GRIP_OPEN if self.grip_cmd > 0 else GRIP_CLOSE
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self.t += 1
        ok = self.success()
        self._success_hold = self._success_hold + 1 if ok else 0
        done = self._success_hold >= 3 or self.t >= self.max_steps
        info = {"success": self._success_hold >= 3, "ik_pos_err": pos_err,
                "timeout": self.t >= self.max_steps}
        return self.obs(), float(ok), done, info

    # ------------------------------------------------------------ 观测/状态
    def tcp(self):
        return self.data.site_xpos[self.layout["tcp_site"]].copy()

    def grip_width(self):
        return float(self.data.qpos[self.layout["finger_qadr"]].sum())

    def cube_pos(self, i):
        a = self.layout["cubes"][i]["qadr"]
        return self.data.qpos[a:a + 3].copy()

    def plate_pos(self, i):
        return self.model.body_pos[self.layout["plates"][i]["bid"]].copy()

    def state(self):
        p = self.tcp()
        return np.array([p[0], p[1], p[2], np.sin(self.ee_yaw), np.cos(self.ee_yaw),
                         self.grip_width(), self.grip_cmd], dtype=np.float32)

    def render(self, cam):
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def obs(self):
        o = {"state": self.state(), "instruction": self.spec.instruction}
        for cam in self.render_cams:
            o[cam] = self.render(cam)
        return o

    def world_to_pixel(self, cam, p):
        """世界坐标 → 归一化图像坐标 [-1,1]²，外加一个"在画面内"的标志。

        相机沿自己的 −z 看，+x 右、+y 上：
            p_cam = R_camᵀ (p − t_cam);  f = (hw/2)/tan(fovy/2)
            u = hw/2 + f·x/(−z);  v = hw/2 − f·y/(−z)
        用 data.cam_* 而不是 model.cam_*——targetbody 模式下朝向是运行时算的。
        """
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        R = self.data.cam_xmat[cid].reshape(3, 3)
        pc = R.T @ (np.asarray(p) - self.data.cam_xpos[cid])
        hw = self.img_hw
        f = (hw / 2) / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
        if -pc[2] < 1e-3:
            return np.array([0.0, 0.0], np.float32), 0.0
        u = hw / 2 + f * pc[0] / (-pc[2])
        v = hw / 2 - f * pc[1] / (-pc[2])
        inside = float(0 <= u < hw and 0 <= v < hw)
        return np.array([u / hw * 2 - 1, v / hw * 2 - 1], np.float32), inside

    def privileged(self):
        """特权信息：目标方块和目标盘子在两个相机里的像素坐标（策略看不到，只用来做辅助监督）。

        布局 (10,)：[前视目标方块 uv(2), 可见(1), 腕视目标方块 uv(2), 可见(1),
                     前视目标盘子 uv(2), 可见(1), 目标方块离桌面高度(1)]
        """
        s = self.spec
        c = self.cube_pos(s.target_cube)
        pl = self.plate_pos(s.target_plate)
        out = []
        for cam, p in (("front", c), ("wrist", c), ("front", pl)):
            uv, vis = self.world_to_pixel(cam, p)
            out += [uv[0], uv[1], vis]
        out.append(float(c[2] - TABLE_TOP))
        return np.array(out, np.float32)

    # ------------------------------------------------------------- 成功判据
    def success(self):
        s = self.spec
        c = self.cube_pos(s.target_cube)
        if s.task_type == "lift":
            return bool(c[2] > TABLE_TOP + CUBE_HALF + 0.08)
        p = self.plate_pos(s.target_plate)
        in_xy = np.linalg.norm(c[:2] - p[:2]) < PLATE_R - 0.015
        on_plate = abs(c[2] - (TABLE_TOP + 0.01 + CUBE_HALF)) < 0.03
        released = self.grip_width() > 0.06
        return bool(in_xy and on_plate and released)
