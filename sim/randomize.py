"""域随机化：仿真里最便宜、文献里收益最大的一件事。

依据（见 docs/lit-review.md）：
  - 《Decomposing the Generalization Gap》(ICRA 2024) 把泛化因素从易到难排成：
    背景 < 干扰物/光照 < 物体纹理 < 桌面纹理/位置 < **相机位姿**。
    真机上换相机位姿成功率从 91.7% 掉到 45.8%——是最难的一项。
  - 《Data Scaling Laws in Imitation Learning》(ICLR 2025)：泛化对**环境数量**呈幂律，
    而对"每个环境的演示数"很快饱和。32 环境 × 每个 50 条 → 新环境新物体 90%。

也就是说：**与其在一个固定场景刷 800 条演示，不如在几十个随机化场景各采几十条。**
真机上"多样性"是最贵的东西，仿真里几乎免费——这是本项目相对真机工作的结构性优势。

用法：`TabletopEnv(dr=DomainRandomizer(level=1.0))`，每次 reset 时随机化一遍。
所有随机化都是**运行时改 model 的字段**，不重新编译模型（编译一次约 1 秒，每局都编译受不了）。
"""
from __future__ import annotations

import dataclasses
import numpy as np
import mujoco

FRONT_POS = np.array([1.02, 0.0, 0.78])       # 基准相机位置（sim/assets.py 里的默认值）
FRONT_FOVY = 52.0


@dataclasses.dataclass
class DomainRandomizer:
    level: float = 1.0        # 总强度，0 = 关掉
    camera: bool = True       # 相机位姿 + 视场角（最难泛化的因素）
    lighting: bool = True     # 光源位置和强度
    table: bool = True        # 桌面颜色
    cube_shade: bool = True   # 方块明暗（不动色相——色相是标签）

    def apply(self, model, rng, layout):
        L = self.level
        if L <= 0:
            return
        if self.camera:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
            model.cam_pos[cid] = FRONT_POS + rng.uniform(-1, 1, 3) * np.array([0.10, 0.14, 0.09]) * L
            model.cam_fovy[cid] = FRONT_FOVY + rng.uniform(-4, 4) * L
            # 腕部相机只轻微抖：它是刚性装在手上的，真机上不会乱动
            wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            model.cam_pos[wid] = np.array([0.0, -0.055, 0.0]) + rng.uniform(-1, 1, 3) * 0.006 * L
        if self.lighting:
            for i in range(model.nlight):
                model.light_pos[i] = np.array([0.0, 0.0, 1.5]) + rng.uniform(-1, 1, 3) * np.array([0.8, 0.8, 0.3]) * L
                d = float(np.clip(0.7 + rng.uniform(-0.35, 0.35) * L, 0.25, 1.2))
                model.light_diffuse[i] = [d, d, d]
                a = float(np.clip(0.25 + rng.uniform(-0.15, 0.15) * L, 0.05, 0.5))
                model.light_ambient[i] = [a, a, a]
        if self.table:
            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "table_mat")
            if mid >= 0:
                # 主要改明暗、只轻微改色调：桌面如果随机成红/绿，会和盘子的颜色标签混淆
                base = np.array([0.72, 0.66, 0.55])
                shade = 1.0 + rng.uniform(-0.30, 0.30) * L
                model.mat_rgba[mid, :3] = np.clip(base * shade + rng.uniform(-1, 1, 3) * 0.05 * L, 0.15, 0.95)
        if self.cube_shade:
            # 只改明暗（乘一个标量），不改色相——色相就是指令里的那个词，动了等于擦标签
            for kind in ("cubes", "plates"):
                for o in layout[kind]:
                    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{o['name']}_geom")
                    rgb = model.geom_rgba[gid, :3]
                    model.geom_rgba[gid, :3] = np.clip(rgb * (1.0 + rng.uniform(-0.18, 0.18) * L), 0.02, 1.0)

    def spec(self):
        return dataclasses.asdict(self)
