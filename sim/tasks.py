"""任务与场景随机化：决定"桌上有什么"和"这一局要做什么"。

一局（episode）由一个 TaskSpec 完全确定，TaskSpec 可以序列化进数据集，
所以任何一条演示都能精确复现（调试和"修复前/修复后"对比图靠这个）。
"""
from __future__ import annotations

import dataclasses
import numpy as np

from sim.assets import COLORS, WORKSPACE, TABLE_TOP, CUBE_HALF, PLATE_R

TASK_TYPES = ("lift", "place")


@dataclasses.dataclass
class TaskSpec:
    task_type: str                 # lift / place
    cube_colors: list              # 桌上方块的颜色，按 cube0..cubeN
    plate_colors: list             # 桌上盘子的颜色
    cube_xy: np.ndarray            # (N,2)
    cube_yaw: np.ndarray           # (N,)
    plate_xy: np.ndarray           # (M,2)
    target_cube: int
    target_plate: int
    instruction: str

    def to_dict(self):
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def _sample_xy(rng, n, min_dist, existing=None):
    pts = list(existing) if existing else []
    out = []
    for _ in range(n):
        for _try in range(200):
            p = np.array([rng.uniform(*WORKSPACE["x"]), rng.uniform(*WORKSPACE["y"])])
            if all(np.linalg.norm(p - q) > min_dist for q in pts):
                pts.append(p)
                out.append(p)
                break
        else:
            raise RuntimeError("采样不到不重叠的位置，把工作区放大或物体数量调小")
    return np.array(out)


def sample_task(rng, task_type="place", n_cubes=3, n_plates=1) -> TaskSpec:
    colors = list(COLORS)
    rng.shuffle(colors)
    cube_colors = colors[:n_cubes]
    plate_colors = colors[n_cubes:n_cubes + n_plates]
    if len(plate_colors) < n_plates:      # 颜色不够就允许盘子和方块同色
        plate_colors = list(rng.choice(colors, n_plates, replace=False))

    plate_xy = _sample_xy(rng, n_plates, 0.22)
    cube_xy = _sample_xy(rng, n_cubes, 0.11, existing=list(plate_xy))
    # 方块不能压在盘子上
    for i, c in enumerate(cube_xy):
        while any(np.linalg.norm(c - p) < PLATE_R + 0.05 for p in plate_xy):
            c = _sample_xy(rng, 1, 0.11, existing=list(plate_xy) + list(cube_xy))[0]
            cube_xy[i] = c

    tc = int(rng.integers(n_cubes))
    tp = int(rng.integers(n_plates))
    if task_type == "lift":
        instr = f"拿起{cube_colors[tc]}色方块"
    else:
        instr = f"把{cube_colors[tc]}色方块放进{plate_colors[tp]}色盘子"
    return TaskSpec(task_type, cube_colors, plate_colors, cube_xy,
                    rng.uniform(-0.4, 0.4, n_cubes), plate_xy, tc, tp, instr)
