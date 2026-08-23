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


def sample_task(rng, task_type="place", n_cubes=3, n_plates=2) -> TaskSpec:
    """采一局任务。

    关键设计：**盘子的颜色从方块用过的颜色里选**。
    如果盘子颜色永远不和方块重复（第一版就是这样），那么「颜色集合」就足以定位目标——
    指令里出现的两个颜色，能在桌上找到方块的那个就是被抓的，另一个必然是盘子。
    语序不携带任何信息，于是模型学到的最小充分表示就是一个词袋，
    实测「红→黄」和「黄→红」的编码距离只有 0.046（换一组颜色是 2.72）。

    让盘子和方块共用颜色之后，「把红色方块放进黄色盘子」和「把黄色方块放进红色盘子」
    在同一个场景里都成立且结果不同——语序才真正成为必需的信息。
    """
    colors = list(COLORS)
    rng.shuffle(colors)
    cube_colors = colors[:n_cubes]
    # 盘子颜色从方块颜色里取，保证两种读法在同一场景下都说得通
    plate_colors = [str(c) for c in rng.choice(cube_colors, min(n_plates, n_cubes), replace=False)]

    plate_xy = _sample_xy(rng, n_plates, 0.22)
    cube_xy = _sample_xy(rng, n_cubes, 0.11, existing=list(plate_xy))
    # 方块不能压在盘子上
    for i, c in enumerate(cube_xy):
        while any(np.linalg.norm(c - p) < PLATE_R + 0.05 for p in plate_xy):
            c = _sample_xy(rng, 1, 0.11, existing=list(plate_xy) + list(cube_xy))[0]
            cube_xy[i] = c

    tc = int(rng.integers(n_cubes))
    # 目标盘子的颜色不等于目标方块的颜色，否则「把红方块放进红盘子」少了一半信息量
    cand = [i for i in range(len(plate_colors)) if plate_colors[i] != cube_colors[tc]]
    tp = int(rng.choice(cand)) if cand else int(rng.integers(len(plate_colors)))
    if task_type == "lift":
        instr = f"拿起{cube_colors[tc]}色方块"
    else:
        instr = f"把{cube_colors[tc]}色方块放进{plate_colors[tp]}色盘子"
    return TaskSpec(task_type, cube_colors, plate_colors, cube_xy,
                    rng.uniform(-0.4, 0.4, n_cubes), plate_xy, tc, tp, instr)
