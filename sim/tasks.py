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
    any_plate: bool = False        # True 时指令不指定盘子，放进任意盘子都算成功

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
            if min_dist > 0.06:                       # 放宽间距重试（物体多时工作区会挤）
                return _sample_xy(rng, n, min_dist * 0.85, existing)
            raise RuntimeError("采样不到不重叠的位置，把工作区放大或物体数量调小")
    return np.array(out)


# 「左 / 右」按**前视相机画面**定义：世界 −y 出现在画面左侧，+y 在右侧（实测 y=±0.2 → u=∓0.603）。
# 这个约定必须写死并说清楚——换成"机器人自己的左右"会正好相反。
SAME_COLOR_MIN_DY = 0.10          # 两个同色方块至少差这么多 y，"左/右"才说得清


def _positional_word(rng, cube_xy, idxs, tgt):
    """目标方块和别的同色方块并存时，用画面里的左右来指认。"""
    ys = sorted(idxs, key=lambda i: cube_xy[i][1])
    return "左边的" if tgt == ys[0] else "右边的"


def sample_task(rng, task_type="place", n_cubes=3, n_plates=2,
                same_color_prob=0.0, any_plate=False) -> TaskSpec:
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
    if n_cubes <= len(colors):
        cube_colors = colors[:n_cubes]
    else:
        # 干扰物比颜色多时允许重复，但**目标方块的颜色必须唯一**，否则指令有歧义
        cube_colors = colors + [str(c) for c in rng.choice(colors, n_cubes - len(colors))]
        rng.shuffle(cube_colors)
    # 盘子颜色从方块颜色里取，保证两种读法在同一场景下都说得通
    pool = list(dict.fromkeys(cube_colors))          # 去重后的方块颜色
    plate_colors = [str(c) for c in rng.choice(pool, min(n_plates, len(pool)), replace=False)]

    plate_xy = _sample_xy(rng, n_plates, 0.22)
    cube_xy = _sample_xy(rng, n_cubes, 0.11, existing=list(plate_xy))
    # 方块不能压在盘子上
    for i, c in enumerate(cube_xy):
        while any(np.linalg.norm(c - p) < PLATE_R + 0.05 for p in plate_xy):
            c = _sample_xy(rng, 1, 0.11, existing=list(plate_xy) + list(cube_xy))[0]
            cube_xy[i] = c

    # 让两个方块同色：这样"红色方块"就不够用了，必须靠位置词指认。
    # 训练集只有 12 条指令是本项目最大的局限（见 notes/03），这是扩大指令空间的第一步。
    dup = None
    if same_color_prob > 0 and n_cubes >= 2 and rng.random() < same_color_prob:
        i, j = rng.choice(n_cubes, 2, replace=False)
        # 两个同色方块的 y 必须拉开，否则"左/右"说不清
        if abs(cube_xy[i][1] - cube_xy[j][1]) < SAME_COLOR_MIN_DY:
            cube_xy[j][1] = cube_xy[i][1] + (SAME_COLOR_MIN_DY + 0.06) * (1 if cube_xy[i][1] < 0 else -1)
            cube_xy[j][1] = float(np.clip(cube_xy[j][1], *WORKSPACE["y"]))
        if abs(cube_xy[i][1] - cube_xy[j][1]) >= SAME_COLOR_MIN_DY:
            cube_colors[int(j)] = cube_colors[int(i)]
            dup = (int(i), int(j))

    if dup is not None:
        tc = int(rng.choice(dup))
    else:
        uniq = [i for i, c in enumerate(cube_colors) if cube_colors.count(c) == 1]
        tc = int(rng.choice(uniq)) if uniq else int(rng.integers(len(cube_colors)))
    # 目标盘子的颜色不等于目标方块的颜色，否则「把红方块放进红盘子」少了一半信息量
    cand = [i for i in range(len(plate_colors)) if plate_colors[i] != cube_colors[tc]]
    tp = int(rng.choice(cand)) if cand else int(rng.integers(len(plate_colors)))
    same = [i for i, c in enumerate(cube_colors) if c == cube_colors[tc]]
    qual = _positional_word(rng, cube_xy, same, tc) if len(same) > 1 else ""
    if any_plate:
        # 多峰任务：指令不说放进哪个盘子，专家**每局随机挑一个**，放进任意盘子都算成功。
        # 关键是这个选择必须和场景无关（不能挑最近的），否则策略能从图像里推出来，
        # 条件分布就又变回单峰了。见 notes/18。
        tp = int(rng.integers(len(plate_colors)))
        instr = f"把{qual}{cube_colors[tc]}色方块放进盘子"
    elif task_type == "lift":
        instr = f"拿起{qual}{cube_colors[tc]}色方块"
    else:
        instr = f"把{qual}{cube_colors[tc]}色方块放进{plate_colors[tp]}色盘子"
    return TaskSpec(task_type, cube_colors, plate_colors, cube_xy,
                    rng.uniform(-0.4, 0.4, n_cubes), plate_xy, tc, tp, instr, any_plate)
