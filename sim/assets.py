"""程序化搭建桌面操作场景（Franka Panda + 方块 + 盘子 + 两路相机）。

为什么全部用 Python 的 MjSpec 生成而不是写死 XML：
场景要按任务随机化（几个方块、什么颜色、放哪儿），生成式写法让"改一个变量看结果"
这件事只改一个参数，而不是改 XML 文本。

坐标约定（世界系，单位 m）：
  机器人底座在原点，桌面顶面 z = TABLE_TOP = 0.30，
  物体工作区 x ∈ [0.35, 0.65]，y ∈ [-0.22, 0.22]。
"""
from __future__ import annotations

import numpy as np
import mujoco

MENAGERIE = "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"

TABLE_CENTER = np.array([0.55, 0.0, 0.15])
TABLE_HALF = np.array([0.30, 0.45, 0.15])
TABLE_TOP = float(TABLE_CENTER[2] + TABLE_HALF[2])   # 0.30

CUBE_HALF = 0.022
PLATE_R = 0.065

# Panda 手掌坐标系原点到两指中心（TCP）的距离，来自 franka 官方手册
TCP_OFFSET = 0.1034

WORKSPACE = dict(x=(0.36, 0.66), y=(-0.22, 0.22))

COLORS = {
    "红": (0.85, 0.15, 0.15),
    "绿": (0.15, 0.70, 0.20),
    "蓝": (0.15, 0.30, 0.85),
    "黄": (0.90, 0.75, 0.10),
}

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.9, 0.0, 1.9, -0.7853])


def build_scene(cube_colors, plate_colors, img_hw=128, seed_geom=True):
    """返回 (model, layout)。layout 记录每个物体的名字和 qpos 里的地址。"""
    spec = mujoco.MjSpec.from_file(MENAGERIE)

    # --- 桌子 -------------------------------------------------------------
    mat = spec.add_material()
    mat.name = "table_mat"
    mat.rgba = [0.72, 0.66, 0.55, 1.0]
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = TABLE_CENTER
    g = table.add_geom()
    g.name = "table_top"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = TABLE_HALF
    g.material = "table_mat"
    g.friction = [1.0, 0.02, 0.001]

    layout = {"cubes": [], "plates": []}

    # --- 盘子（目标容器，静态） -------------------------------------------
    for i, cname in enumerate(plate_colors):
        b = spec.worldbody.add_body()
        b.name = f"plate{i}"
        b.pos = [0.5, 0.3 * (i * 2 - 1), TABLE_TOP + 0.005]
        gg = b.add_geom()
        gg.name = f"plate{i}_geom"
        gg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        gg.size = [PLATE_R, 0.005, 0.0]
        gg.rgba = [*COLORS[cname], 1.0]
        gg.contype, gg.conaffinity = 4, 4     # 只和方块碰，不和手指抢接触
        layout["plates"].append({"name": b.name, "color": cname})

    # --- 方块（可抓，freejoint） ------------------------------------------
    for i, cname in enumerate(cube_colors):
        b = spec.worldbody.add_body()
        b.name = f"cube{i}"
        b.pos = [0.5, 0.1 * i, TABLE_TOP + CUBE_HALF]
        b.add_freejoint()
        gg = b.add_geom()
        gg.name = f"cube{i}_geom"
        gg.type = mujoco.mjtGeom.mjGEOM_BOX
        gg.size = [CUBE_HALF] * 3
        gg.rgba = [*COLORS[cname], 1.0]
        gg.mass = 0.05
        gg.friction = [1.4, 0.05, 0.002]
        gg.solref = [0.01, 1.0]
        gg.condim = 4                          # 允许扭转摩擦，夹住不打转
        layout["cubes"].append({"name": b.name, "color": cname})

    # --- 相机 --------------------------------------------------------------
    front = spec.worldbody.add_camera()
    front.name = "front"
    front.pos = [1.02, 0.0, 0.78]
    front.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    front.targetbody = "table"
    front.fovy = 52

    hand = spec.body("hand")
    wrist = hand.add_camera()
    wrist.name = "wrist"
    wrist.pos = [0.0, -0.055, 0.0]
    # 相机沿自己的 -z 看；手掌 +z 指向两指，绕 x 转 180° 让相机顺着手指看出去
    wrist.quat = [0.0, 1.0, 0.0, 0.0]
    wrist.fovy = 70

    # TCP：两指中心，抓取的目标点就是它
    tcp = hand.add_site()
    tcp.name = "tcp"
    tcp.pos = [0.0, 0.0, TCP_OFFSET]
    tcp.size = [0.005, 0.005, 0.005]
    tcp.rgba = [1.0, 0.0, 1.0, 0.0]

    model = spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, img_hw)
    model.vis.global_.offheight = max(model.vis.global_.offheight, img_hw)

    for item in layout["cubes"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item["name"])
        item["bid"] = bid
        item["qadr"] = model.jnt_qposadr[model.body_jntadr[bid]]
    for item in layout["plates"]:
        item["bid"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item["name"])

    layout["tcp_site"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    layout["hand_bid"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    layout["arm_qadr"] = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
    )
    layout["arm_dofadr"] = np.array(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
    )
    layout["finger_qadr"] = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
         for j in ("finger_joint1", "finger_joint2")]
    )
    return model, layout


if __name__ == "__main__":
    m, lay = build_scene(["红", "绿", "蓝"], ["黄"])
    print("nq", m.nq, "nu", m.nu, "ncam", m.ncam)
    for k in ("cubes", "plates"):
        print(k, [(o["name"], o["color"], o.get("qadr")) for o in lay[k]])
    print("arm_qadr", lay["arm_qadr"], "finger_qadr", lay["finger_qadr"])
