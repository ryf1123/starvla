"""一张图讲清整个项目：从仿真到闭环评测的全链路。

    python -m scripts.explain_overview   →  docs/figs/overview.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from scripts._style import save, C


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="white", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top",
            fontsize=10.5, color=color)
    for i, t in enumerate(lines):
        ax.text(x + 0.016, y + h - 0.095 - i * 0.044, t, ha="left", va="top", fontsize=8.2)


def arrow(ax, p, q, color, text=None, rad=0.0, dy=0.02):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", color=color, lw=1.6,
                                 connectionstyle=f"arc3,rad={rad}", mutation_scale=14))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center",
                fontsize=8.2, color=color)


def main():
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.01, 0.60, 0.30, 0.37, "① 世界（sim/）", [
        "MuJoCo + Franka Panda（7 轴 + 夹爪）",
        "3 个彩色方块 + 2 个盘子",
        "盘子颜色从方块颜色里取 → 语序有信息",
        "前视 128×128 + 腕视 128×128",
        "物理 500 Hz，控制 20 Hz（decimation 25）",
        "动作 [dx,dy,dz,dyaw,夹爪] → 阻尼最小二乘 IK",
    ], C["front"])

    box(ax, 0.35, 0.60, 0.30, 0.37, "② 专家与数据（expert/）", [
        "7 段状态机，用特权信息，成功率 100%",
        "动作加噪声 σ=0.05 → 数据从线摊成带",
        "60% 的局中途插入随机扰动（不记录）",
        "再记录专家怎么纠正回来 → 恢复数据",
        "专家抓空会退回第 0 段重抓",
        "800 局 / 27 255 步 / 149 MB",
    ], C["act"])

    box(ax, 0.69, 0.60, 0.30, 0.37, "③ 策略（policy/）", [
        "视觉：4 层 CNN → 16×16 → 空间 softmax",
        "  （logits 必须用原始卷积输出）",
        "语言：字符 [CLS] Transformer / 冻结 bge",
        "条件化：FiLM 在视觉中间层调制",
        "动作头：回归分块 / 离散 token / 扩散",
        "1.33 M 参数，MPS 上 40 分钟训完",
    ], C["lang"])

    box(ax, 0.18, 0.20, 0.30, 0.30, "④ 闭环评测（policy/eval.py）", [
        "策略放回环境，自己承担自己的偏移",
        "时间集成 k=4（抑制抖动）",
        "失败归因：抓错方块 / 没抓起来 / 没进盘子",
        "bc_v3：50 局 74%，抓错方块 0 局",
        "平均 66.7 步（专家 33 步，多的是重试）",
    ], C["warn"])

    box(ax, 0.53, 0.20, 0.30, 0.30, "⑤ 消融与泛化", [
        "一次只改一个：视觉 / 语言 / 相机 /",
        "分块长度 / 动作头 / 骨干 / 数据量",
        "泛化：干扰物、角落、换措辞、角色对调",
        "每组 ≥ 40 局闭环——只信成功率，不信 loss",
    ], C["grey"])

    arrow(ax, (0.31, 0.78), (0.35, 0.78), C["grey"], "环境接口")
    arrow(ax, (0.65, 0.78), (0.69, 0.78), C["grey"], "演示数据")
    arrow(ax, (0.80, 0.60), (0.44, 0.50), C["grey"], "训好的策略", rad=0.12)
    arrow(ax, (0.48, 0.35), (0.53, 0.35), C["grey"], "结果")
    arrow(ax, (0.18, 0.35), (0.06, 0.60), C["warn"], "", rad=0.3)
    ax.text(0.015, 0.50, "失败归因\n→ 回去改\n世界/数据/模型", fontsize=8.5, color=C["warn"])

    ax.text(0.5, 0.11, "关键纪律：训练 loss 不是指标，闭环成功率才是；"
                       "怀疑哪个模块就把它的输出直接打印出来；数据里没有的能力模型不会有",
            ha="center", fontsize=10, color=C["warn"])
    ax.text(0.5, 0.045, "github.com/ryf1123/starvla     Mac mini M4 / 16 GB / 无 CUDA",
            ha="center", fontsize=9, color=C["grey"])
    ax.set_title("StarVLA：在一台 Mac mini 上从零跑通一个闭环 VLA", fontsize=13)
    save(fig, "docs/figs/overview.png")


if __name__ == "__main__":
    main()
