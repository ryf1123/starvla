"""讲解图的统一样式：中文字体、配色、把标注画到渲染图上的小工具。"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
font_manager.fontManager.addfont(_FONT)
CJK = font_manager.FontProperties(fname=_FONT).get_name()
matplotlib.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 130
matplotlib.rcParams["savefig.bbox"] = "tight"

C = dict(front="#2b6cb0", wrist="#c05621", lang="#6b46c1", act="#2f855a",
         warn="#c53030", grey="#4a5568")


def annotate(ax, xy, text, xytext, color="#c53030", fs=8):
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=fs,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9))


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("→", path)
