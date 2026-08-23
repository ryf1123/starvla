"""语言表示对比：三种编码方式各自把什么信息保住了、把什么信息丢了。

产出 docs/figs/lang_repr.png —— 六条指令两两的余弦相似度矩阵，三种表示并排。

我们要的表示应该满足两件事：
    A. 角色对调的两句（红→黄 / 黄→红）**必须区分**，因为在同一个场景里结果不同；
    B. 同义改写的两句（把红色方块放进黄色盘子 / 拿起红色的方块，放到黄色盘子里）**应该接近**，
       否则换个说法就不认得。

实测结论：字符词袋 A 完全做不到（相似度恰好 1.000）；
冻结的中文句向量 B 做得很好但 A 同样做不到（0.998）；
只有保留 token 序列、让模型自己学着读，才可能两者兼得。
这正是真实 VLA 用 VLM 的 token 序列而不是一个句子 embedding 的原因。

    python -m scripts.explain_lang_repr
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts._style import save, C

TEXTS = [
    ("把红色方块放进黄色盘子", "基准"),
    ("把黄色方块放进红色盘子", "角色对调"),
    ("拿起红色的方块，放到黄色盘子里", "同义改写"),
    ("红方块到黄盘子", "极简说法"),
    ("把红色方块放进蓝色盘子", "换目标盘子"),
    ("把绿色方块放进蓝色盘子", "全换"),
]


def cos(M):
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ M.T


def char_bag(texts):
    vocab = sorted({c for t in texts for c in t})
    M = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        for c in t:
            M[i, vocab.index(c)] += 1
    return M


def main():
    texts = [t for t, _ in TEXTS]
    labels = [f"{i}. {n}" for i, (_, n) in enumerate(TEXTS)]

    from policy.text_encoder import PretrainedText
    pre = PretrainedText()
    feats, mask = pre.encode(texts)
    pool = feats[:, 0].numpy()                                   # CLS
    tokmean = ((feats * (~mask)[..., None]).sum(1) /
               (~mask).sum(1, keepdim=True)).numpy()             # token 平均（不含 padding）

    mats = [("字符词袋", cos(char_bag(texts))),
            ("冻结句向量 bge-small-zh（CLS）", cos(pool)),
            ("冻结 token 特征的平均", cos(tokmean))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (name, M) in zip(axes, mats):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(texts))); ax.set_yticks(range(len(texts)))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right")
        ax.set_yticklabels(labels, fontsize=7.5)
        for i in range(len(texts)):
            for j in range(len(texts)):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7.5,
                        color="w" if M[i, j] > 0.75 else "k")
        ax.set_title(name, fontsize=10)
        # 圈出两个关键格子
        for (i, j), c in ((( 0, 1), "#00e5ff"), ((0, 2), "#00ff88")):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=c, lw=2.5))
    fig.suptitle("六条指令的两两相似度。青框 = 角色对调（必须区分，越低越好），"
                 "绿框 = 同义改写（应该接近，越高越好）\n"
                 f"{TEXTS[0][0]}  vs  {TEXTS[1][0]}  ——  两句话的字符集完全相同", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "docs/figs/lang_repr.png")

    print(f"{'表示':<28}{'角色对调(越低越好)':>20}{'同义改写(越高越好)':>20}")
    for name, M in mats:
        print(f"{name:<28}{M[0,1]:>20.3f}{M[0,2]:>20.3f}")


def fig_vocab(run="runs/bc_v3"):
    """把字符词表实际能认出多少字画出来——换措辞为什么会 0%，一眼就看懂。"""
    import torch
    from policy.eval import load_policy
    _, vocab, _, _, _ = load_policy(run, torch.device("cpu"))
    texts = [t for t, _ in TEXTS]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(texts) + 1.6))
    ax.axis("off")
    ax.set_title(f"训练指令建出来的字符词表只有 {len(vocab)-1} 个字："
                 f"{''.join(vocab.itos[1:])}\n"
                 "灰色 = 词表里没有这个字，会被当成 padding 丢掉", fontsize=10.5)
    for i, (t, name) in enumerate(TEXTS):
        y = 1 - (i + 1) * (1.0 / (len(texts) + 1))
        ax.text(-0.01, y, name, fontsize=9, ha="right", va="center", color=C["grey"])
        kept = 0
        for j, ch in enumerate(t):
            known = ch in vocab.stoi
            kept += known
            ax.text(0.02 + j * 0.045, y, ch, fontsize=13, ha="center", va="center",
                    color="#1a202c" if known else "#cbd5e0",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#ebf8ff" if known else "#f7fafc",
                              ec="#90cdf4" if known else "#e2e8f0"))
        ax.text(0.02 + 16 * 0.045, y, f"认出 {kept}/{len(t)} 字", fontsize=9,
                va="center", color=C["act"] if kept == len(t) else C["warn"])
    ax.set_xlim(-0.22, 0.92); ax.set_ylim(0, 1)
    save(fig, "docs/figs/lang_vocab.png")


if __name__ == "__main__":
    main()
    fig_vocab()
