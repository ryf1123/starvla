"""策略讲解图：这 1.3 M 个参数里到底发生了什么。

产出（docs/figs/）：
    policy_arch.png       架构图，每条边上标真实张量形状
    policy_keypoints.png  空间 softmax 的 32 个关键点画回原图，看它盯着哪儿
    policy_lang.png       语言编码：有序 Transformer vs 字符词袋的相似度矩阵
    policy_chunk.png      预测的动作分块 vs 专家动作；时间集成前后的动作曲线

    python -m scripts.explain_policy --run runs/bc_v1
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from scripts._style import save, C
from policy.eval import load_policy, Runner
from policy.dataset import load_episodes, DemoDataset, CharVocab
from sim.tabletop_env import TabletopEnv
from expert.scripted import ScriptedExpert


def fig_arch(model, cfg):
    H, A = model.H, model.act_dim
    L = cfg["model"]["lang_mode"]
    boxes = [
        (0.02, 0.72, "front\n(3,128,128) uint8", C["front"]),
        (0.02, 0.44, "wrist\n(3,128,128) uint8", C["wrist"]),
        (0.02, 0.22, "state\n(7,)", C["grey"]),
        (0.02, 0.02, f"指令 tokens\n(20,) int64", C["lang"]),
        (0.28, 0.02, f"字符嵌入 + 位置\n1 层 Transformer\n({L})", C["lang"]),
        (0.52, 0.02, "z_lang\n(128,)", C["lang"]),
        (0.28, 0.72, "4 层 CNN\n→ (128,8,8)", C["front"]),
        (0.28, 0.44, "4 层 CNN\n→ (128,8,8)", C["wrist"]),
        (0.52, 0.72, "空间 softmax\n(256,) = 128 点 × xy", C["front"]),
        (0.52, 0.44, "空间 softmax\n(256,)", C["wrist"]),
        (0.52, 0.22, "MLP\n(64,)", C["grey"]),
        (0.74, 0.40, f"concat (704,)\n→ MLP 512×2", C["act"]),
        (0.90, 0.40, f"动作分块\n({H},{A})", C["act"]),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 0.95); ax.axis("off")
    pos = {}
    for x, y, t, col in boxes:
        w = 0.155 if x < 0.88 else 0.13
        ax.add_patch(FancyBboxPatch((x, y), w, 0.15, boxstyle="round,pad=0.012",
                                    fc="white", ec=col, lw=1.6))
        ax.text(x + w / 2, y + 0.075, t, ha="center", va="center", fontsize=8.5, color=col)
        pos[t] = (x, y, w)

    def arrow(a, b, col, rad=0.0):
        xa, ya, wa = pos[a]; xb, yb, wb = pos[b]
        ax.add_patch(FancyArrowPatch((xa + wa, ya + 0.075), (xb, yb + 0.075),
                                     arrowstyle="->", color=col, lw=1.2,
                                     connectionstyle=f"arc3,rad={rad}", mutation_scale=12))
    arrow("front\n(3,128,128) uint8", "4 层 CNN\n→ (128,8,8)", C["front"])
    arrow("wrist\n(3,128,128) uint8", "4 层 CNN\n→ (128,8,8)", C["wrist"])
    arrow("4 层 CNN\n→ (128,8,8)", "空间 softmax\n(256,) = 128 点 × xy", C["front"])
    arrow("4 层 CNN\n→ (128,8,8)", "空间 softmax\n(256,)", C["wrist"])
    arrow("state\n(7,)", "MLP\n(64,)", C["grey"])
    arrow("指令 tokens\n(20,) int64", f"字符嵌入 + 位置\n1 层 Transformer\n({L})", C["lang"])
    arrow(f"字符嵌入 + 位置\n1 层 Transformer\n({L})", "z_lang\n(128,)", C["lang"])
    for src in ("空间 softmax\n(256,) = 128 点 × xy", "空间 softmax\n(256,)", "MLP\n(64,)", "z_lang\n(128,)"):
        arrow(src, f"concat (704,)\n→ MLP 512×2", C["act"], rad=0.06)
    arrow(f"concat (704,)\n→ MLP 512×2", f"动作分块\n({H},{A})", C["act"])

    # FiLM 的回边
    ax.add_patch(FancyArrowPatch((0.60, 0.17), (0.355, 0.72), arrowstyle="->",
                                 color=C["lang"], lw=1.4, ls="--",
                                 connectionstyle="arc3,rad=-0.35", mutation_scale=12))
    ax.text(0.40, 0.36, "FiLM：z_lang → (γ, β)\n在 CNN 第 3 层调制特征图\n"
                        "「该看哪个颜色」发生在视觉里，\n不是等到最后拼接才补救",
            fontsize=8, color=C["lang"])
    ax.set_title(f"StarVLA 策略：{sum(p.numel() for p in model.parameters())/1e6:.2f} M 参数", fontsize=11)
    save(fig, "docs/figs/policy_arch.png")


def fig_keypoints(model, dev, run, cfg, n=3):
    """空间 softmax 的关键点画回原图。看它在抓取前盯着哪儿。"""
    env = TabletopEnv(seed=42, img_hw=128)
    from policy.eval import load_policy
    _, vocab, _, smean, sstd = load_policy(run, dev)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 6.6))
    for j in range(n):
        obs = env.reset(seed=200 + j)
        ex = ScriptedExpert(env, rng=np.random.default_rng(j))
        for _ in range(6):                       # 走到接近抓取的时刻
            obs, *_ = env.step(ex.act())
        b = {"tokens": torch.from_numpy(vocab.encode(obs["instruction"]))[None].to(dev),
             "state": torch.from_numpy(((obs["state"] - smean) / sstd).astype(np.float32))[None].to(dev)}
        for c in model.cams:
            b[c] = torch.from_numpy(obs[c].transpose(2, 0, 1).copy())[None].to(dev)
        with torch.no_grad():
            _, heats = model(b, return_heat=True)
        for i, cam in enumerate(model.cams):
            # heats 里已经是 softmax 之后的概率（别再 softmax 一次——我第一次就栽在这）
            heat = heats[cam][0].cpu().numpy()          # (C, h, w)
            C, hh, ww = heat.shape
            p = heat.reshape(C, -1)
            gy, gx = np.meshgrid(np.linspace(0, 127, hh), np.linspace(0, 127, ww), indexing="ij")
            xs = (p * gx.reshape(-1)).sum(1); ys = (p * gy.reshape(-1)).sum(1)
            sharp = p.max(1)                            # 每个关键点有多确信
            keep = np.argsort(-sharp)[:20]              # 只画最确信的 20 个，否则糊成一团
            xs, ys, sharp = xs[keep], ys[keep], sharp[keep]
            ax = axes[i, j]
            ax.imshow(obs[cam]); ax.axis("off")
            ax.scatter(xs, ys, s=10 + 260 * sharp, c=sharp, cmap="autumn", alpha=0.8,
                       edgecolors="k", linewidths=0.3)
            # 叠加目标方块的真实像素位置：看关键点有没有真的盯着它
            uv, vis = env.world_to_pixel(cam, env.cube_pos(env.spec.target_cube))
            if vis:
                gx_, gy_ = (uv + 1) / 2 * env.img_hw
                ax.plot(gx_, gy_, "*", ms=20, mfc="#00e5ff", mec="k", mew=0.8)
            if i == 0:
                ax.set_title(f"「{obs['instruction']}」", fontsize=8)
            ax.set_xlabel(cam)
    fig.suptitle("空间 softmax 最确信的 20 个关键点（点越大越确信）；青色★是目标方块的真实位置\n"
                 "每个通道输出一个期望坐标——策略要的是「东西在哪」，这一步把它白送给网络", fontsize=10)
    save(fig, "docs/figs/policy_keypoints.png")


def fig_lang(model, dev, vocab):
    """有序编码 vs 字符词袋：两句颠倒颜色的指令会不会被编码成同一个东西。"""
    texts = ["把红色方块放进黄色盘子", "把黄色方块放进红色盘子",
             "把绿色方块放进蓝色盘子", "把蓝色方块放进绿色盘子"]
    tok = torch.from_numpy(np.stack([vocab.encode(t) for t in texts])).to(dev)
    with torch.no_grad():
        z_seq = model.lang(tok)
        old = model.lang.mode
        model.lang.mode = "bow"
        z_bow = model.lang(tok)
        model.lang.mode = old

    def cos(z):
        z = torch.nn.functional.normalize(z, dim=-1)
        return (z @ z.T).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, z, name in zip(axes, (z_seq, z_bow), ("有序 Transformer（本项目默认）", "字符词袋（消融组）")):
        M = cos(z)
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(texts))); ax.set_yticks(range(len(texts)))
        ax.set_xticklabels([t[1:3] + "→" + t[7:9] for t in texts], fontsize=8)
        ax.set_yticklabels([t[1:3] + "→" + t[7:9] for t in texts], fontsize=8)
        for i in range(len(texts)):
            for j in range(len(texts)):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8,
                        color="k" if abs(M[i, j]) < 0.6 else "w")
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="余弦相似度")
    fig.suptitle("同一个训好的模型，只换语言池化方式：\n"
                 "「红→黄」和「黄→红」的字符集完全相同，词袋把它们编码成同一个向量（相似度 1.000）", fontsize=10)
    save(fig, "docs/figs/policy_lang.png")


def fig_chunk(model, dev, run, cfg):
    """预测的分块 vs 专家动作，以及时间集成前后的动作曲线。"""
    _, vocab, _, smean, sstd = load_policy(run, dev)
    env = TabletopEnv(seed=77, img_hw=128)
    obs = env.reset(seed=77)
    ex = ScriptedExpert(env, rng=np.random.default_rng(0))
    runner = Runner(model, vocab, smean, sstd, dev, k=cfg["eval"]["ensemble_k"])
    runner.reset(obs["instruction"])
    raw, ens, exp, chunks = [], [], [], []
    for t in range(26):
        a_exp = ex.act()
        b = {"tokens": runner.tok,
             "state": torch.from_numpy(((obs["state"] - smean) / sstd).astype(np.float32))[None].to(dev)}
        for c in model.cams:
            b[c] = torch.from_numpy(obs[c].transpose(2, 0, 1).copy())[None].to(dev)
        with torch.no_grad():
            pred = model(b)[0].cpu().numpy()
        a_ens = runner.act(obs)
        raw.append(pred[0]); ens.append(a_ens); exp.append(a_exp)
        if t in (2, 8, 14):
            chunks.append((t, pred.copy()))
        obs, r, done, info = env.step(a_exp)          # 沿专家轨迹走，保证对齐
        if done:
            break
    raw, ens, exp = map(np.array, (raw, ens, exp))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    ax = axes[0]
    ax.plot(exp[:, 2], color="k", lw=2, label="专家 dz")
    ax.plot(raw[:, 2], color=C["warn"], lw=1, alpha=0.8, label="策略预测的第 1 步")
    for t, ch in chunks:
        ax.plot(np.arange(t, t + model.H), ch[:, 2], "-o", ms=3, lw=1.4, alpha=0.9,
                color=C["front"], label="预测的整块（未来 8 步）" if t == chunks[0][0] else None)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel("dz")
    ax.set_title("动作分块：每一步都预测未来 8 步，蓝色的三段是其中三次预测", fontsize=10)

    ax = axes[1]
    ax.plot(exp[:, 2], color="k", lw=2, label="专家 dz")
    ax.plot(raw[:, 2], color=C["warn"], lw=1, alpha=0.7, label="不做集成（只用最新预测）")
    ax.plot(ens[:, 2], color=C["act"], lw=1.6, label=f"时间集成 k={runner.k}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel("dz"); ax.set_xlabel("控制步")
    jr = np.abs(np.diff(raw[:, 2])).mean(); je = np.abs(np.diff(ens[:, 2])).mean()
    ax.set_title(f"时间集成把相邻步的抖动从 {jr:.3f} 降到 {je:.3f}（平均 |Δ动作|）", fontsize=10)
    save(fig, "docs/figs/policy_chunk.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bc_v1")
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, smean, sstd = load_policy(a.run, dev)
    fig_arch(model, cfg)
    fig_lang(model, dev, vocab)
    fig_keypoints(model, dev, a.run, cfg)
    fig_chunk(model, dev, a.run, cfg)
