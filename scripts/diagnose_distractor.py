"""同色干扰物在场时，视觉特征里还剩多少"目标在哪"的信息？

第一版量的是空间 softmax 分布的**熵**和"目标附近的概率质量"，结果两组都等于均匀分布
（熵 5.543 vs ln(256)=5.545，质量 0.050 vs 随机 0.049）——**这个探针没用**。
原因见 notes/02：本项目最好的配置里空间 softmax 本来就接近均匀，
信息不在"分布尖不尖"，而在**期望坐标**上那点微小的偏移。

所以改用**线性探针**：从 256 维关键点向量线性回归目标方块的像素坐标，
比较"两个同色方块"和"颜色唯一"两组的留出误差。误差大 = 特征里关于目标位置的信息少。

notes/20 剩下的机制假设之一。做法不用训练，只做前向：

对每一帧，取前视相机的空间 softmax 分布（每个通道一张 H×W 的概率图），量三件事：
    熵          分布有多散。变钝 → 熵升高。
    目标附近的概率质量   有多少概率落在目标方块的像素位置附近（半径 2 格）。
    干扰物附近的概率质量  同色的另一个方块附近有多少。

对照组是同一个模型在"颜色唯一"场景下的同样三个数。

    python -m scripts.diagnose_distractor --run runs/rel_cls_hist --episodes 40
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from policy.eval import load_policy, Runner
from sim.tabletop_env import TabletopEnv


def collect_feats(model, runner, env, episodes, seed, steps, dev):
    """跑若干局，收集 (关键点向量, 目标像素坐标, 是否有同色干扰物)。"""
    K, U, D = [], [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        s = env.spec
        tc = s.target_cube
        dup = any(c == s.cube_colors[tc] for i, c in enumerate(s.cube_colors) if i != tc)
        runner.reset(obs["instruction"])
        for t in range(steps):
            uv, vis = env.world_to_pixel("front", env.cube_pos(tc))
            if vis:
                b = {c: torch.from_numpy(obs[c].transpose(2, 0, 1).copy())[None].to(dev)
                     for c in model.cams}
                with torch.no_grad():
                    z = model.lang(runner.tok)
                    kp, _ = model.enc["front"](b["front"], z)
                K.append(kp[0].cpu().numpy()); U.append(uv); D.append(dup)
            obs, _, done, _ = env.step(runner.act(obs))
            if done:
                break
    return np.array(K), np.array(U), np.array(D)


def probe(K, U, lam=1.0, folds=4):
    """岭回归线性探针，K 折留出，返回像素误差（归一化坐标）。"""
    n = len(K)
    idx = np.arange(n)
    errs = []
    for f in range(folds):
        te = idx % folds == f
        tr = ~te
        if te.sum() < 3 or tr.sum() < 10:
            continue
        X = np.c_[K[tr], np.ones(tr.sum())]
        W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ U[tr])
        P = np.c_[K[te], np.ones(te.sum())] @ W
        errs.append(np.abs(P - U[te]).mean())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rel_cls_hist")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=6, help="每局取前几步（抓取前的定位阶段）")
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, sm, ss = load_policy(a.run, dev)
    env = TabletopEnv(seed=a.seed, same_color_prob=cfg["eval"].get("same_color_prob", 0.0),
                      img_hw=cfg["eval"].get("img_hw", 128))
    runner = Runner(model, vocab, sm, ss, dev, k=cfg["eval"].get("ensemble_k", 1),
                    stride=cfg["eval"].get("stride", 1),
                    state_history=cfg["model"].get("state_history", 1))
    K, U, D = collect_feats(model, runner, env, a.episodes, a.seed, a.steps, dev)
    print(f"{a.run}；前视相机关键点 → 目标方块像素坐标 的线性探针")
    print(f"（坐标归一化到 [-1,1]，误差 0.1 ≈ 128 像素图上的 6.4 个像素）\n")
    print(f"{'':<24}{'帧数':>6}{'留出误差':>12}{'瞎猜（预测均值）':>18}")
    for nm, m in (("两个同色方块", D), ("颜色唯一（对照）", ~D)):
        if m.sum() < 20:
            continue
        base = float(np.abs(U[m] - U[m].mean(0)).mean())
        print(f"{nm:<24}{int(m.sum()):>6}{probe(K[m], U[m]):>12.4f}{base:>18.4f}")
    print(f"{'两组合起来':<24}{len(K):>6}{probe(K, U):>12.4f}"
          f"{float(np.abs(U - U.mean(0)).mean()):>18.4f}")


if __name__ == "__main__":
    main()
