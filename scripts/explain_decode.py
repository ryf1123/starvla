"""同一份权重、同一批种子，只换解码方式——多峰任务上的同屏对比。

    python -m scripts.explain_decode --run runs/any_discrete --episodes 4

argmax 会**承诺**去某一个盘子；对 softmax 求期望会把两个合法模式**平均**掉，
方块落在两个盘子中间。这是 notes/18 那个"同一个开关，符号相反"的可视化版本。
产出 videos/decode_cmp.mp4 / .gif
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv
from scripts.video_grid import grid


def rollout(run, decode, episodes, seed, dev):
    model, vocab, cfg, sm, ss = load_policy(run, dev)
    model.decode = decode
    env = TabletopEnv(seed=seed, any_plate=cfg["eval"].get("any_plate", False),
                      same_color_prob=cfg["eval"].get("same_color_prob", 0.0),
                      img_hw=cfg["eval"].get("img_hw", 128))
    runner = Runner(model, vocab, sm, ss, dev, k=cfg["eval"].get("ensemble_k", 1),
                    stride=cfg["eval"].get("stride", 1),
                    state_history=cfg["model"].get("state_history", 1))
    frames, outs = [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        runner.reset(obs["instruction"])
        done = False
        while not done:
            frames.append(np.concatenate([obs["front"], obs["wrist"]], axis=1))
            obs, r, done, info = env.step(runner.act(obs))
        frames += [frames[-1]] * 8          # 结尾停一下
        outs.append(classify(env))
    return frames, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/any_discrete")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seqs, labels = [], []
    for dec, cn in (("argmax", "argmax：承诺一个盘子"),
                    ("expect", "期望解码：平均两个盘子")):
        f, o = rollout(a.run, dec, a.episodes, a.seed, dev)
        seqs.append(f)
        labels.append(f"{cn}  {o.count('success')}/{len(o)} 成功")
        print(f"{dec:<8}{o}")
    grid(seqs, labels, "videos/decode_cmp.mp4", fps=14, cols=1)


if __name__ == "__main__":
    main()
