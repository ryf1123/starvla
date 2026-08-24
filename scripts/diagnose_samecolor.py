"""两个同色方块时，策略是不是瞄准了它们的中点？

空间 softmax 输出的是每个通道的**期望坐标**。如果 FiLM 把"红色"这个条件
同时点亮了两个红色方块，期望坐标就会落在两者中间——那里什么都没有。
这是这个架构的一个结构性局限，只有把指令空间扩大到需要"左边的/右边的"之后才暴露。

    python -m scripts.diagnose_samecolor --run runs/rel_cls_hist --episodes 60
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rel_cls_hist")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, sm, ss = load_policy(a.run, dev)
    env = TabletopEnv(seed=a.seed, any_plate=cfg["eval"].get("any_plate", False),
                      same_color_prob=cfg["eval"].get("same_color_prob", 0.0),
                      img_hw=cfg["eval"].get("img_hw", 128))
    runner = Runner(model, vocab, sm, ss, dev, k=cfg["eval"].get("ensemble_k", 1),
                    stride=cfg["eval"].get("stride", 1),
                    state_history=cfg["model"].get("state_history", 1))
    dup_rows, uni_rows = [], []
    for ep in range(a.episodes):
        obs = env.reset(seed=a.seed + ep)
        s = env.spec
        tc = s.target_cube
        same = [i for i, c in enumerate(s.cube_colors) if c == s.cube_colors[tc] and i != tc]
        runner.reset(obs["instruction"])
        log, done = [], False
        while not done:
            act = runner.act(obs)
            tcp = env.tcp()[:2]
            row = dict(grip=float(act[4]),
                       d_tgt=float(np.linalg.norm(tcp - env.cube_pos(tc)[:2])))
            if same:
                o = same[0]
                mid = (env.cube_pos(tc)[:2] + env.cube_pos(o)[:2]) / 2
                row["d_other"] = float(np.linalg.norm(tcp - env.cube_pos(o)[:2]))
                row["d_mid"] = float(np.linalg.norm(tcp - mid))
            log.append(row)
            obs, r, done, info = env.step(act)
        g = np.array([l["grip"] for l in log])
        t = int(np.argmax(g < 0)) if (g < 0).any() else len(log) - 1
        rec = dict(outcome=classify(env), d_tgt=log[t]["d_tgt"],
                   d_min_tgt=min(l["d_tgt"] for l in log))
        if same:
            rec.update(d_other=log[t]["d_other"], d_mid=log[t]["d_mid"],
                       d_min_mid=min(l["d_mid"] for l in log))
            dup_rows.append(rec)
        else:
            uni_rows.append(rec)

    print(f"{a.run}，{a.episodes} 局；数字都是**发出闭合指令那一刻**的水平距离（mm）\n")
    print(f"{'':<24}{'局数':>5}{'成功率':>8}{'离目标方块':>12}{'离同色的另一个':>15}{'离两者中点':>12}")
    if dup_rows:
        r = dup_rows
        print(f"{'两个同色方块':<24}{len(r):>5}"
              f"{np.mean([x['outcome']=='success' for x in r]):>8.1%}"
              f"{np.median([x['d_tgt'] for x in r])*1000:>12.0f}"
              f"{np.median([x['d_other'] for x in r])*1000:>15.0f}"
              f"{np.median([x['d_mid'] for x in r])*1000:>12.0f}")
        closer_mid = np.mean([x["d_mid"] < x["d_tgt"] for x in r])
        print(f"\n  闭合时**离中点比离目标更近**的比例：{closer_mid:.1%}")
        print(f"  全程最接近目标 {np.median([x['d_min_tgt'] for x in r])*1000:.0f} mm，"
              f"最接近中点 {np.median([x['d_min_mid'] for x in r])*1000:.0f} mm")
    if uni_rows:
        r = uni_rows
        print(f"\n{'颜色唯一（对照）':<24}{len(r):>5}"
              f"{np.mean([x['outcome']=='success' for x in r]):>8.1%}"
              f"{np.median([x['d_tgt'] for x in r])*1000:>12.0f}")
        print(f"  全程最接近目标 {np.median([x['d_min_tgt'] for x in r])*1000:.0f} mm")


if __name__ == "__main__":
    main()
