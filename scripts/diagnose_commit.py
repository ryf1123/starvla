"""多峰任务上的关键问题：策略是**承诺**了一个模式，还是把几个模式**平均**了？

（notes/18 在结果出来之前就把这两种可能分开了，这个脚本就是那两个判据。）

判据一：方块最终落点离"两个盘子连线中点"有多近。
    走中间 → 这个数很小，而且比它到最近盘子的距离还小。
判据二：选中的盘子和几个**确定性线索**的相关性。
    如果策略学的是"总去离方块近的那个"或"总去画面左边那个"，
    那它就是学了个 tie-break 规则，而不是在多峰之间平均——
    这两种行为的失败方式完全不同，不能混为一谈。

    python -m scripts.diagnose_commit --run runs/any_regress --episodes 60
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv
from sim.assets import PLATE_R
from expert.scripted import ScriptedExpert


def rollout(run, episodes, seed, device, expert=False):
    if not expert:
        model, vocab, cfg, sm, ss = load_policy(run, device)
        env = TabletopEnv(seed=seed, any_plate=True,
                          img_hw=cfg["eval"].get("img_hw", 128))
        agent = Runner(model, vocab, sm, ss, device, k=cfg["eval"].get("ensemble_k", 1),
                       stride=cfg["eval"].get("stride", 1),
                       state_history=cfg["model"].get("state_history", 1))
    else:
        env = TabletopEnv(seed=seed, any_plate=True)
    rows = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        s = env.spec
        p0, p1 = np.array(s.plate_xy[0]), np.array(s.plate_xy[1])
        c0 = np.array(s.cube_xy[s.target_cube])
        if expert:
            agent = ScriptedExpert(env, rng=np.random.default_rng(ep))
        else:
            agent.reset(obs["instruction"])
        done = False
        while not done:
            a = agent.act() if expert else agent.act(obs)
            obs, r, done, info = env.step(a)
        c = env.cube_pos(s.target_cube)[:2]
        d = [np.linalg.norm(c - p0), np.linalg.norm(c - p1)]
        mid = (p0 + p1) / 2
        rows.append(dict(
            outcome=classify(env),
            chosen=int(np.argmin(d)), d_near=float(min(d)),
            d_mid=float(np.linalg.norm(c - mid)),
            near_at_start=int(np.argmin([np.linalg.norm(c0 - p0), np.linalg.norm(c0 - p1)])),
            left_in_image=int(np.argmin([p0[1], p1[1]])),   # 世界 −y 在画面左侧
            lifted=float(getattr(env, "max_cube_h", 0.0)),
        ))
    return rows


def stats(rows):
    d_near = np.array([r["d_near"] for r in rows])
    d_mid = np.array([r["d_mid"] for r in rows])
    return dict(n=len(rows),
                sr=float(np.mean([r["outcome"] == "success" for r in rows])),
                committed=float((d_near < PLATE_R).mean()),
                middled=float(((d_mid < d_near) & (d_near > PLATE_R)).mean()),
                d_near_med=float(np.median(d_near)), d_mid_med=float(np.median(d_mid)),
                agree_near=float(np.mean([r["chosen"] == r["near_at_start"] for r in rows])),
                agree_left=float(np.mean([r["chosen"] == r["left_in_image"] for r in rows])))


def report(name, rows):
    n = len(rows)
    d_near = np.array([r["d_near"] for r in rows])
    d_mid = np.array([r["d_mid"] for r in rows])
    committed = (d_near < PLATE_R).mean()
    middled = ((d_mid < d_near) & (d_near > PLATE_R)).mean()
    agree_near = np.mean([r["chosen"] == r["near_at_start"] for r in rows])
    agree_left = np.mean([r["chosen"] == r["left_in_image"] for r in rows])
    sr = np.mean([r["outcome"] == "success" for r in rows])
    print(f"{name:<18}{sr:>7.1%}{committed:>12.1%}{middled:>12.1%}"
          f"{np.median(d_near)*1000:>11.0f}{np.median(d_mid)*1000:>11.0f}"
          f"{agree_near:>12.1%}{agree_left:>12.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["runs/any_regress", "runs/any_discrete",
                                                  "runs/any_diffusion"])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="runs/commit.json")
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    import os
    print("承诺 = 落点进了某个盘子；走中间 = 落点离中点比离最近的盘子还近\n")
    print(f"{'策略':<18}{'成功率':>7}{'承诺':>12}{'走中间':>12}"
          f"{'离最近盘 mm':>11}{'离中点 mm':>11}{'=更近的盘':>12}{'=画面左盘':>12}")
    import json
    out = {}
    rows = rollout(None, a.episodes, a.seed, dev, expert=True)
    report("脚本专家", rows); out["expert"] = stats(rows)
    for r in a.runs:
        if os.path.exists(f"{r}/latest.pt"):
            rows = rollout(r, a.episodes, a.seed, dev)
            report(os.path.basename(r), rows); out[os.path.basename(r)] = stats(rows)
        else:
            print(f"{os.path.basename(r):<18}{'还没训完':>7}")
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
