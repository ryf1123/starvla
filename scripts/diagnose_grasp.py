"""闭环失败诊断：策略在什么时候、离方块多远的地方闭合了夹爪。

成功率只告诉你"失败了"，这个脚本告诉你"错在哪一毫米"。

产出：
    docs/figs/diag_grasp.png   闭合时刻的水平误差分布 + 一局的距离/夹爪时间曲线
    终端表格：每局的闭合时刻、闭合时水平误差、高度差、最终结果

    python -m scripts.diagnose_grasp --run runs/bc_v2 --episodes 20
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts._style import save, C
from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv
from expert.scripted import ScriptedExpert


def trace(policy_or_expert, env, seed, is_expert=False):
    obs = env.reset(seed=seed)
    tc = env.spec.target_cube
    if is_expert:
        agent = ScriptedExpert(env, rng=np.random.default_rng(seed))
    else:
        policy_or_expert.reset(obs["instruction"])
    log = []
    done = False
    while not done:
        a = agent.act() if is_expert else policy_or_expert.act(obs)
        log.append(dict(d_xy=float(np.linalg.norm(env.tcp()[:2] - env.cube_pos(tc)[:2])),
                        dz=float(env.tcp()[2] - env.cube_pos(tc)[2]),
                        grip_cmd=float(a[4]), grip_w=env.grip_width()))
        obs, r, done, info = env.step(a)
    return log, classify(env), env.t


def summarize(logs):
    rows = []
    for log, outcome, T in logs:
        g = np.array([l["grip_cmd"] for l in log])
        d = np.array([l["d_xy"] for l in log])
        z = np.array([l["dz"] for l in log])
        t_close = int(np.argmax(g < 0)) if (g < 0).any() else -1
        rows.append(dict(outcome=outcome, steps=T, t_close=t_close,
                         d_close=d[t_close] if t_close >= 0 else np.nan,
                         z_close=z[t_close] if t_close >= 0 else np.nan,
                         d_min=d.min()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/bc_v2")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, sm, ss = load_policy(a.run, dev)
    env = TabletopEnv(seed=a.seed)
    runner = Runner(model, vocab, sm, ss, dev, k=cfg["eval"]["ensemble_k"])

    pol = [trace(runner, env, a.seed + i) for i in range(a.episodes)]
    exp = [trace(None, env, a.seed + i, is_expert=True) for i in range(a.episodes)]
    rp, re = summarize(pol), summarize(exp)

    print(f"{'局':<4}{'结果':<12}{'步数':<6}{'闭合步':<8}{'闭合时水平误差':<16}{'闭合时高度差':<14}{'最近距离'}")
    for i, r in enumerate(rp):
        print(f"{i:<4}{r['outcome']:<12}{r['steps']:<6}{r['t_close']:<8}"
              f"{r['d_close']*1000:>8.1f} mm      {r['z_close']*1000:>+8.1f} mm    {r['d_min']*1000:>6.1f} mm")
    ok = [r for r in rp if r["outcome"] == "success"]
    print(f"\n成功 {len(ok)}/{len(rp)}；闭合时水平误差 中位数 "
          f"{np.nanmedian([r['d_close'] for r in rp])*1000:.1f} mm"
          f"（专家 {np.nanmedian([r['d_close'] for r in re])*1000:.1f} mm）")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bins = np.linspace(0, 100, 21)
    ax.hist([r["d_close"] * 1000 for r in re], bins=bins, alpha=0.65, label="专家", color=C["act"])
    ax.hist([r["d_close"] * 1000 for r in rp], bins=bins, alpha=0.65, label="策略", color=C["warn"])
    ax.axvline(22, color="k", ls="--", lw=1)
    ax.text(23, ax.get_ylim()[1] * 0.85, "方块半宽 22 mm\n超过这条线基本抓空", fontsize=8)
    ax.set_xlabel("闭合夹爪时的水平误差 (mm)"); ax.set_ylabel("局数")
    ax.legend(fontsize=8); ax.set_title("夹爪闭合时离方块有多远", fontsize=10)

    ax = axes[1]
    log = pol[0][0]
    ax.plot([l["d_xy"] * 1000 for l in log], color=C["front"], label="TCP–方块 水平距离 (mm)")
    ax.plot([l["grip_w"] * 1000 for l in log], color=C["act"], label="夹爪开度 (mm)")
    g = np.array([l["grip_cmd"] for l in log])
    if (g < 0).any():
        ax.axvline(int(np.argmax(g < 0)), color=C["warn"], ls="--", label="发出闭合指令")
    ax.set_xlabel("控制步"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"第 0 局：{pol[0][1]}", fontsize=10)
    save(fig, "docs/figs/diag_grasp.png")


if __name__ == "__main__":
    main()
