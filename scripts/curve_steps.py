"""闭环成功率 vs 训练步数：证明"验证 loss 不是指标"。

需要 runs/<name>/ckpt_*.pt（训练时 train.save_ckpts 存的）。

    python -m scripts.curve_steps --run runs/lang_bow --episodes 30
"""
from __future__ import annotations

import argparse, glob, json, os, re, shutil
import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts._style import save, C
from policy.eval import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    a = ap.parse_args()

    cks = sorted(glob.glob(f"{a.run}/ckpt_*.pt"),
                 key=lambda p: int(re.search(r"ckpt_(\d+)", p).group(1)))
    cks.append(f"{a.run}/latest.pt")
    log = [json.loads(l) for l in open(f"{a.run}/log.jsonl")]
    val = {r["step"]: r["val_l1"] for r in log if "val_l1" in r}
    total = max(val)

    steps, srs, vls = [], [], []
    tmp = f"{a.run}/_tmp"
    os.makedirs(tmp, exist_ok=True)
    for c in cks:
        step = total if c.endswith("latest.pt") else int(re.search(r"ckpt_(\d+)", c).group(1))
        shutil.copy(c, f"{tmp}/latest.pt")
        sr, _ = evaluate(tmp, episodes=a.episodes, seed=1000)
        steps.append(step); srs.append(sr)
        vls.append(min(val.items(), key=lambda kv: abs(kv[0] - step))[1])
        print(f"  {step:>6} 步  成功率 {sr:.1%}  val_l1 {vls[-1]:.4f}", flush=True)
    shutil.rmtree(tmp)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, [s * 100 for s in srs], "-o", color=C["act"], label="闭环成功率 (%)")
    ax.set_xlabel("训练步数"); ax.set_ylabel("闭环成功率 (%)", color=C["act"])
    ax.grid(alpha=0.3); ax.set_ylim(0, 100)
    ax2 = ax.twinx()
    ax2.plot(steps, vls, "-s", color=C["warn"], label="验证 L1")
    ax2.set_ylabel("验证 L1", color=C["warn"])
    ax.set_title(f"{os.path.basename(a.run)}：验证 L1 平缓下降，成功率却是阶跃的\n"
                 f"（每点 {a.episodes} 局闭环，同样的随机种子）", fontsize=10)
    save(fig, f"docs/figs/curve_steps_{os.path.basename(a.run)}.png")
    json.dump(dict(steps=steps, sr=srs, val_l1=vls),
              open(f"{a.run}/curve_steps.json", "w"), indent=2)


if __name__ == "__main__":
    main()
