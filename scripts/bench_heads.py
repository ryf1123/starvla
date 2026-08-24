"""量三种动作头的单步推理耗时（notes/08 预测 3）。

回归 1 次前向；离散 1 次前向（各维度独立预测，不自回归）；扩散 10 次去噪。
20 Hz 控制环的预算是 50 ms——重点看有没有哪个超预算，而不是谁快几毫秒。

    python -m scripts.bench_heads
"""
from __future__ import annotations

import time
import numpy as np
import torch

from policy.eval import load_policy
from sim.tabletop_env import TabletopEnv

RUNS = {"regress": "runs/bc_v5_hist", "discrete": "runs/head_discrete_h",
        "diffusion": "runs/head_diffusion_h"}


def bench(run, device, n=60, warmup=10):
    import os
    if not os.path.exists(f"{run}/latest.pt"):
        return None
    model, vocab, cfg, sm, ss = load_policy(run, device)
    env = TabletopEnv(seed=0, img_hw=cfg["eval"].get("img_hw", 128))
    obs = env.reset(seed=0)
    b = {c: torch.from_numpy(obs[c].transpose(2, 0, 1).copy())[None].to(device) for c in model.cams}
    b["tokens"] = torch.from_numpy(vocab.encode(obs["instruction"]))[None].to(device)
    sdim = model.state_mlp[0].in_features
    b["state"] = torch.zeros(1, sdim, device=device)
    with torch.no_grad():
        for i in range(warmup + n):
            if i == warmup:
                if device.type == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
            model(b)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / n
    return dt * 1000, model.head_type, sum(p.numel() for p in model.parameters()) / 1e6


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"设备 {dev}；20 Hz 控制环每步预算 50 ms\n")
    print(f"{'动作头':<12}{'实验':<24}{'参数':>8}{'单步推理':>12}{'占预算':>10}")
    for name, run in RUNS.items():
        r = bench(run, dev)
        if r is None:
            print(f"{name:<12}{run:<24}{'—':>8}{'还没训完':>12}")
            continue
        ms, ht, npar = r
        print(f"{name:<12}{run:<24}{npar:>7.2f}M{ms:>10.1f} ms{ms/50:>9.0%}")


if __name__ == "__main__":
    main()
