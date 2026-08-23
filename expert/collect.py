"""采集专家演示 → data/demos/<name>/shard_*.npz

每条 episode 存：两路图像(uint8)、状态、动作、指令、TaskSpec、是否成功。
只有成功的 episode 会被写进数据集（失败的另存一份用于分析，见 --keep-fail）。

用法：
    python -m expert.collect --name place400 --episodes 400 --workers 8 --noise 0.05
"""
from __future__ import annotations

import argparse, json, os, time
import multiprocessing as mp
import numpy as np


def _worker(args):
    wid, n_ep, seed0, cfg = args
    from sim.tabletop_env import TabletopEnv
    from expert.scripted import ScriptedExpert
    env = TabletopEnv(n_cubes=cfg["n_cubes"], n_plates=cfg["n_plates"],
                      img_hw=cfg["img_hw"], task_type=cfg["task_type"], seed=seed0)
    rng = np.random.default_rng(seed0)
    eps, n_fail = [], 0
    for k in range(n_ep):
        obs = env.reset(seed=seed0 + k)
        ex = ScriptedExpert(env, rng=rng, noise=cfg["noise"])
        F, W, S, A = [], [], [], []
        done = False
        while not done:
            a = ex.act()
            F.append(obs["front"]); W.append(obs["wrist"])
            S.append(obs["state"]); A.append(a.astype(np.float32))
            obs, r, done, info = env.step(a)
        if not info["success"]:
            n_fail += 1
            if not cfg["keep_fail"]:
                continue
        eps.append(dict(front=np.array(F, np.uint8), wrist=np.array(W, np.uint8),
                        state=np.array(S, np.float32), action=np.array(A, np.float32),
                        instruction=env.spec.instruction, success=bool(info["success"]),
                        spec=json.dumps(env.spec.to_dict(), ensure_ascii=False)))
    return wid, eps, n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--img-hw", type=int, default=128)
    ap.add_argument("--n-cubes", type=int, default=3)
    ap.add_argument("--n-plates", type=int, default=1)
    ap.add_argument("--task-type", default="place")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-fail", action="store_true")
    args = ap.parse_args()

    cfg = dict(noise=args.noise, img_hw=args.img_hw, n_cubes=args.n_cubes,
               n_plates=args.n_plates, task_type=args.task_type, keep_fail=args.keep_fail)
    out = f"data/demos/{args.name}"
    os.makedirs(out, exist_ok=True)
    per = [args.episodes // args.workers + (i < args.episodes % args.workers)
           for i in range(args.workers)]
    jobs = [(i, per[i], args.seed + 10000 * i, cfg) for i in range(args.workers) if per[i]]

    t0 = time.time()
    with mp.Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)

    total, fails, steps = 0, 0, 0
    for wid, eps, nf in results:
        fails += nf
        if not eps:
            continue
        np.savez_compressed(
            f"{out}/shard_{wid:02d}.npz",
            **{f"{k}_{i}": e[k] for i, e in enumerate(eps) for k in ("front", "wrist", "state", "action")},
            meta=json.dumps([{ "instruction": e["instruction"], "success": e["success"],
                               "spec": e["spec"], "T": len(e["action"]) } for e in eps],
                            ensure_ascii=False))
        total += len(eps)
        steps += sum(len(e["action"]) for e in eps)

    json.dump(dict(cfg=cfg, episodes=total, steps=steps, failed=fails, seed=args.seed),
              open(f"{out}/info.json", "w"), ensure_ascii=False, indent=2)
    dt = time.time() - t0
    print(f"{total} episodes / {steps} steps -> {out}  失败 {fails}  用时 {dt:.0f}s "
          f"({steps/dt:.0f} step/s)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
