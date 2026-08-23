"""交互式窗口看场景 / 看专家跑 / 看策略跑。macOS 上必须用 mjpython。

    mjpython scripts/view.py                      # 静态看场景，可拖拽
    mjpython scripts/view.py --expert             # 脚本专家实时跑
    mjpython scripts/view.py --run runs/bc_v2     # 策略实时跑
关窗口即退出。
"""
from __future__ import annotations

import argparse, time
import numpy as np
import mujoco
import mujoco.viewer

from sim.tabletop_env import TabletopEnv, CTRL_HZ
from expert.scripted import ScriptedExpert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env = TabletopEnv(seed=a.seed, img_hw=64)
    obs = env.reset(seed=a.seed)
    agent = None
    if a.run:
        import torch
        from policy.eval import load_policy, Runner
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model, vocab, cfg, sm, ss = load_policy(a.run, dev)
        agent = Runner(model, vocab, sm, ss, dev, k=cfg["eval"]["ensemble_k"])
        agent.reset(obs["instruction"])
    elif a.expert:
        agent = ScriptedExpert(env, rng=np.random.default_rng(a.seed))

    print(f"指令：「{obs['instruction']}」")
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        while v.is_running():
            t0 = time.time()
            if agent is not None:
                act = agent.act() if a.expert else agent.act(obs)
                obs, r, done, info = env.step(act)
                if done:
                    print(f"  结束：成功={info['success']}  步数={env.t}")
                    obs = env.reset()
                    if a.expert:
                        agent = ScriptedExpert(env, rng=np.random.default_rng())
                    else:
                        agent.reset(obs["instruction"])
                    print(f"指令：「{obs['instruction']}」")
            else:
                mujoco.mj_forward(env.model, env.data)
            v.sync()
            time.sleep(max(0.0, 1.0 / CTRL_HZ - (time.time() - t0)))


if __name__ == "__main__":
    main()
