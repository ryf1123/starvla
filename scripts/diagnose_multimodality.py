"""量数据的条件变异——任何模型的误差下限。

方法借自姊妹项目 seamless-interaction 的「第一环」：
**一个模型不可能比数据本身的条件随机性做得更准。**
所以在比较"确定性回归 vs 生成式"之前，先把这个下限摆出来。

做法：正常跑若干局，在沿途的状态上**冻结环境**，用 K 个不同的随机种子问专家 K 次，
得到"同一个观测下专家动作的真实分布"，然后算两条下限：

    均值式模型的下限   E|a_k − mean_k a|      回归能达到的最好水平
    采样式模型的下限   E|a_k − a_k'|          采样式模型（扩散）能达到的最好水平

（两条不一样：回归的最优解是条件均值，采样式模型的最优解是从条件分布里采一个。
  分布越多峰，第二条越大——这也是为什么不能拿同一个 L1 去比这两类模型。）

    python -m scripts.diagnose_multimodality --episodes 12 --k 16
"""
from __future__ import annotations

import argparse, copy
import numpy as np

from sim.tabletop_env import TabletopEnv
from expert.scripted import ScriptedExpert


def conditional_actions(env, expert, k, noise, rng, resample_target=False):
    """在当前状态上，问专家 k 次（每次不同的噪声种子），返回 (k,5)。

    resample_target=True 时**连"这局要放进哪个盘子"也一起重采**——
    在 any_plate 任务里这个选择是观测里看不到的隐变量，
    条件分布（给定图像和指令）本来就包含它；不重采等于偷看了答案。
    """
    phase, hold = expert.phase, expert.hold
    tp0 = env.spec.target_plate
    out = []
    for i in range(k):
        expert.phase, expert.hold = phase, hold
        expert.rng = np.random.default_rng(int(rng.integers(1 << 30)))
        expert.noise = noise
        if resample_target:
            env.spec.target_plate = int(rng.integers(len(env.spec.plate_colors)))
        out.append(np.asarray(expert.act(), dtype=np.float64).copy())
    expert.phase, expert.hold = phase, hold
    env.spec.target_plate = tp0
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--same-color-prob", type=float, default=0.0)
    ap.add_argument("--any-plate", action="store_true", help="用「放进任意盘子」的多峰任务")
    a = ap.parse_args()

    env = TabletopEnv(seed=0, any_plate=a.any_plate)
    rng = np.random.default_rng(0)
    per_phase = {}
    mean_lb, samp_lb = [], []
    for ep in range(a.episodes):
        obs = env.reset(seed=1000 + ep)
        ex = ScriptedExpert(env, rng=np.random.default_rng(ep), noise=a.noise)
        done = False
        while not done:
            A = conditional_actions(env, ex, a.k, a.noise, rng, resample_target=a.any_plate)
            m = A.mean(0)
            d_mean = np.abs(A - m).mean()                      # 到条件均值的平均距离
            d_pair = np.abs(A[:, None] - A[None]).mean()       # 两个独立样本之间
            mean_lb.append(d_mean); samp_lb.append(d_pair)
            per_phase.setdefault(ex.phase, []).append((d_mean, d_pair))
            ex.rng = np.random.default_rng(ep)                 # 恢复正常推进
            ex.noise = a.noise
            obs, r, done, info = env.step(ex.act())

    names = {0: "0 悬停对准", 1: "1 下降", 2: "2 闭合夹爪", 3: "3 抬起",
             4: "4 平移到盘子上方", 5: "5 下降放置", 6: "6 松开撤离"}
    print(f"每个状态问专家 {a.k} 次（只换噪声种子），{a.episodes} 局，动作归一化到 [-1,1]\n")
    print(f"{'阶段':<18}{'样本数':>7}{'均值式模型的下限':>20}{'采样式模型的下限':>20}")
    for p in sorted(per_phase):
        v = np.array(per_phase[p])
        print(f"{names.get(p, p):<18}{len(v):>7}{v[:,0].mean():>20.4f}{v[:,1].mean():>20.4f}")
    print(f"{'全部':<18}{len(mean_lb):>7}{np.mean(mean_lb):>20.4f}{np.mean(samp_lb):>20.4f}")
    print(f"\n对照：训练好的模型在验证集上的 L1")
    print(f"  回归头 0.0843   离散头 0.0868   扩散头（DDIM 采样后）0.1396")


if __name__ == "__main__":
    main()
