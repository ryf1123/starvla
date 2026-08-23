"""把整条链路跑一遍，打印每个张量的形状、含义和真实数值。

对照着读 sim/tabletop_env.py、policy/dataset.py、policy/model.py。
有训好的策略就传 --run，没有就只走到"数据长什么样"。

    python -m scripts.walkthrough --run runs/bc_v1
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

from sim.tabletop_env import TabletopEnv, ACT_POS, ACT_YAW, CTRL_HZ
from expert.scripted import ScriptedExpert


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    hr("1. 环境：模型规格")
    env = TabletopEnv(seed=args.seed)
    m = env.model
    print(f"  nq={m.nq}（7 手臂 + 2 手指 + 3 方块 × 7 自由关节）  nv={m.nv}  nu={m.nu}（7 位置执行器 + 1 夹爪腱）")
    print(f"  物理步长 {m.opt.timestep} s（{1/m.opt.timestep:.0f} Hz），控制 {CTRL_HZ} Hz，decimation={env.decimation}")
    print(f"  相机 {[m.camera(i).name for i in range(m.ncam)]}，图像 {env.img_hw}×{env.img_hw}")

    hr("2. 一局任务：TaskSpec 完全决定这一局")
    obs = env.reset(seed=args.seed)
    s = env.spec
    print(f"  指令        「{s.instruction}」")
    print(f"  方块颜色    {s.cube_colors}    目标方块 cube{s.target_cube}")
    print(f"  盘子颜色    {s.plate_colors}    目标盘子 plate{s.target_plate}")
    for i, c in enumerate(s.cube_colors):
        print(f"    cube{i} {c}  xy={np.round(s.cube_xy[i],3).tolist()}  yaw={s.cube_yaw[i]:+.3f} rad")
    print(f"    plate     xy={np.round(s.plate_xy[s.target_plate],3).tolist()}")

    hr("3. 观测：策略每一步看到什么")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:<12}{str(v.shape):<18}{v.dtype}  范围 [{v.min()}, {v.max()}]")
        else:
            print(f"  {k:<12}{'str':<18}「{v}」")
    names = ["tcp_x", "tcp_y", "tcp_z", "sin yaw", "cos yaw", "夹爪开度", "上一步夹爪指令"]
    print("  state 逐项：")
    for n, v in zip(names, obs["state"]):
        print(f"    {n:<16}{v:+.4f}")

    hr("4. 一个控制步：动作 → 关节指令 → 物理")
    ex = ScriptedExpert(env, rng=np.random.default_rng(0))
    a = ex.act()
    ee_before, tcp_before = env.ee_pos.copy(), env.tcp().copy()
    obs, r, done, info = env.step(a)
    print(f"  专家动作 a          [{', '.join(f'{x:+.3f}' for x in a)}]  （阶段 {ex.phase}）")
    print(f"  位移增量            a[0:3] × {ACT_POS} = {np.round(a[:3]*ACT_POS, 4).tolist()} m")
    print(f"  偏航增量            a[3] × {ACT_YAW} = {a[3]*ACT_YAW:+.4f} rad")
    print(f"  末端目标            {np.round(ee_before,4).tolist()} → {np.round(env.ee_pos,4).tolist()}")
    print(f"  IK 残差             {info['ik_pos_err']*1000:.2f} mm")
    print(f"  关节位置指令 ctrl   {np.round(env.data.ctrl[:7], 3).tolist()}")
    print(f"  夹爪 ctrl[7]        {env.data.ctrl[7]:.0f}（255 张开 / 0 闭合）")
    print(f"  {env.decimation} 个物理子步后 TCP  {np.round(tcp_before,4).tolist()} → {np.round(env.tcp(),4).tolist()}")
    print(f"  实际位移 {np.linalg.norm(env.tcp()-tcp_before)*1000:.1f} mm vs "
          f"指令位移 {np.linalg.norm(env.ee_pos-ee_before)*1000:.1f} mm（跟踪滞后）")

    hr("5. 走完一局")
    while not done:
        obs, r, done, info = env.step(ex.act())
    print(f"  用了 {env.t} 步（{env.t/CTRL_HZ:.1f} 秒），成功={info['success']}，超时={info['timeout']}")

    if not args.run:
        print("\n（没传 --run，跳过策略部分）")
        return

    hr("6. 数据集样本：动作分块")
    import yaml
    from policy.dataset import DemoDataset, CharVocab, load_episodes
    cfg = yaml.safe_load(open(f"{args.run}/config.yaml"))
    eps = load_episodes(cfg["data"]["path"])[:4]
    vocab = CharVocab([e["instruction"] for e in eps])
    ds = DemoDataset(eps, vocab, cfg["model"]["horizon"], train=False)
    sample = ds[10]
    print(f"  数据集 {len(eps)} 局 → {len(ds)} 个样本；词表 {len(vocab)} 个字符 {vocab.itos}")
    for k, v in sample.items():
        print(f"  {k:<10}{str(tuple(v.shape)):<14}{v.dtype}")
    txt = eps[ds.index[10][0]]["instruction"]
    print(f"  指令「{txt}」→ tokens {sample['tokens'][:len(txt)].tolist()} + padding")
    print(f"  action 分块（未来 {cfg['model']['horizon']} 步）:")
    for i, row in enumerate(sample["action"]):
        print(f"    t+{i}  [{', '.join(f'{x:+.2f}' for x in row)}]  mask={sample['mask'][i]:.0f}")

    hr("7. 策略前向：每一层的形状")
    from policy.eval import load_policy, Runner
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab2, cfg2, smean, sstd = load_policy(args.run, dev)
    b = {k: v[None].to(dev) for k, v in sample.items() if k in ("front", "wrist", "state", "tokens")}
    with torch.no_grad():
        z = model.lang(b["tokens"])
        print(f"  语言编码器（{cfg2['model']['lang_mode']}）  tokens {tuple(b['tokens'].shape)} → z_lang {tuple(z.shape)}")
        for cam in model.cams:
            kp, heat = model.enc[cam](b[cam], z)
            print(f"  {cam:<6}图像 {tuple(b[cam].shape)} → 特征图 {tuple(heat.shape)} "
                  f"→ 空间 softmax 关键点 {tuple(kp.shape)}（{kp.shape[1]//2} 个点的 x,y）")
        act = model(b)
        print(f"  拼接 → MLP → 动作分块 {tuple(act.shape)}")
        print(f"  预测第一步 [{', '.join(f'{x:+.3f}' for x in act[0,0].cpu())}]")
        print(f"  数据里第一步 [{', '.join(f'{x:+.3f}' for x in sample['action'][0])}]")

    hr("8. 闭环推理：时间集成")
    runner = Runner(model, vocab2, smean, sstd, dev, k=cfg2["eval"]["ensemble_k"])
    obs = env.reset(seed=args.seed + 100)
    runner.reset(obs["instruction"])
    print(f"  指令「{obs['instruction']}」，集成窗口 k={runner.k}")
    for t in range(4):
        a = runner.act(obs)
        print(f"  第 {t} 步：缓冲区里 {len(runner.buf)} 条预测 → 融合动作 "
              f"[{', '.join(f'{x:+.3f}' for x in a)}]")
        obs, r, done, info = env.step(a)


if __name__ == "__main__":
    main()
