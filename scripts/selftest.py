"""两分钟自检：把整条链路的每一环都跑一遍最小规模，确认没有被改坏。

    python -m scripts.selftest          # 全部
    python -m scripts.selftest --fast   # 跳过训练和闭环

写这个的原因：这两天改了十几次数据集和模型的签名（多步历史、预训练语言编码器、
域随机化、失败归因分类……），每次都靠手动回归测试，漏掉一次就是浪费一小时训练。
"""
from __future__ import annotations

import argparse, os, shutil, subprocess, sys, tempfile, time
import numpy as np

OK, FAIL = "✓", "✗"
results = []


def check(name, fn):
    t0 = time.time()
    try:
        info = fn() or ""
        results.append((name, True, info, time.time() - t0))
        print(f"{OK} {name:<34} {info}  ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        results.append((name, False, repr(e), time.time() - t0))
        print(f"{FAIL} {name:<34} {type(e).__name__}: {e}", flush=True)


def t_scene():
    from sim.assets import build_scene
    m, lay = build_scene(["红", "绿", "蓝"], ["黄", "蓝"])
    assert m.nq == 30 and m.ncam == 2
    return f"nq={m.nq} 相机={m.ncam}"


def t_ik():
    import mujoco
    from sim.assets import build_scene, HOME_QPOS, TABLE_TOP
    from sim.ik import solve_ik
    m, lay = build_scene(["红"], ["黄"])
    d = mujoco.MjData(m); d.qpos[lay["arm_qadr"]] = HOME_QPOS
    q, ep, er = solve_ik(m, d, lay, np.array([0.45, -0.12, TABLE_TOP + 0.05]), 0.3)
    assert ep < 0.005, f"IK 残差 {ep}"
    return f"残差 {ep*1000:.2f} mm"


def t_env():
    from sim.tabletop_env import TabletopEnv
    env = TabletopEnv(seed=0, img_hw=64)
    o = env.reset(seed=0)
    assert o["front"].shape == (64, 64, 3) and o["state"].shape == (7,)
    o, r, done, info = env.step(np.zeros(5))
    return f"decim={env.decimation} 指令「{o['instruction']}」"


def t_dr():
    from sim.tabletop_env import TabletopEnv
    from sim.randomize import DomainRandomizer
    env = TabletopEnv(seed=0, img_hw=64, dr=DomainRandomizer(level=1.0))
    a = env.reset(seed=0)["front"].astype(float)
    b = env.reset(seed=1)["front"].astype(float)
    assert abs(a.mean() - b.mean()) > 1.0, "域随机化好像没生效"
    return f"两局亮度差 {abs(a.mean()-b.mean()):.1f}"


def t_expert():
    from sim.tabletop_env import TabletopEnv
    from expert.scripted import ScriptedExpert
    env = TabletopEnv(seed=1, img_hw=64)
    ok = 0
    for ep in range(6):
        env.reset(seed=ep); ex = ScriptedExpert(env, rng=np.random.default_rng(ep))
        done = False
        while not done:
            _, _, done, info = env.step(ex.act())
        ok += info["success"]
    assert ok >= 5, f"专家只成功 {ok}/6"
    return f"专家 {ok}/6"


def t_model():
    import torch
    from policy.model import VLAPolicy
    for head in ("regress", "discrete", "diffusion"):
        for K, sd in ((1, 7), (3, 36)):
            m = VLAPolicy(vocab=13, head=head, state_dim=sd, ss_raw=False).eval()
            b = dict(front=torch.randint(0, 255, (2, 3, 128, 128), dtype=torch.uint8),
                     wrist=torch.randint(0, 255, (2, 3, 128, 128), dtype=torch.uint8),
                     state=torch.randn(2, sd), tokens=torch.randint(1, 13, (2, 20)),
                     action=torch.rand(2, 8, 5) * 2 - 1, mask=torch.ones(2, 8))
            m.loss(b)
            with torch.no_grad():
                assert m(b).shape == (2, 8, 5)
    return "3 种动作头 × 2 种历史长度"


def t_resolution():
    import torch
    from policy.model import VLAPolicy
    m = VLAPolicy(vocab=13, ss_raw=False).eval()
    for hw in (128, 160):
        b = dict(front=torch.randint(0, 255, (1, 3, hw, hw), dtype=torch.uint8),
                 wrist=torch.randint(0, 255, (1, 3, hw, hw), dtype=torch.uint8),
                 state=torch.randn(1, 7), tokens=torch.randint(1, 13, (1, 20)))
        with torch.no_grad():
            assert m(b).shape == (1, 8, 5)
    return "128 和 160 都能吃"


def t_collect(tmp):
    subprocess.run([sys.executable, "-m", "expert.collect", "--name", f"_selftest",
                    "--episodes", "4", "--workers", "2", "--img-hw", "64"],
                   check=True, capture_output=True)
    from policy.dataset import load_episodes
    eps = load_episodes("data/demos/_selftest")
    assert len(eps) >= 3 and eps[0]["priv"] is not None
    return f"{len(eps)} 局，含特权信息"


def t_train_eval():
    import yaml, json
    cfg = yaml.safe_load(open("configs/bc_place.yaml"))
    cfg["data"]["path"] = "data/demos/_selftest"
    cfg["eval"]["img_hw"] = 64
    yaml.safe_dump(cfg, open("/tmp/_selftest.yaml", "w"), allow_unicode=True)
    subprocess.run([sys.executable, "-m", "policy.train", "--config", "/tmp/_selftest.yaml",
                    "--name", "_selftest", "--steps", "8",
                    "--set", "train.val_every=8", "train.log_every=100", "model.state_history=3"],
                   check=True, capture_output=True)
    out = subprocess.run([sys.executable, "-m", "policy.eval", "--run", "runs/_selftest",
                          "--episodes", "2"], check=True, capture_output=True, text=True)
    assert "成功率" in out.stdout
    return "训练 8 步 + 闭环 2 局"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    print("StarVLA 自检\n" + "-" * 60)
    check("场景构建", t_scene)
    check("阻尼最小二乘 IK", t_ik)
    check("环境 step/reset", t_env)
    check("域随机化", t_dr)
    check("脚本专家", t_expert)
    check("模型前向/损失", t_model)
    check("输入分辨率自适应", t_resolution)
    if not a.fast:
        check("数据采集", lambda: t_collect(None))
        check("训练 + 闭环评测", t_train_eval)
        shutil.rmtree("data/demos/_selftest", ignore_errors=True)
        shutil.rmtree("runs/_selftest", ignore_errors=True)
    bad = [r for r in results if not r[1]]
    print("-" * 60)
    print(f"{len(results)-len(bad)}/{len(results)} 通过" + ("" if not bad else f"，失败：{[b[0] for b in bad]}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
