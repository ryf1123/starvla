"""泛化边界测试：训练分布之外，策略什么时候碎。

每一项只改一个变量，其余和训练分布一致，跑 N 局闭环，报成功率和失败归因。

    python -m policy.generalize --run runs/lang_cls --episodes 40

七项：
    base        训练分布（对照组）
    distract5   干扰方块从 3 个加到 5 个（模型没见过这么挤的桌面）
    corner      物体只出现在工作区四角（训练里也有，但很少）
    paraphrase  换指令措辞：「把红色方块放进蓝色盘子」→「拿起红色的方块，放到蓝色盘子里」
    terse       更短的措辞：「红方块到蓝盘子」
    swap_role   角色对调：同一组颜色，指令里方块和盘子的颜色互换（考语序）
    plate3      盘子从 2 个加到 3 个

注意 paraphrase / terse 这两项的天花板：词表是**从训练指令的字符里建的**，
「拿」「起」「到」「里」这些字在词表里不存在，会被映射成 padding。
所以这两项测的其实是「把已知字符打乱/删掉之后还认不认得」，
而不是真正的语义泛化——真正的语义泛化要等第五环换成预训练的语言模型骨干。
这个天花板本身就是「为什么需要 VLM 骨干」的实证。
"""
from __future__ import annotations

import argparse, json
import numpy as np
import torch

from policy.eval import load_policy, Runner, classify
from sim.tabletop_env import TabletopEnv
from sim.tasks import sample_task, TaskSpec
from sim.assets import WORKSPACE


def paraphrase(spec, style):
    c = spec.cube_colors[spec.target_cube]
    p = spec.plate_colors[spec.target_plate]
    if style == "paraphrase":
        return f"拿起{c}色的方块，放到{p}色盘子里"
    if style == "terse":
        return f"{c}方块到{p}盘子"
    return spec.instruction


def make_env(kind, seed, same_color_prob=0.0):
    kw = dict(seed=seed, same_color_prob=same_color_prob)
    if kind == "distract5":
        return TabletopEnv(n_cubes=5, n_plates=2, **kw)
    if kind == "plate3":
        return TabletopEnv(n_cubes=3, n_plates=3, **kw)
    return TabletopEnv(n_cubes=3, n_plates=2, **kw)


def push_to_corner(spec, rng):
    """把所有物体推到工作区四角（训练分布里很少出现的极端布局）。"""
    xs, ys = WORKSPACE["x"], WORKSPACE["y"]
    corners = np.array([[xs[0] + .02, ys[0] + .02], [xs[0] + .02, ys[1] - .02],
                        [xs[1] - .02, ys[0] + .02], [xs[1] - .02, ys[1] - .02]])
    idx = rng.permutation(4)
    for i in range(len(spec.cube_xy)):
        spec.cube_xy[i] = corners[idx[i % 4]] + rng.normal(0, 0.012, 2)
    return spec


def run_case(kind, run, episodes, seed0, device, video=None):
    model, vocab, cfg, smean, sstd = load_policy(run, device)
    env = make_env(kind, seed0, cfg["eval"].get("same_color_prob", 0.0))
    runner = Runner(model, vocab, smean, sstd, device, k=cfg["eval"]["ensemble_k"])
    rng = np.random.default_rng(seed0)
    out, frames = [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep)
        spec = env.spec
        if kind == "corner":
            obs = env.reset(spec=push_to_corner(spec, rng))
        instr = spec.instruction
        if kind in ("paraphrase", "terse"):
            instr = paraphrase(spec, kind)
        if kind == "swap_role":
            # 同一场景，把指令里的两个颜色对调；目标随之改成"另一种读法"
            c, p = spec.cube_colors[spec.target_cube], spec.plate_colors[spec.target_plate]
            if p not in spec.cube_colors or c not in spec.plate_colors:
                continue                      # 这一局没法对调，跳过
            spec.target_cube = spec.cube_colors.index(p)
            spec.target_plate = spec.plate_colors.index(c)
            instr = f"把{p}色方块放进{c}色盘子"
            obs = env.reset(spec=spec)
        runner.reset(instr)
        done = False
        while not done:
            if video and ep < 4:
                frames.append(obs["front"])
            obs, r, done, info = env.step(runner.act(obs))
        out.append(dict(success=bool(info["success"]), outcome=classify(env), instruction=instr))
    if video and frames:
        import imageio
        imageio.mimsave(video, frames, fps=20, quality=7)
    return out


CASES = ["base", "distract5", "corner", "paraphrase", "terse", "swap_role", "plate3"]
DESC = {"base": "训练分布（对照）", "distract5": "5 个方块（训练是 3 个）",
        "corner": "物体全在工作区四角", "paraphrase": "换措辞：拿起…放到…里",
        "terse": "更短的措辞：红方块到蓝盘子", "swap_role": "指令里方块/盘子颜色对调",
        "plate3": "3 个盘子（训练是 2 个）"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/lang_cls")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--cases", nargs="*", default=CASES)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    from collections import Counter
    rows = []
    for kind in a.cases:
        res = run_case(kind, a.run, a.episodes, a.seed, dev,
                       video=f"videos/gen_{kind}.mp4")
        cnt = Counter(r["outcome"] for r in res)
        sr = float(np.mean([r["success"] for r in res])) if res else float("nan")
        rows.append(dict(case=kind, desc=DESC[kind], n=len(res), sr=sr,
                         **{k: cnt.get(k, 0) for k in ("success", "wrong_cube", "no_grasp", "off_plate")}))
        print(f"{kind:<11} {sr:6.1%}  n={len(res):<3} {dict(cnt)}", flush=True)
    out = a.out or f"{a.run}/generalize.json"
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=2)
    print("→", out)


if __name__ == "__main__":
    main()
