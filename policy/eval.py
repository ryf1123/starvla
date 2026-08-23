"""闭环评测：把策略放回 MuJoCo 里跑，统计成功率。

这是整个项目的"闭环"二字所在——训练时的 L1 误差低不代表任务能做成，
只有把策略接回环境、让它自己承担自己造成的状态偏移，成功率才有意义。

时间集成（temporal ensembling，ACT 的做法）：
    第 t 步可以用第 t、t-1、…、t-k+1 步预测出的、都指向 t 的那一项动作，
    做指数加权平均。好处是动作连续、抖动小；坏处是对"该果断闭合夹爪"这类
    突变响应变慢，所以 k 不宜大（默认 4）。`--ensemble 1` 就是关掉它。

    python -m policy.eval --run runs/bc_v1 --episodes 50 --video videos/bc_v1.mp4
"""
from __future__ import annotations

import argparse, json
import numpy as np
import torch

from policy.dataset import CharVocab, MAX_LEN
from policy.model import VLAPolicy
from sim.tabletop_env import TabletopEnv
from sim.tasks import sample_task


def load_policy(run, device):
    ck = torch.load(f"{run}/latest.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    vocab = CharVocab.load(ck["vocab"])
    from policy.train import model_kwargs
    model = VLAPolicy(vocab=len(vocab), **model_kwargs(cfg))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    smean, sstd = ck["state_stats"]
    return model, vocab, cfg, np.asarray(smean), np.asarray(sstd)


class Runner:
    """维护动作队列 + 时间集成。"""

    def __init__(self, model, vocab, smean, sstd, device, k=4, m=0.3):
        self.model, self.vocab, self.dev = model, vocab, device
        self.smean, self.sstd, self.k, self.m = smean, sstd, k, m
        self.H = model.H

    def reset(self, instruction):
        self.tok = torch.from_numpy(self.vocab.encode(instruction))[None].to(self.dev)
        if self.model.lang.mode in ("ppool", "ptok"):
            # 现编现用：这样才能测"训练里没出现过的说法"
            if not hasattr(self, "_pre"):
                from policy.text_encoder import PretrainedText
                self._pre = PretrainedText()
            f, m = self._pre.encode([instruction])
            self.lang_feat, self.lang_mask = f.to(self.dev), m.to(self.dev)
        self.buf = []          # 每项 (剩余的预测序列, 生成时的年龄)

    @torch.no_grad()
    def act(self, obs):
        b = {"tokens": self.tok,
             **({"lang_feat": self.lang_feat, "lang_mask": self.lang_mask}
                if self.model.lang.mode in ("ppool", "ptok") else {}),
             "state": torch.from_numpy(((obs["state"] - self.smean) / self.sstd).astype(np.float32))[None].to(self.dev)}
        for c in self.model.cams:
            b[c] = torch.from_numpy(obs[c].transpose(2, 0, 1).copy())[None].to(self.dev)
        pred = self.model(b)[0].cpu().numpy()          # (H,5)
        self.buf.append([pred, 0])
        self.buf = self.buf[-self.k:]
        acts, ws = [], []
        for entry in self.buf:
            p, age = entry
            if age < self.H:
                acts.append(p[age])
                ws.append(np.exp(-self.m * age))
            entry[1] += 1
        w = np.array(ws) / np.sum(ws)
        a = (np.array(acts) * w[:, None]).sum(0)
        a[4] = 1.0 if a[4] > 0 else -1.0               # 夹爪是离散的，集成后再二值化
        return a


def classify(env, moved_thresh=0.03):
    """失败归因：光看成功率不知道错在哪，要把失败分类。

    wrong_cube  动了别的方块 → 语言没接上（抓错东西）
    no_grasp    目标方块基本没动 → 定位或抓取失败
    off_plate   目标方块动了但没进盘子 → 放置精度不够
    """
    s = env.spec
    moved = [np.linalg.norm(env.cube_pos(i)[:2] - s.cube_xy[i]) for i in range(len(s.cube_colors))]
    tgt_moved = moved[s.target_cube] > moved_thresh
    other_moved = any(m > moved_thresh for i, m in enumerate(moved) if i != s.target_cube)
    if env.success():
        return "success"
    if other_moved and not tgt_moved:
        return "wrong_cube"
    if not tgt_moved:
        return "no_grasp"
    return "off_plate"


def evaluate(run, episodes=50, seed=1000, k=4, video=None, env_kwargs=None,
             instruction_fn=None, device=None):
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, smean, sstd = load_policy(run, device)
    env = TabletopEnv(seed=seed, **(env_kwargs or {}))
    runner = Runner(model, vocab, smean, sstd, device, k=k)
    frames, results = [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        instr = instruction_fn(env.spec) if instruction_fn else obs["instruction"]
        runner.reset(instr)
        done = False
        while not done:
            if video and ep < 6:
                frames.append(np.concatenate([obs["front"], obs["wrist"]], axis=1))
            a = runner.act(obs)
            obs, r, done, info = env.step(a)
        results.append(dict(success=bool(info["success"]), steps=env.t, outcome=classify(env),
                            instruction=instr, target=env.spec.target_cube,
                            target_color=env.spec.cube_colors[env.spec.target_cube]))
    sr = float(np.mean([r["success"] for r in results]))
    if video and frames:
        import imageio
        imageio.mimsave(video, frames, fps=20, quality=7)
        imageio.mimsave(video.replace(".mp4", ".gif"), frames[::2], fps=10, loop=0)
    return sr, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None, help="把逐 episode 结果写成 json")
    args = ap.parse_args()
    sr, res = evaluate(args.run, args.episodes, args.seed, args.ensemble, args.video)
    from collections import Counter
    cnt = Counter(r["outcome"] for r in res)
    print(f"成功率 {sr:.1%}  ({sum(r['success'] for r in res)}/{len(res)})  "
          f"平均步数 {np.mean([r['steps'] for r in res]):.1f}")
    print("  失败归因：" + "  ".join(f"{k}={v}" for k, v in cnt.most_common()))
    if args.out:
        json.dump(dict(run=args.run, success_rate=sr, results=res),
                  open(args.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
