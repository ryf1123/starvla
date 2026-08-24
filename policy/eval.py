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
    from policy.train import model_kwargs, state_dim_of
    model = VLAPolicy(vocab=len(vocab), state_dim=state_dim_of(cfg), **model_kwargs(cfg))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    smean, sstd = ck["state_stats"]
    return model, vocab, cfg, np.asarray(smean), np.asarray(sstd)


class Runner:
    """维护动作队列 + 时间集成。"""

    def __init__(self, model, vocab, smean, sstd, device, k=4, m=0.3, state_history=1,
                 stride=1):
        """stride：每隔多少步重新规划一次。

        动作分块预测了未来 H 步，但我们默认每一步都重新规划（stride=1），
        只用最新预测的第 0 项——等于把"分块"只当成训练时的正则。
        ACT 原版是执行完一整块再重规划（stride=H）。中间还有 stride=2/4。
        这是**纯推理侧的开关，不用重训**，和时间集成一样应该先扫一遍再说。
        """
        self.model, self.vocab, self.dev = model, vocab, device
        self.smean, self.sstd, self.k, self.m = smean, sstd, k, m
        self.H = model.H
        self.K = max(1, int(state_history))
        self.stride = max(1, int(stride))

    def reset(self, instruction):
        self.hist_s, self.hist_a = [], []          # 多步历史观测的滚动缓冲
        self.pending = []                          # stride>1 时，块里还没执行完的动作
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
        sn = (obs["state"] - self.smean) / self.sstd
        self.hist_s.append(sn.astype(np.float32))
        self.hist_s = self.hist_s[-self.K:]
        while len(self.hist_s) < self.K:           # 开局不足 K 帧就用最早那帧补
            self.hist_s.insert(0, self.hist_s[0])
        if self.K == 1:
            sv = self.hist_s[-1]
        else:
            pa = list(self.hist_a[-self.K:])
            while len(pa) < self.K:
                pa.insert(0, np.zeros(5, np.float32))
            sv = np.concatenate([np.concatenate(self.hist_s), np.concatenate(pa)]).astype(np.float32)
        b = {"tokens": self.tok,
             **({"lang_feat": self.lang_feat, "lang_mask": self.lang_mask}
                if self.model.lang.mode in ("ppool", "ptok") else {}),
             "state": torch.from_numpy(sv)[None].to(self.dev)}
        for c in self.model.cams:
            b[c] = torch.from_numpy(obs[c].transpose(2, 0, 1).copy())[None].to(self.dev)
        if self.pending:                               # 块里还有没执行完的动作，直接用
            a = self.pending.pop(0)
            self.hist_a.append(a.astype(np.float32)); self.hist_a = self.hist_a[-8:]
            return a
        pred = self.model(b)[0].cpu().numpy()          # (H,5)
        if self.stride > 1:
            self.pending = [np.concatenate([p[:4], [1.0 if p[4] > 0 else -1.0]])
                            for p in pred[1:self.stride]]
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
        self.hist_a.append(a.astype(np.float32))
        self.hist_a = self.hist_a[-8:]
        return a


def classify(env, moved_thresh=0.03):
    """失败归因：光看成功率不知道错在哪。

    wrong_cube  动了别的方块、目标方块没动 → 语言没接上
    no_grasp    目标方块基本没动 → 完全没碰到
    pushed      方块动了但**从没被抬起来**（最大高度 < 5 cm）→ 夹爪没对准，把方块推走了
    dropped     抬起来过，但最后停在离盘子很远的地方 → 搬运途中掉了
    off_plate   抬起来过、停在盘子附近但没进去 → **放置精度**不够（真正差一点点的那种）

    这套分类改过两次，两次都是因为**合并的类别掩盖了不同的病因**：
      第一版只有 off_plate 一类 → 以为是放置精度不够，动手改了专家的松手高度；
      第二版按"离盘子远不远"拆出 dropped → 以为是搬运途中掉了；
      量了"方块被抬到的最大高度"才发现：那些局方块最大只离桌面 29 mm（成功局是 148 mm），
      **根本没被抬起来过**，是被推着走的。真正的分水岭是**闭合夹爪时的对准误差**
      （成功局中位数 11.7 mm，失败局 45.6 mm，方块半宽 22 mm）。见 notes/12。
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
    from sim.assets import PLATE_R
    if getattr(env, "max_cube_h", 0.0) < 0.05:          # 从没抬起来 → 是推走的，不是搬掉的
        return "pushed"
    # any_plate 下"目标盘子"没有意义，用**最近的**盘子判定远近
    idxs = range(len(s.plate_colors)) if getattr(s, "any_plate", False) else [s.target_plate]
    d = min(np.linalg.norm(env.cube_pos(s.target_cube)[:2] - env.plate_pos(i)[:2]) for i in idxs)
    return "off_plate" if d < 2 * PLATE_R else "dropped"


def evaluate(run, episodes=50, seed=1000, k=None, video=None, env_kwargs=None,
             instruction_fn=None, device=None, stride=None, decode=None, decode_temp=1.0,
             fixed_noise=False, infer_steps=None):
    """k=None 时用该 run 自己 config 里的 eval.ensemble_k，别写死默认值——
    我一开始把默认值写成 4，配置改成 1 之后所有评测还在偷偷用 4。"""
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab, cfg, smean, sstd = load_policy(run, device)
    if decode is not None:
        model.decode, model.decode_temp = decode, decode_temp
    model.diff_fixed_noise = bool(fixed_noise)
    if infer_steps is not None:
        model.diff_infer_steps = int(infer_steps)
    # 评测环境要和训练时的任务分布一致：训练用了同色方块（"左边的红色方块"）时，
    # 评测也必须开着，否则是在另一个分布上测。
    kw = dict(same_color_prob=cfg["eval"].get("same_color_prob", 0.0),
              img_hw=cfg["eval"].get("img_hw", 128))     # 训练用多大分辨率，评测就得渲染多大
    if cfg["eval"].get("dr", 0) > 0:
        from sim.randomize import DomainRandomizer
        kw["dr"] = DomainRandomizer(level=cfg["eval"]["dr"])
    kw.update(env_kwargs or {})
    env = TabletopEnv(seed=seed, **kw)
    k = cfg["eval"].get("ensemble_k", 1) if k is None else k
    stride = cfg["eval"].get("stride", 1) if stride is None else stride
    runner = Runner(model, vocab, smean, sstd, device, k=k, stride=stride,
                    state_history=cfg["model"].get("state_history", 1))
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
                            cube_to_plate=float(np.linalg.norm(
                                env.cube_pos(env.spec.target_cube)[:2]
                                - env.plate_pos(env.spec.target_plate)[:2])),
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
    ap.add_argument("--ensemble", type=int, default=None,
                    help="时间集成窗口；不传就用该 run 的 config 里的 eval.ensemble_k")
    ap.add_argument("--stride", type=int, default=None, help="每隔多少步重新规划")
    ap.add_argument("--decode", default=None, choices=["argmax", "expect"],
                    help="离散头的解码方式：argmax 或对 softmax 求期望（纯推理侧开关）")
    ap.add_argument("--decode-temp", type=float, default=1.0)
    ap.add_argument("--fixed-noise", action="store_true",
                    help="扩散头：每步从同一份初始噪声出发（纯推理侧开关）")
    ap.add_argument("--infer-steps", type=int, default=None, help="扩散头推理的 DDIM 步数")
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None, help="把逐 episode 结果写成 json")
    args = ap.parse_args()
    sr, res = evaluate(args.run, args.episodes, args.seed, args.ensemble, args.video,
                       stride=args.stride, decode=args.decode, decode_temp=args.decode_temp,
                       fixed_noise=args.fixed_noise, infer_steps=args.infer_steps)
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
