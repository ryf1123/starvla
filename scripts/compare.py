"""配对比较两个策略：同一批初始状态、同一批种子，然后做 McNemar 检验。

为什么要配对 + 检验（依据见 docs/lit-review.md）：
    PhAIL（2026）调查了 13 篇真机 VLA 论文，每个条件的众数 N 只有 10–20，
    **没有一篇报告置信区间或配对检验**。他们算出想要 ±5 个百分点的 95% 区间，
    单臂需要约 380 次 rollout。
    我们在 MuJoCo 里 rollout 几乎免费——这是相对整个领域的结构性优势，不用白不用。

McNemar 检验只看**不一致的那些局**：A 成功而 B 失败的有 b 局，反过来有 c 局。
零假设是 b 和 c 同分布。这比独立比较两个成功率强得多，因为它消掉了"这一局本身好不好做"
这个最大的方差来源。

    python -m scripts.compare --a runs/bc_v4 --b runs/vis_relu16 --episodes 200
"""
from __future__ import annotations

import argparse, json
import numpy as np
from math import comb

from policy.eval import evaluate


def mcnemar_exact(b, c):
    """精确二项检验（双侧）。b = A 成功 B 失败的局数，c = 反过来。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--k", type=int, default=None, help="时间集成窗口，默认用各自 config 里的")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {}
    for name in (a.a, a.b):
        kw = dict(episodes=a.episodes, seed=a.seed)
        if a.k is not None:
            kw["k"] = a.k
        sr, r = evaluate(name, **kw)
        res[name] = [x["success"] for x in r]
        lo, hi = wilson(sum(res[name]), len(res[name]))
        print(f"{name:<28} {sr:6.1%}  95% 区间 {lo:.1%}–{hi:.1%}  (n={len(res[name])})", flush=True)

    A, B = np.array(res[a.a]), np.array(res[a.b])
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    both = int((A & B).sum()); neither = int((~A & ~B).sum())
    only_a = int((A & ~B).sum()); only_b = int((~A & B).sum())
    p = mcnemar_exact(only_a, only_b)
    print(f"\n配对表（{n} 局）：都成功 {both}  都失败 {neither}  "
          f"只有 A 成功 {only_a}  只有 B 成功 {only_b}")
    print(f"McNemar 精确检验 p = {p:.4f}  →  "
          + ("**差异显著**（p < 0.05）" if p < 0.05 else "看不出显著差异（不能说谁更好）"))
    print(f"差值 {A.mean()-B.mean():+.1%}（配对），不一致的局共 {only_a+only_b} 个")
    if a.out:
        json.dump(dict(a=a.a, b=a.b, n=n, sr_a=float(A.mean()), sr_b=float(B.mean()),
                       both=both, neither=neither, only_a=only_a, only_b=only_b, p=p,
                       per_episode_a=A.astype(int).tolist(),      # 存逐局结果，方便画配对图
                       per_episode_b=B.astype(int).tolist()),
                  open(a.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
