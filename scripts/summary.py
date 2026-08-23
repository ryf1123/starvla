"""生成一页总结：所有实验的结果表 + 关键结论 + 一张汇总图。

    python -m scripts.summary                 # 打印并写 docs/SUMMARY.md
    python -m scripts.summary --feishu        # 顺便推到飞书父页（追加一节）

设计意图：跑了一夜之后，早上应该有**一页**能看完的东西，而不是一堆 json。
"""
from __future__ import annotations

import argparse, glob, json, os, subprocess, datetime
import numpy as np

SUITE_TITLE = {
    "vision": "视觉分支：空间 softmax 怎么接、特征图多大",
    "lang": "语言：到底有没有起作用",
    "cams": "相机：腕部相机值多少",
    "chunk": "动作分块：预测未来几步",
    "heads": "动作表示：回归 / 离散 token / 扩散",
    "backbone": "视觉骨干：从零 CNN vs 预训练 ResNet18",
    "aux": "辅助监督：能不能替代更多数据",
    "data": "数据量曲线",
}


def suite_rows(suite):
    f = f"runs/ablate_{suite}.json"
    if os.path.exists(f):
        return json.load(open(f)), True
    # 套件没跑完：从各个 run 的 eval.json 拼一个部分结果
    from policy.ablate import SUITES
    rows = []
    for name, ov, why in SUITES.get(suite, []):
        p = f"runs/{name}/eval.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        from collections import Counter
        c = Counter(r["outcome"] for r in d["results"])
        rows.append(dict(name=name, why=why, sr=d["success_rate"],
                         **{k: c.get(k, 0) for k in ("success", "wrong_cube", "no_grasp", "off_plate")}))
    return rows, False


def wilson(k, n, z=1.96):
    """Wilson 区间：n=40 时 60% 的 95% 区间约 ±15 个百分点。

    加这个是为了**不把噪声当结论**——两组差 10 个百分点，在 40 局下基本说明不了问题。
    要下"A 比 B 好"的结论，要么差距远大于区间，要么把局数加上去。
    """
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def md_table(rows):
    out = ["| 实验 | 成功率 | 95% 区间 | 抓错方块 | 没抓起来 | 没进盘子 | 在问什么 |",
           "|---|---|---|---|---|---|---|"]
    best = max((r["sr"] for r in rows), default=0)
    for r in sorted(rows, key=lambda r: -r["sr"]):
        star = " ★" if r["sr"] >= best - 1e-9 else ""
        n = r["success"] + r["wrong_cube"] + r["no_grasp"] + r["off_plate"]
        lo, hi = wilson(r["success"], n)
        out.append(f"| `{r['name']}`{star} | **{r['sr']:.0%}** | {lo:.0%}–{hi:.0%} | {r['wrong_cube']} | "
                   f"{r['no_grasp']} | {r['off_plate']} | {r['why']} |")
    out.append("")
    out.append(f"（n={n} 局。95% Wilson 区间——差距小于区间宽度的结论不要当真。）")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feishu", action="store_true")
    a = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# StarVLA 实验总结（{ts}）", ""]

    if os.path.exists("runs/bc_v3/eval.json"):
        d = json.load(open("runs/bc_v3/eval.json"))
        from collections import Counter
        c = Counter(r["outcome"] for r in d["results"])
        L += ["## 基线", "",
              f"`bc_v3`：1.33 M 参数，800 条演示，16000 步训练（约 40 分钟），"
              f"50 局闭环 **{d['success_rate']:.0%}**"
              f"（抓错方块 {c.get('wrong_cube',0)} / 没抓起来 {c.get('no_grasp',0)} / "
              f"没进盘子 {c.get('off_plate',0)}）。", ""]

    for suite in ("vision", "lang", "cams", "chunk", "heads", "backbone", "aux", "data"):
        rows, done = suite_rows(suite)
        if not rows:
            continue
        L += [f"## {SUITE_TITLE.get(suite, suite)}" + ("" if done else "（进行中）"), "",
              md_table(rows), ""]

    gens = sorted(glob.glob("runs/*/generalize.json"))
    if gens:
        L += ["## 泛化边界", ""]
        for g in gens:
            rows = json.load(open(g))
            L += [f"`{os.path.basename(os.path.dirname(g))}`：", "",
                  "| 测试项 | 成功率 | 说明 |", "|---|---|---|"]
            for r in rows:
                L.append(f"| {r['case']} | {r['sr']:.0%} | {r['desc']} |")
            L.append("")

    L += ["## 代码与文档", "",
          "- 代码：https://github.com/ryf1123/starvla",
          "- 踩坑笔记：`notes/`；概念速查：`docs/concepts.md`；讲解图：`docs/figs/`", ""]

    txt = "\n".join(L)
    open("docs/SUMMARY.md", "w").write(txt)
    print(txt)
    print("→ docs/SUMMARY.md")


if __name__ == "__main__":
    main()
