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
    "pretrained_lang": "预训练语言编码器：字符级 vs 冻结句向量 vs 冻结 token 序列",
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
                         **{k: c.get(k, 0) for k in ("success", "wrong_cube", "no_grasp", "off_plate", "dropped")}))
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
    out = ["| 实验 | 成功率 | 95% 区间 | 抓错方块 | 没抓起来 | 半路掉了 | 没进盘子 | 在问什么 |",
           "|---|---|---|---|---|---|---|---|"]
    best = max((r["sr"] for r in rows), default=0)
    for r in sorted(rows, key=lambda r: -r["sr"]):
        star = " ★" if r["sr"] >= best - 1e-9 else ""
        n = sum(r.get(k, 0) for k in ("success", "wrong_cube", "no_grasp", "off_plate", "dropped"))
        lo, hi = wilson(r["success"], n)
        out.append(f"| `{r['name']}`{star} | **{r['sr']:.0%}** | {lo:.0%}–{hi:.0%} | "
                   f"{r.get('wrong_cube',0)} | {r.get('no_grasp',0)} | {r.get('dropped',0)} | "
                   f"{r.get('off_plate',0)} | {r['why']} |")
    out.append("")
    out.append(f"（n={n} 局。95% Wilson 区间——差距小于区间宽度的结论不要当真。）")
    return "\n".join(out)


def fig_all():
    """所有已完成套件的成功率总图：每个套件一组柱子，基线用虚线标出。"""
    import matplotlib.pyplot as plt
    from scripts._style import save, C
    suites = [(k, suite_rows(k)[0]) for k in
              ("vision", "lang", "pretrained_lang", "cams", "chunk", "heads", "backbone", "aux", "data")]
    suites = [(k, r) for k, r in suites if len(r) >= 2]
    if not suites:
        return
    base = None
    if os.path.exists("runs/bc_v3/eval.json"):
        base = json.load(open("runs/bc_v3/eval.json"))["success_rate"]
    n = len(suites)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, rows) in zip(axes, suites):
        rows = sorted(rows, key=lambda r: -r["sr"])
        x = np.arange(len(rows))
        cols = [C["act"] if r["name"] == "bc_v3" else C["front"] for r in rows]
        ax.bar(x, [r["sr"] * 100 for r in rows], color=cols, alpha=0.9)
        for xi, r in zip(x, rows):
            nn = sum(r.get(k, 0) for k in ("success", "wrong_cube", "no_grasp", "off_plate", "dropped"))
            lo, hi = wilson(r["success"], nn)
            ax.plot([xi, xi], [lo * 100, hi * 100], color="k", lw=1.2)
            ax.text(xi, r["sr"] * 100 + 2, f"{r['sr']:.0%}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["name"].replace("lang_", "").replace("vis_", "").replace("cam_", "")
                            for r in rows], fontsize=8, rotation=30, ha="right")
        if base:
            ax.axhline(base * 100, color=C["grey"], ls="--", lw=1)
        ax.set_title(SUITE_TITLE.get(k, k).split("：")[0], fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("闭环成功率 (%)"); axes[0].set_ylim(0, 105)
    fig.suptitle("消融总览（每组 40 局闭环，误差棒为 95% Wilson 区间，虚线为基线 bc_v3）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "docs/figs/ablation_all.png")


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

    for suite in ("vision", "lang", "pretrained_lang", "cams", "chunk", "heads",
                  "backbone", "aux", "data"):
        rows, done = suite_rows(suite)
        if len(rows) < 2:            # 只有基线一行的套件还没开跑，不占版面
            continue
        L += [f"## {SUITE_TITLE.get(suite, suite)}" + ("" if done else "（进行中）"), "",
              md_table(rows), ""]

    if os.path.exists("runs/ablate_ensemble2.json"):
        rows = json.load(open("runs/ablate_ensemble2.json"))
        if os.path.exists("runs/ablate_ensemble.json"):
            rows += json.load(open("runs/ablate_ensemble.json"))
        rows = sorted(rows, key=lambda r: (r["k"], -r["n"]))
        L += ["## 时间集成（纯推理侧开关，不用重训）", "",
              "| 集成窗口 k | 局数 | 成功率 | 平均步数 |", "|---|---|---|---|"]
        seen = set()
        for r in rows:
            if (r["k"], r["n"]) in seen:
                continue
            seen.add((r["k"], r["n"]))
            L.append(f"| {r['k']} | {r['n']} | **{r['sr']:.0%}** | {r['steps']:.0f} |")
        L += ["", "k 越大越差，且平均步数越长。见 `notes/07`。", ""]

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

    fig_all()
    txt = "\n".join(L)
    open("docs/SUMMARY.md", "w").write(txt)
    print(txt)
    print("→ docs/SUMMARY.md")


if __name__ == "__main__":
    main()
