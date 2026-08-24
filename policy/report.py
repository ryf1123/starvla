"""把 runs/ 下所有实验汇总成一张表 + 训练曲线图。

    python -m policy.report                      # 全部
    python -m policy.report --runs runs/lang_*   # 指定
    python -m policy.report --curves             # additionally 画训练/验证曲线
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np
import yaml


def load_run(d):
    cfg = yaml.safe_load(open(f"{d}/config.yaml")) if os.path.exists(f"{d}/config.yaml") else {}
    log = [json.loads(l) for l in open(f"{d}/log.jsonl")] if os.path.exists(f"{d}/log.jsonl") else []
    ev = json.load(open(f"{d}/eval.json")) if os.path.exists(f"{d}/eval.json") else None
    gen = json.load(open(f"{d}/generalize.json")) if os.path.exists(f"{d}/generalize.json") else None
    tr = [r for r in log if "l1" in r and "val_l1" not in r]
    va = [r for r in log if "val_l1" in r]
    m = cfg.get("model", {})
    row = dict(
        name=os.path.basename(d), steps=tr[-1]["step"] if tr else 0,
        lang=m.get("lang_mode"), head=m.get("head", "regress"),
        backbone=m.get("backbone", "cnn"), cams=",".join(m.get("cams", [])),
        H=m.get("horizon"), data=os.path.basename(str(cfg.get("data", {}).get("path", ""))),
        limit=cfg.get("data", {}).get("limit"),
        train_l1=tr[-1]["l1"] if tr else np.nan,
        val_l1=va[-1]["val_l1"] if va else np.nan,
        kp_std=va[-1].get("front_kp_std") if va else None,
        max_p=va[-1].get("front_max_p") if va else None,
        sr=ev["success_rate"] if ev else None,
    )
    if ev:
        from collections import Counter
        c = Counter(r["outcome"] for r in ev["results"])
        row.update({k: c.get(k, 0) for k in ("wrong_cube", "no_grasp", "off_plate", "dropped")})
        row["n_eval"] = len(ev["results"])
    if gen:
        row["gen"] = {g["case"]: g["sr"] for g in gen}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--curves", action="store_true")
    a = ap.parse_args()
    dirs = a.runs or sorted(d for d in glob.glob("runs/*") if os.path.isdir(d))
    rows = [load_run(d) for d in dirs if os.path.exists(f"{d}/config.yaml")]
    if not rows:
        print("runs/ 下还没有实验")
        return

    hdr = f"{'实验':<18}{'步数':>7}{'语言':>6}{'动作头':>10}{'骨干':>10}{'相机':>13}{'H':>3}" \
          f"{'数据':>10}{'val L1':>8}{'kp_std':>8}{'成功率':>8}  失败分解"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        sr = f"{r['sr']:.1%}" if r["sr"] is not None else "—"
        fail = (f"抓错{r.get('wrong_cube',0)} 没抓起{r.get('no_grasp',0)} "
                f"掉了{r.get('dropped',0)} 没进盘{r.get('off_plate',0)}"
                if r["sr"] is not None else "")
        print(f"{r['name']:<18}{r['steps']:>7}{str(r['lang']):>6}{str(r['head']):>10}"
              f"{str(r['backbone']):>10}{r['cams']:>13}{str(r['H']):>3}"
              f"{str(r['limit'] or r['data'])[:10]:>10}{r['val_l1']:>8.4f}"
              f"{(r['kp_std'] if r['kp_std'] is not None else float('nan')):>8.3f}{sr:>8}  {fail}")

    gens = [r for r in rows if r.get("gen")]
    if gens:
        cases = sorted({c for r in gens for c in r["gen"]})   # 不同 run 测的项可能不一样
        print(f"\n泛化测试\n{'实验':<18}" + "".join(f"{c:>12}" for c in cases))
        for r in gens:
            print(f"{r['name']:<18}" + "".join(
                (f"{r['gen'][c]:>11.0%} " if c in r["gen"] else f"{'—':>11} ") for c in cases))

    if a.curves:
        import matplotlib.pyplot as plt
        from scripts._style import save
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for d in dirs:
            if not os.path.exists(f"{d}/log.jsonl"):
                continue
            log = [json.loads(l) for l in open(f"{d}/log.jsonl")]
            tr = [(r["step"], r["l1"]) for r in log if "l1" in r and "val_l1" not in r]
            va = [(r["step"], r["val_l1"]) for r in log if "val_l1" in r]
            kp = [(r["step"], r["front_kp_std"]) for r in log if "front_kp_std" in r]
            n = os.path.basename(d)
            if tr: axes[0].plot(*zip(*tr), lw=1, label=n)
            if va: axes[1].plot(*zip(*va), lw=1.4, marker="o", ms=3, label=n)
            if kp: axes[2].plot(*zip(*kp), lw=1.4, marker="o", ms=3, label=n)
        for ax, t in zip(axes, ["训练 L1", "验证 L1", "关键点跨场景 std（视觉健康度）"]):
            ax.set_title(t, fontsize=10); ax.set_xlabel("步"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        save(fig, "docs/figs/report_curves.png")


if __name__ == "__main__":
    main()
