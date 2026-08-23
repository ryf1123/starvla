"""消融套件：一次只改一个变量，顺序跑训练 + 闭环评测，最后汇总成表。

    python -m policy.ablate --suite lang           # 语言消融四组
    python -m policy.ablate --suite lang --steps 6000 --episodes 40

结果写到 runs/<name>/eval.json，汇总表打印并存到 runs/ablate_<suite>.json。
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
import numpy as np

SUITES = {
    # name              overrides                      一句话：这一组在问什么
    "lang": [
        ("lang_cls", ["model.lang_mode=cls"], "CLS 读出 + 放大位置编码（修复版，主基线）"),
        ("lang_seq", ["model.lang_mode=seq"], "均值池化的 Transformer（实测会退化成词袋）"),
        ("lang_bow", ["model.lang_mode=bow"], "字符词袋：颜色顺序被抹平"),
        ("lang_none", ["model.lang_mode=none"], "完全拿不到指令：只能瞎猜抓哪个"),
        ("lang_cls_nofilm", ["model.lang_mode=cls", "model.film=false"],
         "有语言但不做 FiLM：语言只在最后拼接"),
    ],
    "cams": [
        ("cam_both", ["model.cams=[front, wrist]"], "两路相机（默认）"),
        ("cam_front", ["model.cams=[front]"], "只有前视：最后 2 cm 看不清"),
        ("cam_wrist", ["model.cams=[wrist]"], "只有腕视：看不到全局，不知道盘子在哪"),
    ],
    "chunk": [
        ("chunk1", ["model.horizon=1"], "不分块：逐步预测，复合误差最大"),
        ("chunk4", ["model.horizon=4"], ""),
        ("chunk8", ["model.horizon=8"], "默认"),
        ("chunk16", ["model.horizon=16"], "块太长：后半段和当前观测关系变弱"),
    ],
}


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--config", default="configs/bc_place.yaml")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    rows = []
    for name, overrides, why in SUITES[args.suite]:
        t0 = time.time()
        if not args.skip_train:
            run([sys.executable, "-u", "-m", "policy.train", "--config", args.config,
                 "--name", name, "--steps", str(args.steps), "--set", *overrides])
        from policy.eval import evaluate
        sr, res = evaluate(f"runs/{name}", episodes=args.episodes, seed=1000,
                           video=f"videos/{name}.mp4")
        json.dump(dict(success_rate=sr, results=res), open(f"runs/{name}/eval.json", "w"),
                  ensure_ascii=False, indent=2)
        from collections import Counter
        cnt = Counter(r["outcome"] for r in res)
        rows.append(dict(name=name, why=why, sr=sr, minutes=(time.time() - t0) / 60,
                         **{k: cnt.get(k, 0) for k in
                            ("success", "wrong_cube", "no_grasp", "off_plate")}))
        print(f"== {name}: 成功率 {sr:.1%}  {dict(cnt)}", flush=True)

    json.dump(rows, open(f"runs/ablate_{args.suite}.json", "w"), ensure_ascii=False, indent=2)
    w = max(len(r["name"]) for r in rows)
    print(f"\n{'实验'.ljust(w)}  成功率  抓错  没抓起  没进盘  说明")
    for r in rows:
        print(f"{r['name'].ljust(w)}  {r['sr']:5.1%}  {r['wrong_cube']:4d}  "
              f"{r['no_grasp']:6d}  {r['off_plate']:6d}  {r['why']}")


if __name__ == "__main__":
    main()
