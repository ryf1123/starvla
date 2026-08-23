"""消融套件：一次只改一个变量，顺序跑训练 + 闭环评测，最后汇总成表。

    python -m policy.ablate --suite lang           # 语言消融四组
    python -m policy.ablate --suite lang --steps 6000 --episodes 40

结果写到 runs/<name>/eval.json，汇总表打印并存到 runs/ablate_<suite>.json。
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
import numpy as np

# 约定：overrides 为 None 的那一项是**共享基线**（runs/bc_v3，16000 步），不重新训练。
# 这样每个套件只需要训变体，一晚上能多跑好几组。
BASELINE_RUN = "runs/bc_v3"

SUITES = {
    # name              overrides                      一句话：这一组在问什么
    "lang": [
        ("bc_v3", None, "CLS 读出 + 放大位置编码（基线）"),
        ("lang_seq", ["model.lang_mode=seq"], "字符 + 位置编码 + Transformer + 均值池化"),
        ("lang_bow", ["model.lang_mode=bow"], "字符词袋：颜色顺序被抹平"),
        ("lang_none", ["model.lang_mode=none"], "完全拿不到指令：只能瞎猜抓哪个"),
        ("lang_cls_nofilm", ["model.lang_mode=cls", "model.film=false"],
         "有语言但不做 FiLM：语言只在最后拼接"),
    ],
    "pretrained_lang": [
        ("bc_v3", None, "字符级 CLS（基线）"),
        ("lang_ppool", ["model.lang_mode=ppool"], "冻结句向量（bge-small-zh 的 CLS）"),
        ("lang_ptok", ["model.lang_mode=ptok"], "冻结 token 特征 + 可学注意力池化"),
    ],
    "cams": [
        ("bc_v3", None, "两路相机（基线）"),
        ("cam_front", ["model.cams=[front]"], "只有前视：最后 2 cm 看不清"),
        ("cam_wrist", ["model.cams=[wrist]"], "只有腕视：看不到全局，不知道盘子在哪"),
    ],
    "vision": [
        ("bc_v3", None, "原始 logits + 16×16 特征图（基线）"),
        ("vis_relu16", ["model.ss_raw=false", "model.last_stride=1"],
         "softmax 前接 GroupNorm+ReLU（第一版写法）"),
        ("vis_raw8", ["model.ss_raw=true", "model.last_stride=2"],
         "原始 logits，但特征图只有 8×8"),
        ("vis_relu8", ["model.ss_raw=false", "model.last_stride=2"],
         "第一版的完整写法：ReLU + 8×8"),
    ],
    "backbone": [
        ("bc_v3", None, "从零 4 层 CNN（基线）"),
        ("bb_r18", ["model.backbone=resnet18"], "ImageNet 预训练 ResNet18，全部微调"),
        ("bb_r18_frozen", ["model.backbone=resnet18", "model.freeze_backbone=true"],
         "预训练 ResNet18，骨干冻结"),
        ("bb_r18_scratch", ["model.backbone=resnet18", "model.pretrained=false"],
         "ResNet18 结构但随机初始化——区分「结构」和「预训练权重」哪个在起作用"),
    ],
    "aux": [
        ("bc_v3", None, "只用动作监督（基线）"),
        ("aux_1", ["model.aux_weight=1.0"], "加「目标在图像哪个像素」的辅助监督"),
        ("aux_1_small", ["model.aux_weight=1.0", "data.limit=150"], "辅助监督 + 只有 150 条数据"),
        ("aux_0_small", ["model.aux_weight=0.0", "data.limit=150"], "无辅助监督 + 只有 150 条数据"),
    ],
    "heads": [
        ("bc_v3", None, "连续回归 + L1（ACT 路线，基线）"),
        ("head_discrete", ["model.head=discrete"], "每维 41 个格子 + 交叉熵（RT-1 路线）"),
        ("head_diffusion", ["model.head=diffusion"], "10 步 DDIM 去噪（Diffusion Policy 路线）"),
    ],
    "data": [
        ("data_075", ["data.limit=75"], "75 条演示"),
        ("data_150", ["data.limit=150"], "150 条"),
        ("data_300", ["data.limit=300"], "300 条"),
        ("bc_v3", None, "800 条（全部，基线）"),
    ],
    "chunk": [
        ("chunk1", ["model.horizon=1"], "不分块：逐步预测，复合误差最大"),
        ("chunk4", ["model.horizon=4"], ""),
        ("bc_v3", None, "基线"),
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
        if overrides is None:
            name = BASELINE_RUN.split("/")[-1]      # 共享基线，不训练
        elif not args.skip_train:
            run([sys.executable, "-u", "-m", "policy.train", "--config", args.config,
                 "--name", name, "--steps", str(args.steps), "--set", *overrides])
        from policy.eval import evaluate
        cache = f"runs/{name}/eval.json"
        if overrides is None and os.path.exists(cache):
            d = json.load(open(cache))             # 基线只评一次，各套件复用
            sr, res = d["success_rate"], d["results"]
        else:
            sr, res = evaluate(f"runs/{name}", episodes=args.episodes, seed=1000,
                               video=f"videos/{name}.mp4")
            json.dump(dict(success_rate=sr, results=res), open(cache, "w"),
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
