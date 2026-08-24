#!/bin/bash
# 队列 v7（接在 v6 之后）：
#   1) 动作噪声 σ=0.15 —— 隔离对照显示 σ=0.05 只把状态覆盖拓宽 1.4 mm，几乎等于没有
#   2) 更高分辨率 160×160 —— 剩下的失败几乎全是放置精度，分辨率是精度的硬上限之一
#   3) 关键比较跑到 200 局 + McNemar 配对检验（PhAIL 2026：n=40 的区间是 ±14 个百分点）
set -x
source .venv/bin/activate
python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_noise --steps 16000 --set data.path=data/demos/place_v2_n15
python -u -m policy.eval --run runs/bc_v5_noise --episodes 80 --seed 1000 --out runs/bc_v5_noise/eval.json

python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_res160 --steps 16000 --set data.path=data/demos/place_v2_160 eval.img_hw=160
python -u -m policy.eval --run runs/bc_v5_res160 --episodes 80 --seed 1000 --out runs/bc_v5_res160/eval.json

python -u -m policy.report
