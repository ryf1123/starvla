#!/bin/bash
# 队列 v6：文献驱动的三个改动，各自单独一组对照。
#   1) 放置精度修正（专家松手高度 3.5cm → 1.2cm）      —— 针对当前最大的失败桶
#   2) 域随机化（相机位姿/光照/桌面颜色）              —— 文献里泛化的主要来源
#   3) 多步历史观测（最近 3 帧状态+动作）              —— RoboVLMs 600 组实验的结论
# 每组只改一个变量，基线是 bc_v4（已知最优配置 + place800 数据）。
set -x
source .venv/bin/activate
python -u -m policy.eval --run runs/bc_v4 --episodes 80 --seed 1000 --out runs/bc_v4/eval.json --video videos/bc_v4.mp4

# 1) 放置精度
python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_place --steps 16000 --set data.path=data/demos/place_v2
python -u -m policy.eval --run runs/bc_v5_place --episodes 80 --seed 1000 --out runs/bc_v5_place/eval.json

# 3) 多步历史观测（用同一份 place_v2 数据，和上一组配对比较）
python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_hist --steps 16000 --set data.path=data/demos/place_v2 model.state_history=3
python -u -m policy.eval --run runs/bc_v5_hist --episodes 80 --seed 1000 --out runs/bc_v5_hist/eval.json

# 2) 域随机化
python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_dr --steps 16000 --set data.path=data/demos/place_dr eval.dr=1.0
python -u -m policy.eval --run runs/bc_v5_dr --episodes 80 --seed 1000 --out runs/bc_v5_dr/eval.json --video videos/bc_v5_dr.mp4

# 辅助定位监督
python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_aux --steps 16000 --set data.path=data/demos/place_v2 model.aux_weight=1.0
python -u -m policy.eval --run runs/bc_v5_aux --episodes 80 --seed 1000 --out runs/bc_v5_aux/eval.json

python -u -m policy.report
