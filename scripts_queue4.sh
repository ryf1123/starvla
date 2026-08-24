#!/bin/bash
# 队列 v5：目标是「更好的效果」。默认配置已换成实测更优的一边：
#   ss_raw=false（空间 softmax 保留 ReLU + 温度 1.0）、ensemble_k=1（关掉时间集成）
set -x
source .venv/bin/activate
# 1) 新基线：已知最优配置
python -u -m policy.train --config configs/bc_place.yaml --name bc_v4 --steps 16000
python -u -m policy.eval --run runs/bc_v4 --episodes 80 --seed 1000 --out runs/bc_v4/eval.json --video videos/bc_v4.mp4
# 2) 训练更久有没有用
python -u -m policy.train --config configs/bc_place.yaml --name bc_v4_long --steps 28000
python -u -m policy.eval --run runs/bc_v4_long --episodes 80 --seed 1000 --out runs/bc_v4_long/eval.json
# 3) 最优配置 + 辅助定位监督
python -u -m policy.train --config configs/bc_place.yaml --name bc_v4_aux --steps 16000 --set model.aux_weight=1.0
python -u -m policy.eval --run runs/bc_v4_aux --episodes 80 --seed 1000 --out runs/bc_v4_aux/eval.json
# 4) 最优配置在「36 条指令」的难任务上
python -u -m policy.train --config configs/bc_place.yaml --name bc_rel2 --steps 16000 \
    --set data.path=data/demos/place_rel eval.same_color_prob=0.5
python -u -m policy.eval --run runs/bc_rel2 --episodes 60 --seed 1000 --out runs/bc_rel2/eval.json --video videos/bc_rel2.mp4
python -u -m policy.report
