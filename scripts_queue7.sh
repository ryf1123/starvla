#!/bin/bash
# 队列 v8（接在 v7 之后）：针对"抓取对准"这个已确认的病因
#   诊断结论：成败几乎全取决于闭合夹爪那一刻对没对准
#   （成功局水平误差中位数 11.7 mm，失败局 45.6 mm，方块半宽 22 mm）
set -x
source .venv/bin/activate
# 抓取时刻过采样：给决定成败的那几步更大的权重
python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_grasp --steps 16000 \
    --set data.path=data/demos/place_v2 data.grasp_oversample=3
python -u -m policy.eval --run runs/bc_v5_grasp --episodes 80 --seed 1000 --out runs/bc_v5_grasp/eval.json
python -u -m policy.report
