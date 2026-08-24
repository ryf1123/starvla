#!/bin/bash
# 队列 v9（最后）：量训练随机种子带来的方差，同时复现"多步历史"这个关键结论。
# 目前所有消融结论都建立在**单个训练种子**上——这是最大的未知方差来源。
# 用新的最优配置（多步历史观测）换两个种子重训，和原来的 seed=0 三者互相配对比较。
set -x
source .venv/bin/activate
for S in 1 2; do
  python -u -m policy.train --config configs/bc_place.yaml --name bc_v5_hist_s$S --steps 16000 \
      --set data.path=data/demos/place_v2 model.state_history=3 train.seed=$S
  python -u -m policy.eval --run runs/bc_v5_hist_s$S --episodes 80 --seed 1000 --out runs/bc_v5_hist_s$S/eval.json
done
python -u -m policy.report
