#!/bin/bash
# 队列 v9（最后）：量训练随机种子带来的方差。
# 目前所有消融结论都建立在**单个训练种子**上——这是我们最大的未知方差来源。
# 同一配置（= bc_v4）换两个种子重训，和 bc_v4 三者两两配对比较。
set -x
source .venv/bin/activate
for S in 1 2; do
  python -u -m policy.train --config configs/bc_place.yaml --name bc_v4_s$S --steps 16000 --set train.seed=$S
  python -u -m policy.eval --run runs/bc_v4_s$S --episodes 80 --seed 1000 --out runs/bc_v4_s$S/eval.json
done
python -u -m policy.report
