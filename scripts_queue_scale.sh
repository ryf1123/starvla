#!/bin/zsh
# 数据密度在 36 条指令的任务上值 +30 个百分点。那么在原任务（12 条指令）上，
# 现在的 93.8% 是不是也被数据卡着？同配置，只把 900 条换成 2700 条。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
while pgrep -f "policy.(train|eval)|expert.collect" > /dev/null; do sleep 120; done
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name bc_v6_big \
  --set data.path=data/demos/place_v2_big --steps 16000
python -u -m policy.eval --run runs/bc_v6_big --episodes 80 --out runs/bc_v6_big/eval.json
python -u -m scripts.compare --a runs/bc_v6_big --b runs/bc_v5_hist --episodes 200 \
  --out runs/cmp_data_scale.json
