#!/bin/zsh
# bb_r18 训练已经在跑（4 小时）。等它结束 → 评测 → 数据密度实验。
# 砍掉了 bb_r18_scratch（区分"结构"和"预训练权重"），因为 ResNet18 只有 70 sample/s，
# 一个 16000 步的臂就要 4 小时，优先保住主实验和数据密度实验。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
while pgrep -f "policy.train.*bb_r18" > /dev/null; do sleep 120; done
python -u -m policy.eval --run runs/bb_r18 --episodes 80 --out runs/bb_r18/eval.json
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name rel_cls_big \
  --set data.path=data/demos/place_rel_big eval.same_color_prob=0.5 model.lang_mode=cls --steps 16000
python -u -m policy.eval --run runs/rel_cls_big --episodes 80 --out runs/rel_cls_big/eval.json
