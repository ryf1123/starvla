#!/bin/zsh
# 条件变异分析说瓶颈是感知/容量（回归离自己的下限差 2.95 倍），
# 预训练视觉骨干是最对症的一条，而且代码早就写好了从没跑过。
# 三个臂分开"结构"和"预训练权重"两个因素。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name bb_r18 \
  --set model.backbone=resnet18 --steps 16000
python -u -m policy.eval --run runs/bb_r18 --episodes 80 --out runs/bb_r18/eval.json
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name bb_r18_scratch \
  --set model.backbone=resnet18 model.pretrained=false --steps 16000
python -u -m policy.eval --run runs/bb_r18_scratch --episodes 80 --out runs/bb_r18_scratch/eval.json
