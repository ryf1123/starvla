#!/bin/zsh
# notes/22 的漏洞：我用了和从零 CNN 相同的学习率微调 ResNet18，
# 这会在前几千步把预训练特征冲掉。标准做法是骨干取头部的 1/10。
# 这一组分开"预训练权重有没有用"和"我有没有正确地微调它"。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
while pgrep -f "policy.(train|eval)" > /dev/null; do sleep 120; done
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name bb_r18_lowlr \
  --set model.backbone=resnet18 train.backbone_lr_mult=0.1 --steps 16000
python -u -m policy.eval --run runs/bb_r18_lowlr --episodes 80 --out runs/bb_r18_lowlr/eval.json
