#!/bin/zsh
# notes/08 预测 2 的检验：没有指令时目标分布是多峰的，
# 回归头输出"平均动作"（实测 0%），离散/扩散头应该会挑一个目标去做（≈17% 随机猜）。
# 三个臂用同一套最佳配置，只改 lang_mode=none + 动作头。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
for H in regress discrete diffusion; do
  python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name mm_none_$H \
    --set model.lang_mode=none model.head=$H --steps 16000
  python -u -m policy.eval --run runs/mm_none_$H --episodes 80
done
