#!/bin/zsh
# notes/17 留下的悬念：扩散头差，是训练预算不够，还是去噪网络太小？
# 这一组只改预算（同样的 3 层 MLP 去噪器，16000 → 32000 步），把两个因素分开。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name head_diffusion_long \
  --set model.head=diffusion --steps 32000
python -u -m policy.eval --run runs/head_diffusion_long --episodes 80 --out runs/head_diffusion_long/eval.json
