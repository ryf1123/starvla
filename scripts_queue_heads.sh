#!/bin/zsh
# 动作表示三条路线：在当前最佳配置（place_v2 + state_history=3）上比回归 / 离散 / 扩散
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name head_discrete_h --set model.head=discrete --steps 16000
python -u -m policy.eval  --run runs/head_discrete_h --episodes 80 --out runs/head_discrete_h/eval.json --video videos/head_discrete_h.mp4
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name head_diffusion_h --set model.head=diffusion --steps 16000
python -u -m policy.eval  --run runs/head_diffusion_h --episodes 80 --out runs/head_diffusion_h/eval.json --video videos/head_diffusion_h.mp4
python -u -m policy.eval  --run runs/bc_v5_hist --episodes 80 --out runs/bc_v5_hist/eval80.json
