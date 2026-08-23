#!/bin/bash
# 队列 v4：视觉消融 → 新指令空间的基线 → 动作头 → 相机 / 分块
set -x
source .venv/bin/activate
python -u -m policy.ablate --suite vision --steps 16000 --episodes 40
# 扩大指令空间之后的基线（12 条指令 → 36 条，加入「左边的/右边的」）
python -u -m policy.train --config configs/bc_place.yaml --name bc_rel --steps 16000 \
    --set data.path=data/demos/place_rel
python -u -m policy.eval --run runs/bc_rel --episodes 40 --seed 1000 --out runs/bc_rel/eval.json \
    --video videos/bc_rel.mp4
python -u -m policy.ablate --suite heads --steps 16000 --episodes 40
python -u -m policy.ablate --suite cams  --steps 16000 --episodes 40
python -u -m policy.ablate --suite chunk --steps 16000 --episodes 40
python -u -m policy.report
