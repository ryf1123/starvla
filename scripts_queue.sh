#!/bin/bash
# 夜间实验队列。每个套件内部按顺序训练 + 评测；单个实验的结果写 runs/<name>/eval.json，
# 套件汇总写 runs/ablate_<suite>.json。中途被打断不丢已完成的实验。
# 统一 8000 步（基线 bc_v3 是 16000 步 74%；8000 步时约 58%），目的是横向比较。
set -x
source .venv/bin/activate
python -u -m policy.ablate --suite vision --steps 8000 --episodes 40
python -u -m policy.ablate --suite lang   --steps 8000 --episodes 40
python -u -m policy.ablate --suite cams   --steps 8000 --episodes 40
python -u -m policy.ablate --suite chunk  --steps 8000 --episodes 40
python -u -m policy.ablate --suite pretrained_lang --steps 8000 --episodes 40
python -u -m policy.ablate --suite heads  --steps 8000 --episodes 40
python -u -m policy.report
