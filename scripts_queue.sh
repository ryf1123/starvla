#!/bin/bash
# 夜间实验队列：一个接一个跑，每个套件跑完自动评测并写 runs/ablate_<suite>.json。
# 用法：nohup ./scripts_queue.sh > runs/queue.log 2>&1 &
set -x
source .venv/bin/activate
python -u -m policy.ablate --suite vision --steps 8000  --episodes 40
python -u -m policy.ablate --suite lang   --steps 8000  --episodes 40
python -u -m policy.ablate --suite cams   --steps 8000  --episodes 40
python -u -m policy.ablate --suite chunk  --steps 8000  --episodes 40
python -u -m policy.report
