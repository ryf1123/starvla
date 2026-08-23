#!/bin/bash
# 夜间实验队列（v2）。
# 关键改动：统一 16000 步。8000 步时同一配置只有 10% 成功率，16000 步是 72.5%——
# 8000 步测的是"谁收敛得快"，不是"谁能力强"。
# 每个套件的基线臂直接复用 runs/bc_v3（16000 步），只训变体，省一半时间。
set -x
source .venv/bin/activate
python -u -m policy.ablate --suite lang            --steps 16000 --episodes 40
python -u -m policy.ablate --suite vision          --steps 16000 --episodes 40
python -u -m policy.ablate --suite pretrained_lang --steps 16000 --episodes 40
python -u -m policy.ablate --suite cams            --steps 16000 --episodes 40
python -u -m policy.ablate --suite chunk           --steps 16000 --episodes 40
python -u -m policy.ablate --suite heads           --steps 16000 --episodes 40
python -u -m policy.report
