#!/bin/bash
# 队列 v3：把「预训练语言编码器」提到视觉前面——泛化测试显示换措辞时成功率是 0%，
# 这一组直接冲那个 0%。
set -x
source .venv/bin/activate
python -u -m policy.ablate --suite pretrained_lang --steps 16000 --episodes 40
python -u -m policy.ablate --suite vision          --steps 16000 --episodes 40
python -u -m policy.ablate --suite cams            --steps 16000 --episodes 40
python -u -m policy.ablate --suite chunk           --steps 16000 --episodes 40
python -u -m policy.ablate --suite heads           --steps 16000 --episodes 40
python -u -m policy.report
