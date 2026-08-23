#!/bin/bash
# 队列 v3。顺序按"能回答什么问题"排：
# 1) pretrained_lang —— 泛化测试里换措辞是 0%，这组直接冲那个 0%
# 2) vision          —— 验证"空间 softmax 接在 ReLU 后会失灵"这个诊断
# 3) heads           —— lang_none 是被单峰回归头的"平均动作"害死的，
#                       离散 token / 扩散头能不能表达多峰
# 4) cams / chunk    —— 常规消融
set -x
source .venv/bin/activate
python -u -m policy.ablate --suite pretrained_lang --steps 16000 --episodes 40
python -u -m policy.ablate --suite vision          --steps 16000 --episodes 40
python -u -m policy.ablate --suite heads           --steps 16000 --episodes 40
python -u -m policy.ablate --suite cams            --steps 16000 --episodes 40
python -u -m policy.ablate --suite chunk           --steps 16000 --episodes 40
python -u -m policy.report
