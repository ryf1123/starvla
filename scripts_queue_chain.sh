#!/bin/zsh
# 串起今晚剩下的三组实验，机器不空转。
cd ~/Documents/StarVLA
while pgrep -f "policy.(train|eval|generalize)" > /dev/null; do sleep 60; done
./scripts_queue_multimodal.sh >> runs/queue_multimodal.log 2>&1
./scripts_queue_rel.sh        >> runs/queue_rel.log 2>&1
