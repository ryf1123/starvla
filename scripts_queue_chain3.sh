#!/bin/zsh
cd ~/Documents/StarVLA
# 等前两段链和数据采集都结束
while pgrep -f "scripts_queue_chain2|scripts_queue_diff|expert.collect" > /dev/null; do sleep 120; done
while pgrep -f "policy.(train|eval|generalize)" > /dev/null; do sleep 60; done
./scripts_queue_anyplate.sh >> runs/queue_anyplate.log 2>&1
