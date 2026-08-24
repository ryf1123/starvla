#!/bin/zsh
cd ~/Documents/StarVLA
while pgrep -f "scripts_queue_backbone|scripts_queue_chain4|expert.collect" > /dev/null; do sleep 180; done
./scripts_queue_density.sh >> runs/queue_density.log 2>&1
