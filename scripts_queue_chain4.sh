#!/bin/zsh
cd ~/Documents/StarVLA
while pgrep -f scripts_queue_master > /dev/null; do sleep 180; done
./scripts_queue_backbone.sh >> runs/queue_backbone.log 2>&1
