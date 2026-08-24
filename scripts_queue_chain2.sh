#!/bin/zsh
cd ~/Documents/StarVLA
while kill -0 38298 2>/dev/null; do sleep 120; done
./scripts_queue_diff.sh >> runs/queue_diff.log 2>&1
