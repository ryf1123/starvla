#!/bin/zsh
# 数据量曲线：36 条指令的任务上，从 300 条扫到 2700 条。
# 两个已知点（900→47.5%、2700→77.5%）在饱和曲线的陡峭段，
# 补上 300 和 1800 就能看出拐点在哪。用 data.limit 从同一份 2700 条里取子集，
# 保证是嵌套的子集关系（同一个随机种子的置换前 N 条）。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
for N in 300 600 1800; do
  python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name rel_n$N \
    --set data.path=data/demos/place_rel_big data.limit=$N eval.same_color_prob=0.5 --steps 16000
  python -u -m policy.eval --run runs/rel_n$N --episodes 80 --out runs/rel_n$N/eval.json
done
