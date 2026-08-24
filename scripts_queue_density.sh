#!/bin/zsh
# notes/20 的第三条假设：36 条指令上成功率掉一半，是不是单纯因为
# 每种条件的数据密度降了（900 条摊到 36 条指令 = 每条 25 局）。
# 同样 36 条指令，数据加到 2700 条（每条 75 局，和原来 12 条指令时一样）。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name rel_cls_big \
  --set data.path=data/demos/place_rel_big eval.same_color_prob=0.5 model.lang_mode=cls --steps 16000
python -u -m policy.eval --run runs/rel_cls_big --episodes 80 --out runs/rel_cls_big/eval.json
