#!/bin/zsh
# 扩大指令空间的真正考验：36 条指令（同色方块 + 左右词）上，
# 用当前最佳配置重训，并比较字符级 CLS vs 冻结句向量。
# bc_rel 的 37.5% 是旧配置（ss_raw=true / ensemble_k=4 / 无历史）跑出来的，任务难度和配置缺陷混在一起。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name rel_cls_hist \
  --set data.path=data/demos/place_rel eval.same_color_prob=0.5 model.lang_mode=cls --steps 16000
python -u -m policy.eval --run runs/rel_cls_hist --episodes 80 --out runs/rel_cls_hist/eval.json --video videos/rel_cls_hist.mp4
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name rel_ppool_hist \
  --set data.path=data/demos/place_rel eval.same_color_prob=0.5 model.lang_mode=ppool --steps 16000
python -u -m policy.eval --run runs/rel_ppool_hist --episodes 80 --out runs/rel_ppool_hist/eval.json --video videos/rel_ppool_hist.mp4
python -u -m policy.generalize --run runs/rel_cls_hist --episodes 40 --cases base paraphrase terse swap_role
python -u -m policy.generalize --run runs/rel_ppool_hist --episodes 40 --cases base paraphrase terse swap_role
