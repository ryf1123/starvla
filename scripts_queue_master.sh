#!/bin/zsh
# 今晚剩下的实验，按价值排序。前一个跑完自动接下一个。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
# 等当前这个训练（mm_none_regress）跑完
while pgrep -f "policy.train" > /dev/null; do sleep 60; done
python -u -m policy.eval --run runs/mm_none_regress --episodes 80 --out runs/mm_none_regress/eval.json

# ① 多峰任务 any_plate：notes/18 的预注册预测在这里被检验（最高优先级）
for H in regress discrete diffusion; do
  python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name any_$H \
    --set data.path=data/demos/place_any eval.any_plate=true model.head=$H --steps 16000
  python -u -m policy.eval --run runs/any_$H --episodes 80 --out runs/any_$H/eval.json
done
python -u -m policy.eval --run runs/any_discrete --episodes 80 --decode expect \
  --out runs/any_discrete/eval_expect.json

# ② 扩大指令空间：36 条指令上，字符级 CLS vs 冻结句向量
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name rel_cls_hist \
  --set data.path=data/demos/place_rel eval.same_color_prob=0.5 model.lang_mode=cls --steps 16000
python -u -m policy.eval --run runs/rel_cls_hist --episodes 80 --out runs/rel_cls_hist/eval.json
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name rel_ppool_hist \
  --set data.path=data/demos/place_rel eval.same_color_prob=0.5 model.lang_mode=ppool --steps 16000
python -u -m policy.eval --run runs/rel_ppool_hist --episodes 80 --out runs/rel_ppool_hist/eval.json
python -u -m policy.generalize --run runs/rel_cls_hist --episodes 40 --cases base paraphrase terse swap_role
python -u -m policy.generalize --run runs/rel_ppool_hist --episodes 40 --cases base paraphrase terse swap_role

# ③ 扩散头：是预算不够还是网络太小（只改预算）
python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name head_diffusion_long \
  --set model.head=diffusion --steps 32000
python -u -m policy.eval --run runs/head_diffusion_long --episodes 80 --out runs/head_diffusion_long/eval.json

# ④ 多峰的另一种问法：没有指令时，各个头会不会"挑一个目标去做"
for H in discrete diffusion; do
  python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name mm_none_$H \
    --set model.lang_mode=none model.head=$H --steps 16000
  python -u -m policy.eval --run runs/mm_none_$H --episodes 80 --out runs/mm_none_$H/eval.json
done
