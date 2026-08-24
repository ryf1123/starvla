#!/bin/zsh
# notes/18 的多峰任务实验：同一套骨干、同一份 any_plate 数据，只换动作头。
# 预测已预注册在 notes/18：回归会掉（走两个盘子中间），离散+argmax 会承诺，
# 离散+期望会退回到回归的水平，扩散大概率还是最差。
cd ~/Documents/StarVLA && source .venv/bin/activate
set -x
for H in regress discrete diffusion; do
  python -u -m policy.train --config runs/bc_v5_hist/config.yaml --name any_$H \
    --set data.path=data/demos/place_any eval.any_plate=true model.head=$H --steps 16000
  python -u -m policy.eval --run runs/any_$H --episodes 80 --out runs/any_$H/eval.json
done
# 交叉检验 notes/17 的机制解释：期望解码在多峰任务上应该退回到"走中间"
python -u -m policy.eval --run runs/any_discrete --episodes 80 --decode expect \
  --out runs/any_discrete/eval_expect.json
