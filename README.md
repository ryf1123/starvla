# StarVLA：在 Mac mini 上从零跑通一个闭环 VLA

在 Mac mini（Apple M4，16 GB，**没有 CUDA**）上，把视觉-语言-动作模型（VLA）的整条链路
自己搭一遍并且**闭环跑起来**：

**MuJoCo 桌面环境 → 脚本专家采演示 → 视觉+语言→动作分块策略 → 放回仿真里数成功率**

任务是语言条件的桌面抓放：桌上 3 个不同颜色的方块和 1 个盘子，指令形如
「把红色方块放进黄色盘子」。只有指令能告诉策略该抓哪个，所以"语言到底有没有起作用"
是可以被实验证伪的——这正是本项目最想讲清楚的一件事。

目标是学懂每个设计决策，不是刷分。实验笔记、视频和消融在飞书「StarVLA项目」；
总体计划见 [PLAN.md](PLAN.md)。姊妹项目 [SONIC](https://github.com/ryf1123/sonic) 做的是人形全身运控。

## 0. 环境

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install mujoco torch torchvision numpy scipy imageio imageio-ffmpeg tensorboard tqdm matplotlib pyyaml
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
python -c "import mujoco, torch; print(mujoco.__version__, torch.backends.mps.is_available())"
```

看一眼场景（交互窗口，macOS 上要用 `mjpython`）：

```bash
python -m sim.assets                       # 打印场景规格
mjpython scripts/view.py                   # 拖着看 Panda + 桌面
```

## 目录

```
sim/assets.py        程序化搭场景（Panda + 桌 + 方块 + 盘子 + 前视/腕部相机）
sim/ik.py            阻尼最小二乘 IK：TCP 位姿 → 7 个关节角
sim/tasks.py         任务采样与指令模板（TaskSpec 可序列化，能精确复现任一局）
sim/tabletop_env.py  20 Hz 控制环、5 维末端动作、观测、成功判据
expert/scripted.py   7 段状态机专家（用特权信息，成功率 100%）
expert/collect.py    8 进程采演示 → data/demos/<name>/shard_*.npz
policy/dataset.py    动作分块数据集 + 字符词表 + 平移增强
policy/model.py      CNN + 空间 softmax + 语言 Transformer + FiLM + 分块动作头（1.3 M 参数）
policy/train.py      训练入口（MPS）
policy/eval.py       闭环评测 + 时间集成 + 录像
scripts/             讲解图和视频生成脚本（explain_*.py）
configs/*.yaml       一个实验一份配置，命令行 --set a.b=c 覆盖
runs/<name>/         config.yaml、log.jsonl、latest.pt
```

## 一分钟跑通全链路

```bash
python -m expert.collect --name demo50 --episodes 50 --workers 8      # 采 50 条（约 40 秒）
python -m policy.train --config configs/bc_place.yaml --name quick \
    --set data.path=data/demos/demo50 --steps 2000                    # 训 3 分钟
python -m policy.eval --run runs/quick --episodes 20 --video videos/quick.mp4
```

正式一轮：500 条演示 + 12000 步训练（约 19 分钟）+ 50 局闭环评测。

```bash
python -m expert.collect --name place500 --episodes 500 --workers 8 --noise 0.05
python -m policy.train --config configs/bc_place.yaml --name bc_v1
python -m policy.eval --run runs/bc_v1 --episodes 50 --video videos/bc_v1.mp4
```

## 这个系统长什么样

| 项 | 取值 | 为什么 |
|---|---|---|
| 控制频率 | 20 Hz（物理 500 Hz，decimation 25） | 抓放任务够用；再快数据量翻倍收益很小 |
| 动作空间 | `[dx, dy, dz, dyaw, 夹爪]` ∈ [-1,1]⁵ | 末端增量和相机同一坐标系，比关节角数据效率高得多 |
| 动作分块 | 预测未来 8 步，推理时时间集成 | 抑制复合误差和抖动（ACT 的做法） |
| 观测 | 128×128 前视 + 128×128 腕视 + 7 维本体 + 中文指令 | 腕部相机决定最后 2 cm 能不能抓住 |
| 语言 | 字符嵌入 + 1 层 Transformer → FiLM 调制视觉 | 词袋会把「红→黄」和「黄→红」编码成同一个东西 |
| 策略 | 1.3 M 参数 CNN | M4 上 19 分钟能训完，才谈得上做消融 |

## 速度参考（M4 10 核 / 16 GB）

| 环节 | 速度 |
|---|---|
| 环境步进（含两路 128×128 渲染） | 单进程 ~35 step/s，8 进程 ~55 step/s |
| 训练（batch 64，MPS） | 10.6 it/s ≈ 680 sample/s |
| 闭环推理 | 每步约 12 ms，实时 |

一个踩过的坑：图像用 `float32` 送 GPU 比 `uint8` 慢 2.7 倍（带宽），归一化要放到设备上做。

## 文档

写文档的标准见 `.claude/skills/teaching-doc/SKILL.md`：每个概念都要有
**标注图（在哪）+ 真实数字算例（怎么算）+ 改一个变量的扫参动画（改了会怎样）**。
讲解图统一放 `docs/figs/`，由 `scripts/explain_*.py` 重新生成。
