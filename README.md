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
sim/tabletop_env.py  20 Hz 控制环、5 维末端动作、观测、成功判据、世界→像素投影
expert/scripted.py   7 段状态机专家（用特权信息，成功率 100%，抓空会重试）
expert/collect.py    8 进程采演示；带 DAgger 式扰动，生成"歪掉之后怎么纠正"的数据
policy/dataset.py    动作分块数据集 + 字符词表 + 平移增强
policy/model.py      视觉（从零 CNN / ResNet18）+ 空间 softmax + 语言编码 + FiLM + 三种动作头
policy/train.py      训练入口（MPS），带视觉分支健康检查
policy/eval.py       闭环评测 + 时间集成 + 失败归因 + 录像
policy/ablate.py     消融套件：lang / vision / cams / chunk / heads / backbone / aux / data
policy/generalize.py 泛化边界测试（干扰物、角落、换措辞、角色对调、更多盘子）
policy/report.py     把所有 runs/ 汇总成一张表 + 曲线图
policy/compare.py    两个策略配对同种子 + McNemar 精确检验（scripts/compare.py）
sim/randomize.py     域随机化：相机位姿 / 光照 / 桌面颜色（不动色相）
scripts/             讲解图和视频生成脚本（explain_*.py、diagnose_grasp.py、view.py）
configs/*.yaml       一个实验一份配置，命令行 --set a.b=c 覆盖
runs/<name>/         config.yaml、log.jsonl、latest.pt、eval.json
```

## 可换的部件（一次只改一个，用闭环成功率说话）

| 配置项 | 取值 | 在问什么 |
|---|---|---|
| `model.lang_mode` | none / bow / seq / cls | 语言到底有没有起作用 |
| `model.film` | true / false | 语言该在视觉里生效还是最后拼接 |
| `model.backbone` | cnn / resnet18 | 预训练视觉特征能不能替代更多数据 |
| `model.ss_raw` | true / false | 空间 softmax 前接不接 ReLU（接了会静悄悄失灵） |
| `model.last_stride` | 1 / 2 | 特征图 16×16 还是 8×8 |
| `model.head` | regress / discrete / diffusion | ACT / RT-1 / Diffusion Policy 三条路线 |
| `model.horizon` | 1 / 4 / 8 / 16 | 动作分块多长 |
| `model.cams` | [front,wrist] / [front] / [wrist] | 腕部相机值多少 |
| `model.aux_weight` | 0 / 1 | 辅助定位监督能不能替代更多数据 |
| `model.state_history` | 1 / 3 | 多步历史观测（目前唯一统计显著的改进） |
| `data.limit` | 75 / 150 / 300 / 800 | 数据量曲线 |
| `eval.ensemble_k` | 1 / 4 | 时间集成 |
| `eval.stride` | 1 / 4 / 8 | 每步重规划还是整块开环执行 |
| `eval.any_plate` | false / true | **多峰任务**：指令不指定盘子，放进任意一个都算成功 |
| `--decode`（推理侧） | argmax / expect | 离散头怎么解码（不用重训） |

```bash
python -m policy.ablate --suite lang --steps 8000 --episodes 40
python -m policy.report --curves
python -m policy.generalize --run runs/bc_v3
python -m scripts.compare --a runs/A --b runs/B --episodes 200   # 配对 + McNemar
python -m scripts.diagnose_multimodality --k 16 [--any-plate]    # 数据的条件变异下限
```

**造完任何一个新任务/新数据版本，先跑一遍 `diagnose_multimodality`。**
它给出"任何模型都不可能超过"的两条下限（均值式模型一条、采样式模型一条）。
不知道下限，就分不清"模型不行"和"任务本来就没那么难/那么随机"——
姊妹项目在这上面栽过，我们照搬了它的做法，见 [notes/18](notes/18-造一个真正多峰的任务.md)。

## 一分钟跑通全链路

```bash
python -m expert.collect --name demo50 --episodes 50 --workers 8      # 采 50 条（约 40 秒）
python -m policy.train --config configs/bc_place.yaml --name quick \
    --set data.path=data/demos/demo50 --steps 2000                    # 训 3 分钟
python -m policy.eval --run runs/quick --episodes 20 --video videos/quick.mp4
```

正式一轮：800 条演示 + 16000 步训练（约 40 分钟）+ 50 局闭环评测。
当前基线 `bc_v3`：**80 局闭环 83.8%**（关掉时间集成；开 k=4 时是 75%），抓错方块 0 局。

```bash
python -m expert.collect --name place800 --episodes 800 --workers 8 --noise 0.05 --perturb-prob 0.6
python -m policy.train --config configs/bc_place.yaml --name bc_v3 --steps 16000
python -m policy.eval --run runs/bc_v3 --episodes 50 --video videos/bc_v3.mp4
python -m scripts.diagnose_grasp --run runs/bc_v3    # 失败时错在哪一毫米
```

## 这个系统长什么样

| 项 | 取值 | 为什么 |
|---|---|---|
| 控制频率 | 20 Hz（物理 500 Hz，decimation 25） | 抓放任务够用；再快数据量翻倍收益很小 |
| 动作空间 | `[dx, dy, dz, dyaw, 夹爪]` ∈ [-1,1]⁵ | 末端增量和相机同一坐标系，比关节角数据效率高得多 |
| 动作分块 | 预测未来 8 步，推理时时间集成 | 抑制复合误差和抖动（ACT 的做法） |
| 观测 | 128×128 前视 + 128×128 腕视 + 7 维本体 + 中文指令 | 腕部相机决定最后 2 cm 能不能抓住 |
| 语言 | 字符嵌入 + 1 层 Transformer → FiLM 调制视觉 | 词袋会把「红→黄」和「黄→红」编码成同一个东西 |
| 策略 | 1.3 M 参数 CNN（可换 ResNet18） | M4 上 20–40 分钟能训完，才谈得上做消融 |
| 数据 | 800 条演示，其中 60% 带中途扰动 | 没有"歪掉之后怎么纠正"的数据，策略抓空一次就永远回不来 |

## 速度参考（M4 10 核 / 16 GB）

| 环节 | 速度 |
|---|---|
| 环境步进（含两路 128×128 渲染） | 单进程 ~35 step/s，8 进程 ~55 step/s |
| 训练（batch 64，MPS） | 10.6 it/s ≈ 680 sample/s |
| 闭环推理 | 每步约 12 ms，实时 |

一个踩过的坑：图像用 `float32` 送 GPU 比 `uint8` 慢 2.7 倍（带宽），归一化要放到设备上做。

## 自检

改完代码先跑这个（约两分钟，覆盖场景/IK/环境/域随机化/专家/模型/采集/训练/闭环）：

```bash
python -m scripts.selftest          # 全部
python -m scripts.selftest --fast   # 跳过采集和训练，20 秒
```

写它的直接理由：它上线第一次运行就抓到一个真 bug——数据增强里裁剪尺寸写死了 128，
会让 160×160 的实验静默地把图裁回 128，而评测喂的是完整 160，训练/评测分辨率对不上。

## 目前的结论（2026-08-25）

### 基线

`bc_v5_hist`：1.34 M 参数，900 条演示，16000 步（40 分钟），**80 局闭环 93.8%**，抓错方块 0 局。
三个训练种子 90.0 / 91.2 / 93.8%。配置 = `place_v2` 数据 + **最近 3 帧本体历史** +
字符级 CLS + FiLM + 每步重规划（`ensemble_k=1`）。

从第一次跑通的 34% 到这里，中间有六次判断被自己的对照实验推翻。
过程和更正都保留在 `notes/`，[notes/02](notes/02-闭环失败诊断.md) 和
[notes/12](notes/12-放置失败的真正原因.md) 是最值得读的两页。

### 动作表示三条路线（同骨干同数据，各 80 局）

| 动作头 | 成功率 | 代表工作 |
|---|---|---|
| 连续回归 + L1 | **93.8%** | ACT |
| 离散 token，argmax 解码 | 35.0% | RT-1 / RT-2 / OpenVLA |
| 离散 token，期望解码（纯推理侧开关） | 46.2% | — |
| 扩散，10 步 DDIM（16000 / 32000 步） | 3.8% / **13.8%** | Diffusion Policy / π0 |

而回归头和离散头的 **val L1 几乎相同**（0.0843 vs 0.0868）。
机制量出来了：离散头 2.6% 的步跳变 > 0.5、夹爪指令每局翻转 5.2 次（专家 2.0）。
见 [notes/17](notes/17-动作表示三条路线.md)。

### 多峰任务：同一个开关，符号相反

造了一个真正多峰的任务（指令不指定盘子、放进任意一个都算成功），
分岔点上的条件变异是原任务的 **7 倍**，而抓取阶段完全不变。

| 只看放置阶段 | 单峰任务 | 多峰任务 |
|---|---|---|
| 回归 | **98.7%** | 60.8% |
| 离散 argmax | 82.4% | 57.1% |
| 离散**期望解码** | **90.2%** | **32.7%**（Fisher p=0.043） |

**同一份权重、同一批种子，只换解码规则**：抓取阶段（单峰）期望解码总是有帮助，
放置阶段（多峰）它把两个合法模式平均掉，方块落在两个盘子中间。
**一个设计选择的好坏，取决于数据的一个可测量的性质**——而这个性质可以在训练之前就量出来。
见 [notes/18](notes/18-造一个真正多峰的任务.md)。

### 语言消融（每组 16000 步 + 40 局闭环，同一套种子）

| lang_mode | 成功率 | 抓错方块 | 一句话 |
|---|---|---|---|
| `ppool`（冻结 bge 句向量） | **85.0%** | 0 | 能表达顺序 |
| `seq` | **80.0%** | 0 | 能表达顺序 |
| `cls`（基线） | **72.5%** | 0 | 能表达顺序 |
| `bow` | 12.5% | 12 | 只有颜色集合，分不出谁抓谁 |
| `cls` 但去掉 FiLM | 5.0% | 18 | 语言拿到了，但没进到视觉里 |
| `none` | 0.0% | 15 | 没有指令，回归头输出平均动作 |

注意：能表达顺序的三种（cls / seq / ppool）换个推理设置排序就会翻过来
（k=1 时 seq 85% / cls 83.8% / ppool 80%），**它们之间的差异全在噪声里**。
真正被分开的是「能表达顺序」和「不能」这两类。

两条分界线：**能不能表达顺序**（80%+ vs 12.5%），**能不能在视觉里生效**（72.5% vs 5%）。
后者的机制是：空间 softmax 只输出「在哪」不输出「是什么」，
所以语言必须在池化**之前**通过 FiLM 作用到特征图上。

### 泛化边界（bc_v3，每项 25 局）

| 测试项 | 成功率 | 说明 |
|---|---|---|
| 训练分布 | 80% | 对照 |
| 5 个干扰方块 | 60% | 掉的是精度 |
| 物体全在角落 | 48% | 掉的是精度 |
| **换措辞** | **0%** | 抓错方块 13/25；换成预训练语言编码器后是 **40%** |
| **极简说法** | **0%** | 抓错方块 12/25；预训练编码器 **52%** |
| 指令角色对调 | 77% | 语序确实被用上了（n=30） |
| 3 个盘子 | 52% | 掉的是精度 |

### 几个反直觉的发现

- **训练 loss 不是指标**：同配置 8000 步 val L1=0.1095 / 成功率 10%，16000 步 val L1=0.0925 / 成功率 72.5%。L1 好 16%，成功率好 7 倍。
- **照搬 ACT 的时间集成是负收益**：k=1 → 83.8%，k=2 → 75%，k=4 → 72.5%，k=8 → 65%，单调下降。
- **代理指标不是目标指标**：我发明了两个"视觉健康度"指标（softmax 尖锐度、关键点跨场景标准差），据此断定视觉分支"瞎了"并动手改。2×2 对照跑完发现：**越尖锐成功率越低**（最尖的 59× 均匀那组只有 32.5%，接近均匀的两组是 87.5% 和 75%），而三个指标（含 val L1）没有一个能预测闭环成功率。我的"修复"是负收益。全过程和原始错误推理保留在 [notes/02](notes/02-闭环失败诊断.md)——**那一页是本仓库最值得读的**。

| 空间 softmax | 特征图 16×16 | 特征图 8×8 |
|---|---|---|
| 原始 logits + 温度 0.1（我的"修复"） | 72.5% | 32.5% |
| ReLU 之后 + 温度 1.0（默认写法） | **87.5%** | 75.0% |
- **任务不需要的信息，模型不会去表示**：第一版任务里颜色集合就足以定位目标，于是「有序」的语言编码器自己退化成了词袋——不是架构的错。
- **动作表示的差距远大于预期**：同一套骨干、同一份数据，回归 **93.8%**、离散 token（argmax）**35.0%**、扩散 **3.8–5.0%**。而回归和离散的 val L1 几乎一样（0.0843 vs 0.0868）。机制量出来了：离散头 **2.6% 的步会跳变 >0.5**、夹爪每局多翻转一倍；换成对 softmax 求期望（纯推理侧开关）就回到 46.2%。见 [notes/17](notes/17-动作表示三条路线.md)。
- **辅助监督 −62 个百分点**：用特权信息教视觉分支定位，辅助任务学到 2.4 像素误差，闭环从 86.2% 掉到 23.8%。见 [notes/15](notes/15-辅助监督反而有害.md)。

完整结果见 [docs/SUMMARY.md](docs/SUMMARY.md)，踩坑过程见 `notes/`。

## 文档

写文档的标准见 `.claude/skills/teaching-doc/SKILL.md`：每个概念都要有
**标注图（在哪）+ 真实数字算例（怎么算）+ 改一个变量的扫参动画（改了会怎样）**。
讲解图统一放 `docs/figs/`，由 `scripts/explain_*.py` 重新生成。
