# StarVLA 本地闭环 VLA 项目

在 Mac mini（M4, 16 GB, 无 CUDA）上从零搭一个**能闭环跑的 VLA**：
桌面 MuJoCo 环境 → 脚本专家采数据 → 视觉-语言-动作策略 → 放回仿真里算成功率。
总体计划见 [PLAN.md](PLAN.md)。姊妹项目是 `~/Documents/SONIC`（人形运控），约定一致。

## 环境

- Python 用 `uv` 管理的 3.11 虚拟环境（`.venv/`），不要用系统 Python 3.14。
- 仿真用 MuJoCo（CPU），训练用 PyTorch MPS。Isaac Lab / CUDA 在 macOS 上不可用，不要尝试。
- `data/`、`third_party/`、`runs/`、`.venv/`、`videos/*.mp4` 不进 git。

## Git

远端：`git@github.com:ryf1123/starvla.git`，主分支 `main`。
推送规则见 `.claude/skills/git-push/SKILL.md`，推送前先读。

## 飞书

项目文档「StarVLA项目」在飞书 wiki，通过本机 lark-cli 读写。
token、命令和坑见 `.claude/skills/feishu-doc/SKILL.md`。

## 文档标准

写任何实验记录、方法说明、消融结果之前先读 `.claude/skills/teaching-doc/SKILL.md`：
每个概念都要配**标注图 + 真实数字算例 + 改一个变量的扫参动画**。
这是本项目最重要的约定——产出是"看完能自己改参数并预测结果"的文档，不是流水账。

## 约定

- 每完成 PLAN.md 里的一环，在 `notes/` 下写一页踩坑笔记。
- 代码目录：`sim/`（环境 + IK）、`expert/`（脚本专家 + 采数据）、`policy/`（数据集/模型/训练/闭环评测）、`scripts/`（讲解图和视频）。
- 每个实验一个 `runs/<name>/`：`config.yaml` + `log.jsonl` + `latest.pt`。
- 提交信息用英文，一句话说清改了什么。
