"""小型 VLA 策略：两路图像 + 一句中文指令 + 本体状态 → 未来 H 步动作。

    front ─┐                             ┌── FiLM(γ,β) ──┐
           ├─ CNN ─ 空间 softmax ─ 关键点 ┤               ├─ concat ─ MLP ─ (H,5)
    wrist ─┘                             └───────────────┘
    指令 ─ 字符嵌入 + 位置编码 ─ 1 层 Transformer ─ 均值 ─ 语言向量 z_lang
    state ─ MLP ─────────────────────────────────────────┘

三个值得讲清楚的设计：

1) **空间 softmax（spatial softmax）**：把卷积特征图当成"这个通道认为目标在哪"的
   概率分布，输出期望坐标 (x̄, ȳ)。操作任务里策略要的是位置，不是纹理，
   直接 flatten 会让网络自己从一堆激活里再学一遍"在哪"，而这一步可以白送。
   32 个通道 → 64 维坐标，可视化时能把每个关键点画回图上（docs/figs/keypoints.png）。

2) **语言用有序编码器，不用词袋**：指令「把红色方块放进黄色盘子」里两个颜色词
   顺序决定了谁是被抓的、谁是目标。字符做平均池化会把顺序抹掉，两句话
   「红→黄」和「黄→红」编码完全一样。ablation `lang=bow` 就是拿来看这个坑的。

3) **FiLM 条件化**：语言向量生成每个视觉通道的缩放 γ 和平移 β，
   在卷积中间层作用于特征图（RT-1 / FiLM 的做法）。等价于让"该看红色还是绿色"
   这件事发生在视觉特征里，而不是等到最后拼接时才补救。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("xs", xs.reshape(-1))
        self.register_buffer("ys", ys.reshape(-1))

    def forward(self, feat):                       # (B,C,H,W)
        B, C, H, W = feat.shape
        p = F.softmax(feat.reshape(B, C, H * W), dim=-1)
        x = (p * self.xs).sum(-1)
        y = (p * self.ys).sum(-1)
        return torch.stack([x, y], -1).reshape(B, C * 2), p.reshape(B, C, H, W)


class VisionEncoder(nn.Module):
    """4 层 CNN，第 3 层后接 FiLM。128×128 → 8×8 特征图 → 32 个关键点。"""

    def __init__(self, ch=(32, 64, 128, 128), lang_dim=128, film=True):
        super().__init__()
        c0 = 3
        layers = []
        for c in ch:
            layers += [nn.Conv2d(c0, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.ReLU()]
            c0 = c
        self.stem = nn.Sequential(*layers[:9])      # 前 3 个 block → (B,128,16,16)
        self.tail = nn.Sequential(*layers[9:])      # 第 4 个 block → (B,128,8,8)
        self.film = nn.Linear(lang_dim, 2 * ch[2]) if film else None
        self.ss = SpatialSoftmax(8, 8)
        self.out_dim = ch[-1] * 2

    def forward(self, img, z_lang=None):
        if img.dtype == torch.uint8:                # 归一化放在设备上，省一半带宽
            img = img.float().div_(255.0)
        h = self.stem(img)
        if self.film is not None and z_lang is not None:
            g, b = self.film(z_lang).chunk(2, dim=-1)
            h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        h = self.tail(h)
        kp, heat = self.ss(h)
        return kp, heat


class LanguageEncoder(nn.Module):
    """mode='seq' 有序 Transformer；mode='bow' 字符词袋（消融用）；mode='none' 常数。"""

    def __init__(self, vocab, dim=128, max_len=20, mode="seq"):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        enc = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0.0)
        # enable_nested_tensor=False 是必须的：eval() 模式下 PyTorch 会走 nested tensor 快路径，
        # 而 aten::_nested_tensor_from_mask_left_aligned 在 MPS 上没实现，训练时不报错、
        # 一到验证/闭环推理就崩。见 notes/02-bc-闭环.md
        self.tr = nn.TransformerEncoder(enc, 1, enable_nested_tensor=False)
        self.out_dim = dim

    def forward(self, tokens):
        if self.mode == "none":
            return torch.zeros(tokens.shape[0], self.out_dim, device=tokens.device)
        pad = tokens == 0
        x = self.emb(tokens)
        if self.mode == "bow":
            return x.masked_fill(pad[..., None], 0).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        x = x + self.pos[:, : x.shape[1]]
        x = self.tr(x, src_key_padding_mask=pad)
        x = x.masked_fill(pad[..., None], 0).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        return x


class VLAPolicy(nn.Module):
    def __init__(self, vocab, horizon=8, act_dim=5, state_dim=7, lang_dim=128,
                 hidden=512, lang_mode="seq", film=True, cams=("front", "wrist")):
        super().__init__()
        self.cams, self.H, self.act_dim = tuple(cams), horizon, act_dim
        self.lang = LanguageEncoder(vocab, lang_dim, mode=lang_mode)
        self.enc = nn.ModuleDict({c: VisionEncoder(lang_dim=lang_dim, film=film) for c in self.cams})
        vis_dim = sum(self.enc[c].out_dim for c in self.cams)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(vis_dim + 64 + lang_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * act_dim))

    def forward(self, batch, return_heat=False):
        z = self.lang(batch["tokens"])
        feats, heats = [], {}
        for c in self.cams:
            kp, heat = self.enc[c](batch[c], z)
            feats.append(kp)
            heats[c] = heat
        h = torch.cat(feats + [self.state_mlp(batch["state"]), z], dim=-1)
        a = self.head(h).reshape(-1, self.H, self.act_dim)
        a = torch.cat([a[..., :4].tanh(), a[..., 4:]], dim=-1)   # 位移/偏航压到 [-1,1]，夹爪留给 BCE 之外的 L1
        return (a, heats) if return_heat else a

    def loss(self, batch):
        pred = self(batch)
        err = (pred - batch["action"]).abs().mean(-1) * batch["mask"]
        l1 = err.sum() / batch["mask"].sum().clamp(min=1)
        with torch.no_grad():
            grip = (((pred[..., 4] > 0) == (batch["action"][..., 4] > 0)).float() * batch["mask"]
                    ).sum() / batch["mask"].sum().clamp(min=1)
        return l1, {"l1": l1.item(), "grip_acc": grip.item()}
