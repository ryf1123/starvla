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
    """四种模式，对应四组对照实验：

        none  常数向量——策略拿不到指令，只能瞎猜该抓哪个方块
        bow   字符嵌入求平均（词袋）——「红→黄」和「黄→红」字符集相同，编码必然相同
        seq   加位置编码的 1 层 Transformer + 均值池化——**看起来**有序，实测仍会退化成词袋
        cls   同上但用一个可学的 [CLS] token 读出，位置编码初始化放大到 0.2

    seq 退化这件事是实测出来的（见 notes/03-语言.md）：训完之后
    ‖z(红→黄) − z(黄→红)‖ = 0.046，而换一组颜色是 2.72——均值池化把顺序抹平了，
    位置编码的梯度信号又太弱（std 只从 0.020 长到 0.023）。cls 模式是修复版。
    """

    def __init__(self, vocab, dim=128, max_len=20, mode="seq"):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        pos_scale = 0.2 if mode == "cls" else 0.02
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, dim) * pos_scale)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
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
        if self.mode == "cls":
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
            pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1)
            x = x + self.pos[:, : x.shape[1]]
            return self.tr(x, src_key_padding_mask=pad)[:, 0]
        x = x + self.pos[:, : x.shape[1]]
        x = self.tr(x, src_key_padding_mask=pad)
        return x.masked_fill(pad[..., None], 0).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)


def timestep_embedding(t, dim):
    """扩散头用的正弦时间嵌入（和 Transformer 的位置编码同一个套路）。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class VLAPolicy(nn.Module):
    """三种动作头共用同一套视觉/语言骨干，只换最后一步怎么表示动作：

        regress    直接回归 (H,5)，L1 损失                     —— ACT
        discrete   每个维度离散成 n_bins 个格子，交叉熵         —— RT-1 / RT-2 / OpenVLA
        diffusion  从噪声去噪出整块动作，预测噪声，MSE          —— Diffusion Policy / π0

    回归头的参数名和形状保持不变（self.head 仍是同一个 Sequential），
    所以之前训的 checkpoint 还能加载。
    """

    def __init__(self, vocab, horizon=8, act_dim=5, state_dim=7, lang_dim=128,
                 hidden=512, lang_mode="seq", film=True, cams=("front", "wrist"),
                 head="regress", n_bins=41, diff_steps=100, diff_infer_steps=10):
        super().__init__()
        self.cams, self.H, self.act_dim = tuple(cams), horizon, act_dim
        self.head_type, self.n_bins = head, n_bins
        self.diff_steps, self.diff_infer_steps = diff_steps, diff_infer_steps
        self.lang = LanguageEncoder(vocab, lang_dim, mode=lang_mode)
        self.enc = nn.ModuleDict({c: VisionEncoder(lang_dim=lang_dim, film=film) for c in self.cams})
        vis_dim = sum(self.enc[c].out_dim for c in self.cams)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        out_dim = {"regress": horizon * act_dim,
                   "discrete": horizon * act_dim * n_bins,
                   "diffusion": hidden}[head]
        self.head = nn.Sequential(
            nn.Linear(vis_dim + 64 + lang_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))
        if head == "diffusion":
            self.t_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
            self.denoiser = nn.Sequential(
                nn.Linear(hidden + 128 + horizon * act_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, horizon * act_dim))
            # 余弦噪声表（Nichol & Dhariwal 2021），比线性表在少步数下更稳
            t = torch.linspace(0, 1, diff_steps + 1)
            ac = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            self.register_buffer("alpha_bar", (ac / ac[0]).clamp(1e-5, 1.0))

    def backbone(self, batch):
        """视觉 + 语言 + 本体 → 一个条件向量（三种动作头共用）。"""
        z = self.lang(batch["tokens"])
        feats, heats = [], {}
        for c in self.cams:
            kp, heat = self.enc[c](batch[c], z)
            feats.append(kp)
            heats[c] = heat
        return torch.cat(feats + [self.state_mlp(batch["state"]), z], dim=-1), heats

    def forward(self, batch, return_heat=False):
        """推理：返回 (B,H,5) 的动作分块。"""
        h, heats = self.backbone(batch)
        if self.head_type == "regress":
            a = self.head(h).reshape(-1, self.H, self.act_dim)
            a = torch.cat([a[..., :4].tanh(), a[..., 4:]], dim=-1)
        elif self.head_type == "discrete":
            logits = self.head(h).reshape(-1, self.H, self.act_dim, self.n_bins)
            a = self._bin_centers(logits.argmax(-1))
        else:
            a = self._ddim_sample(self.head(h))
        return (a, heats) if return_heat else a

    # ------------------------------------------------------------ 离散 token 头
    def _bin_centers(self, idx):
        return idx.float() / (self.n_bins - 1) * 2 - 1

    def _to_bins(self, a):
        return ((a.clamp(-1, 1) + 1) / 2 * (self.n_bins - 1)).round().long()

    # ------------------------------------------------------------ 扩散头
    def _eps(self, cond, a_noisy, t):
        te = self.t_mlp(timestep_embedding(t, 64))
        x = torch.cat([cond, te, a_noisy.flatten(1)], dim=-1)
        return self.denoiser(x).reshape(-1, self.H, self.act_dim)

    @torch.no_grad()
    def _ddim_sample(self, cond):
        B = cond.shape[0]
        a = torch.randn(B, self.H, self.act_dim, device=cond.device)
        steps = torch.linspace(self.diff_steps, 0, self.diff_infer_steps + 1).long().to(cond.device)
        for i in range(self.diff_infer_steps):
            t, t_next = steps[i], steps[i + 1]
            ab, ab_next = self.alpha_bar[t], self.alpha_bar[t_next]
            eps = self._eps(cond, a, t.repeat(B))
            a0 = ((a - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            a = ab_next.sqrt() * a0 + (1 - ab_next).sqrt() * eps      # DDIM，确定性
        return a

    # ------------------------------------------------------------------ 损失
    def loss(self, batch):
        h, _ = self.backbone(batch)
        mask, target = batch["mask"], batch["action"]
        denom = mask.sum().clamp(min=1)

        if self.head_type == "regress":
            pred = self.head(h).reshape(-1, self.H, self.act_dim)
            pred = torch.cat([pred[..., :4].tanh(), pred[..., 4:]], dim=-1)
            loss = ((pred - target).abs().mean(-1) * mask).sum() / denom

        elif self.head_type == "discrete":
            logits = self.head(h).reshape(-1, self.H, self.act_dim, self.n_bins)
            tgt = self._to_bins(target)
            ce = F.cross_entropy(logits.reshape(-1, self.n_bins), tgt.reshape(-1),
                                 reduction="none").reshape_as(tgt).mean(-1)
            loss = (ce * mask).sum() / denom
            pred = self._bin_centers(logits.argmax(-1))

        else:
            t = torch.randint(1, self.diff_steps + 1, (h.shape[0],), device=h.device)
            ab = self.alpha_bar[t][:, None, None]
            noise = torch.randn_like(target)
            a_noisy = ab.sqrt() * target + (1 - ab).sqrt() * noise
            eps = self._eps(self.head(h), a_noisy, t)
            loss = (((eps - noise) ** 2).mean(-1) * mask).sum() / denom
            with torch.no_grad():
                pred = ((a_noisy - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)

        with torch.no_grad():
            l1 = ((pred - target).abs().mean(-1) * mask).sum() / denom
            grip = (((pred[..., 4] > 0) == (target[..., 4] > 0)).float() * mask).sum() / denom
        return loss, {"loss": loss.item(), "l1": l1.item(), "grip_acc": grip.item()}
