"""预训练文本编码器（第五环）：用真正学过中文的模型来编码指令。

为什么需要它：本项目默认的字符级编码器，词表是**从训练指令的字符里建的**。
「拿」「起」「到」「里」这些字根本不在词表里，所以换一种说法就等于把指令删掉一半。
要回答"能不能泛化到没见过的措辞"，必须换一个见过中文的模型。

两种用法，对应一个真实的取舍（实测数字见 notes/04）：

    ppool  只取句向量（CLS）。bge-small-zh 下
           cos(「把红色方块放进黄色盘子」,「把黄色方块放进红色盘子」) = **0.998**——
           句向量模型是按"语义相似"训的，颜色对调在它眼里几乎是同一句话。
           换措辞很鲁棒（0.946），但**角色区分没了**。
    ptok   取 token 级特征，交给一个可学的注意力池化去读。
           顺序信息还在，模型自己学"第一个颜色词是要抓的"。

结论预告：**冻结的句向量给你措辞鲁棒性，但会把角色信息压掉；要两者兼得就得用 token 级特征。**
这也是真实 VLA 用 VLM 的 token 序列、而不是一个句子 embedding 的原因。
"""
from __future__ import annotations

import numpy as np
import torch

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
MAX_LEN = 24


class PretrainedText:
    """按需加载 HF 编码器，带缓存。指令只有几十种，编码一次就够。"""

    _cache = {}

    def __init__(self, name=DEFAULT_MODEL, device="cpu"):
        from transformers import AutoTokenizer, AutoModel
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.dim = self.model.config.hidden_size
        self._memo = {}

    @torch.no_grad()
    def encode(self, texts):
        """→ feats (B, MAX_LEN, D) float32, mask (B, MAX_LEN) bool（True = padding）"""
        todo = [t for t in texts if t not in self._memo]
        if todo:
            b = self.tok(todo, padding="max_length", truncation=True,
                         max_length=MAX_LEN, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state.float().cpu()
            m = (b["attention_mask"] == 0).cpu()
            for i, t in enumerate(todo):
                self._memo[t] = (h[i], m[i])
        feats = torch.stack([self._memo[t][0] for t in texts])
        mask = torch.stack([self._memo[t][1] for t in texts])
        return feats, mask


class TextTable:
    """把数据集里出现过的所有指令编码成一张表，训练时按 id 取，零开销。"""

    def __init__(self, texts, name=DEFAULT_MODEL):
        self.enc = PretrainedText(name)
        self.texts = sorted(set(texts))
        self.id = {t: i for i, t in enumerate(self.texts)}
        self.feats, self.mask = self.enc.encode(self.texts)
        self.dim = self.feats.shape[-1]

    def index(self, text):
        return self.id.get(text, -1)
