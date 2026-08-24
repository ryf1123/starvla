"""演示数据集：把 shard_*.npz 读进内存，切成 (观测, 未来 H 步动作) 的样本。

为什么预测"未来 H 步"而不是下一步（动作分块 / action chunking，ACT 的做法）：
    1) 20 Hz 下相邻动作高度相关，逐步预测会让误差每步累积（复合误差）；
    2) 抓取这类任务里"闭合夹爪"是个多步承诺，单步策略容易在半路反悔来回抖。
推理时对重叠的预测做时间集成（见 policy/eval.py）。

样本布局（H=8 时）：
    front (3,128,128) uint8   t 时刻前视图（CHW，归一化在模型里做）
    wrist (3,128,128) uint8   t 时刻腕视图
    state (7,)  float32       归一化后的本体状态
    tokens(L,)  int64         指令的字符 id（定长 padding）
    action(H,5) float32       t..t+H-1 的动作，越界处复制最后一帧
    mask  (H,)  float32       越界处为 0，不计 loss
"""
from __future__ import annotations

import glob, json
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_LEN = 20            # 指令最长字符数（模板最长 15 字，留余量）


class CharVocab:
    """字符级词表。指令是模板生成的，字符集很小（30 个左右），够用且好解释。"""

    PAD = 0

    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = ["<pad>"] + chars
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}

    def encode(self, text, max_len=MAX_LEN):
        ids = [self.stoi.get(c, 0) for c in text][:max_len]
        return np.array(ids + [self.PAD] * (max_len - len(ids)), dtype=np.int64)

    def __len__(self):
        return len(self.itos)

    def save(self):
        return {"itos": self.itos}

    @classmethod
    def load(cls, d):
        v = cls([])
        v.itos = d["itos"]
        v.stoi = {c: i for i, c in enumerate(v.itos) if i > 0}
        return v


def load_episodes(path):
    eps = []
    for shard in sorted(glob.glob(f"{path}/shard_*.npz")):
        z = np.load(shard, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        for i, m in enumerate(meta):
            e = dict(front=z[f"front_{i}"], wrist=z[f"wrist_{i}"],
                     state=z[f"state_{i}"], action=z[f"action_{i}"],
                     instruction=m["instruction"], spec=m["spec"])
            e["priv"] = z[f"priv_{i}"] if f"priv_{i}" in z else None
            eps.append(e)
    return eps


class DemoDataset(Dataset):
    """演示数据集。`grasp_oversample>1` 时会把夹爪开合前后的样本多放几份。"""
    def __init__(self, eps, vocab, horizon=8, state_stats=None, shift_aug=4, train=True,
                 text_table=None, state_history=1, grasp_oversample=1):
        self.eps, self.vocab, self.H = eps, vocab, horizon
        self.text_table = text_table
        # 多步历史观测：RoboVLMs 的 600+ 组受控实验结论是
        # "多步历史观测 + 连续动作 + 独立 policy head" 最好，而且与模型规模无关。
        # 这里取最近 k 帧的 (本体状态, 上一步动作) 拼成一个向量——比堆图像便宜得多。
        self.K = max(1, int(state_history))
        self.shift_aug, self.train = shift_aug, train
        self.index = [(i, t) for i, e in enumerate(eps) for t in range(len(e["action"]))]
        # 抓取时刻过采样：失败归因显示成败几乎全取决于**闭合夹爪那一刻对没对准**
        # （成功局水平误差中位数 11.7 mm，失败局 45.6 mm，方块半宽 22 mm，见 notes/12）。
        # 把夹爪状态翻转前后 ±3 步的样本多放几份，等于给这几步更大的权重。
        if grasp_oversample > 1:
            extra = []
            for i, e in enumerate(eps):
                g = e["action"][:, 4]
                flips = np.nonzero(np.diff(np.sign(g)))[0]
                for f in flips:
                    for t in range(max(0, f - 3), min(len(g), f + 4)):
                        extra += [(i, t)] * (grasp_oversample - 1)
            self.index += extra
        if state_stats is None:
            S = np.concatenate([e["state"] for e in eps])
            state_stats = (S.mean(0), S.std(0) + 1e-6)
        self.smean, self.sstd = state_stats
        self.tok = {e["instruction"]: vocab.encode(e["instruction"]) for e in eps}

    def __len__(self):
        return len(self.index)

    def _img(self, arr):
        """uint8 HWC → float CHW，训练时做随机平移（补边），不做色彩抖动。

        色彩抖动会破坏"红/绿/蓝"这个指令赖以定位的信号——任务是颜色条件的，
        对图像做色相扰动等于把标签擦掉。这是一个必须踩明白的坑。
        """
        if self.train and self.shift_aug:
            p = self.shift_aug
            arr = np.pad(arr, ((p, p), (p, p), (0, 0)), mode="edge")
            i, j = np.random.randint(0, 2 * p + 1, 2)
            arr = arr[i:i + 128, j:j + 128]
        # 保持 uint8：CPU→GPU 的搬运量少 4 倍，转 float 和归一化放到设备上做
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    def _state_vec(self, e, t):
        """最近 K 帧：[s_{t-K+1..t} 归一化] + [a_{t-K..t-1}]（越界处用第 0 帧 / 零动作补）。"""
        idx = np.clip(np.arange(t - self.K + 1, t + 1), 0, len(e["state"]) - 1)
        s = (e["state"][idx] - self.smean) / self.sstd
        if self.K == 1:
            return s[0]
        aidx = np.arange(t - self.K, t)
        a = np.where((aidx >= 0)[:, None], e["action"][np.clip(aidx, 0, None)], 0.0)
        return np.concatenate([s.reshape(-1), a.reshape(-1)])

    @property
    def state_dim(self):
        # K=1 时保持 7 维（和早期 checkpoint 兼容）；K>1 时每帧是 (状态 7 + 上一步动作 5)
        return 7 if self.K == 1 else self.K * 12

    def __getitem__(self, k):
        i, t = self.index[k]
        e = self.eps[i]
        T = len(e["action"])
        idx = np.clip(np.arange(t, t + self.H), 0, T - 1)
        mask = (np.arange(t, t + self.H) < T).astype(np.float32)
        st = self._state_vec(e, t)
        priv = e["priv"][t] if e.get("priv") is not None else np.zeros(10, np.float32)
        return dict(front=self._img(e["front"][t]), wrist=self._img(e["wrist"][t]),
                    priv=torch.from_numpy(priv),
                    state=torch.from_numpy(st.astype(np.float32)),
                    tokens=torch.from_numpy(self.tok[e["instruction"]]),
                    action=torch.from_numpy(e["action"][idx]),
                    instr_id=(self.text_table.index(e["instruction"]) if self.text_table else 0),
                    mask=torch.from_numpy(mask))
