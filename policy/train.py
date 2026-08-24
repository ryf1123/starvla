"""行为克隆训练入口。

    python -m policy.train --config configs/bc_place.yaml --name bc_v1
    python -m policy.train --config configs/bc_place.yaml --name bow --set model.lang_mode=bow

约定和 SONIC 项目一致：每个实验一个 runs/<name>/，里面有 config.yaml、log.jsonl、latest.pt。
"""
from __future__ import annotations

import argparse, json, os, time, copy
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from policy.dataset import DemoDataset, CharVocab, load_episodes
from policy.model import VLAPolicy


def deep_set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    cur = d.get(keys[-1])
    if isinstance(value, str) and value.startswith("["):        # --set model.cams=[front,wrist]
        value = [x.strip() for x in value.strip("[]").split(",") if x.strip()]
    elif isinstance(cur, bool) or value in ("true", "false"):
        value = value == "true"
    elif isinstance(cur, int) and not isinstance(cur, bool) and not isinstance(value, list):
        value = int(value)
    elif isinstance(cur, float):
        value = float(value)
    d[keys[-1]] = value


def attach_text(batch, table, dev):
    """把 batch 搬到设备上；用预训练文本编码器时，按 instr_id 从表里取 token 特征。"""
    out = {k: v.to(dev) for k, v in batch.items()}
    if table is not None:
        idx = batch["instr_id"].long()
        out["lang_feat"] = table.feats[idx].to(dev)
        out["lang_mask"] = table.mask[idx].to(dev)
    return out


def model_kwargs(cfg):
    """从 config 里取模型超参，带默认值——老的 config.yaml 里没有 head 之类的键。"""
    m = cfg["model"]
    return dict(horizon=m["horizon"], lang_mode=m["lang_mode"], film=m["film"],
                cams=tuple(m["cams"]), head=m.get("head", "regress"),
                n_bins=m.get("n_bins", 41), diff_steps=m.get("diff_steps", 100),
                diff_infer_steps=m.get("diff_infer_steps", 10),
                ss_raw=m.get("ss_raw", True), last_stride=m.get("last_stride", 1),
                backbone=m.get("backbone", "cnn"), pretrained=m.get("pretrained", True),
                freeze_backbone=m.get("freeze_backbone", False),
                aux_weight=m.get("aux_weight", 0.0), pre_dim=m.get("pre_dim", 512))


def state_dim_of(cfg):
    K = max(1, cfg["model"].get("state_history", 1))
    return 7 if K == 1 else K * 12


def build(cfg, eps=None):
    eps = eps if eps is not None else load_episodes(cfg["data"]["path"])
    limit = cfg["data"].get("limit")
    if limit:                      # 数据量曲线：固定随机种子取前 N 条，保证是子集关系
        eps = [eps[i] for i in np.random.default_rng(0).permutation(len(eps))[:int(limit)]]
    vocab = CharVocab([e["instruction"] for e in eps])
    n_val = max(1, int(len(eps) * cfg["data"]["val_frac"]))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(eps))
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]
    table = None
    if cfg["model"]["lang_mode"] in ("ppool", "ptok"):
        from policy.text_encoder import TextTable
        table = TextTable([e["instruction"] for e in eps],
                          cfg["model"].get("text_model", "BAAI/bge-small-zh-v1.5"))
        print(f"预训练文本编码器：{len(table.texts)} 条不同指令 → {tuple(table.feats.shape)}")
    K = cfg["model"].get("state_history", 1)
    tr = DemoDataset(tr_eps, vocab, cfg["model"]["horizon"], train=True,
                     shift_aug=cfg["data"]["shift_aug"], text_table=table, state_history=K)
    va = DemoDataset(val_eps, vocab, cfg["model"]["horizon"],
                     state_stats=(tr.smean, tr.sstd), train=False, text_table=table, state_history=K)
    model = VLAPolicy(vocab=len(vocab), state_dim=tr.state_dim, **model_kwargs(cfg))
    return tr, va, model, vocab, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        deep_set(cfg, k, v)
    if args.steps:
        cfg["train"]["steps"] = args.steps

    out = f"runs/{args.name}"
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"), allow_unicode=True)

    tr, va, model, vocab, table = build(cfg)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)
    print(f"训练样本 {len(tr)}  验证样本 {len(va)}  参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M  设备 {dev}")

    bs = cfg["train"]["batch_size"]
    dl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=cfg["train"]["workers"],
                    drop_last=True, persistent_workers=cfg["train"]["workers"] > 0)
    vdl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["wd"])
    total = cfg["train"]["steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["train"]["lr"], total_steps=total,
                                                pct_start=0.05)
    log = open(f"{out}/log.jsonl", "a")
    step, t0 = 0, time.time()
    while step < total:
        for batch in dl:
            batch = attach_text(batch, table, dev)
            loss, info = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % cfg["train"]["log_every"] == 0:
                rec = dict(step=step, lr=sched.get_last_lr()[0], **info,
                           sps=step * bs / (time.time() - t0))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"step {step}/{total} l1 {info['l1']:.4f} grip {info['grip_acc']:.3f} "
                      f"{rec['sps']:.0f} sample/s")
            if step % cfg["train"]["val_every"] == 0 or step == total:
                model.eval()
                vals = []
                with torch.no_grad():
                    for j, vb in enumerate(vdl):
                        vb = attach_text(vb, table, dev)
                        _, vi = model.loss(vb)
                        vals.append(vi)
                        if j >= 20:
                            break
                vmean = {f"val_{k}": float(np.mean([v[k] for v in vals])) for k in vals[0]}
                vmean.update(model.vision_diag(vb))    # 视觉分支健康检查
                log.write(json.dumps(dict(step=step, **vmean)) + "\n"); log.flush()
                print("  val", {k: round(v, 4) for k, v in vmean.items()})
                ck = dict(model=model.state_dict(), cfg=cfg, vocab=vocab.save(),
                          state_stats=(tr.smean, tr.sstd))
                torch.save(ck, f"{out}/latest.pt")
                # 中途存档：用来画"闭环成功率 vs 训练步数"的曲线。
                # 这条曲线是必要的——同一配置 8000 步 10%、16000 步 72.5%，
                # 而验证 L1 只差 16%（见 notes/02）。
                if step in set(cfg["train"].get("save_ckpts", [])):
                    torch.save(ck, f"{out}/ckpt_{step}.pt")
                model.train()
            if step >= total:
                break
    print(f"完成，用时 {(time.time()-t0)/60:.1f} 分钟 → {out}/latest.pt")


if __name__ == "__main__":
    main()
