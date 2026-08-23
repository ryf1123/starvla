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


def model_kwargs(cfg):
    """从 config 里取模型超参，带默认值——老的 config.yaml 里没有 head 之类的键。"""
    m = cfg["model"]
    return dict(horizon=m["horizon"], lang_mode=m["lang_mode"], film=m["film"],
                cams=tuple(m["cams"]), head=m.get("head", "regress"),
                n_bins=m.get("n_bins", 41), diff_steps=m.get("diff_steps", 100),
                diff_infer_steps=m.get("diff_infer_steps", 10))


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
    tr = DemoDataset(tr_eps, vocab, cfg["model"]["horizon"], train=True,
                     shift_aug=cfg["data"]["shift_aug"])
    va = DemoDataset(val_eps, vocab, cfg["model"]["horizon"],
                     state_stats=(tr.smean, tr.sstd), train=False)
    model = VLAPolicy(vocab=len(vocab), **model_kwargs(cfg))
    return tr, va, model, vocab


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

    tr, va, model, vocab = build(cfg)
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
            batch = {k: v.to(dev) for k, v in batch.items()}
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
                        vb = {k: v.to(dev) for k, v in vb.items()}
                        _, vi = model.loss(vb)
                        vals.append(vi)
                        if j >= 20:
                            break
                vmean = {f"val_{k}": float(np.mean([v[k] for v in vals])) for k in vals[0]}
                log.write(json.dumps(dict(step=step, **vmean)) + "\n"); log.flush()
                print("  val", {k: round(v, 4) for k, v in vmean.items()})
                torch.save(dict(model=model.state_dict(), cfg=cfg, vocab=vocab.save(),
                                state_stats=(tr.smean, tr.sstd)), f"{out}/latest.pt")
                model.train()
            if step >= total:
                break
    print(f"完成，用时 {(time.time()-t0)/60:.1f} 分钟 → {out}/latest.pt")


if __name__ == "__main__":
    main()
