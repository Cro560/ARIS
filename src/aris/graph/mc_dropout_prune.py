#!/usr/bin/env python3
import os, json, math, argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------
# Utils
# -----------------------------
def log(msg: str):
    print(msg, flush=True)

def enable_mc_dropout(model: torch.nn.Module):
    """
    Keep model in eval() but turn on dropout modules.
    Standard MC-dropout trick: model.eval(); set dropout layers to train().
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def softmax_np(x: np.ndarray, axis: int = -1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def predictive_entropy(p_mean: np.ndarray) -> np.ndarray:
    # p_mean: [N, C]
    return -(p_mean * np.log(p_mean + 1e-12)).sum(axis=1)

def expected_entropy(p_all: np.ndarray) -> np.ndarray:
    # p_all: [T, N, C]
    ent = -(p_all * np.log(p_all + 1e-12)).sum(axis=2)  # [T,N]
    return ent.mean(axis=0)

def mutual_information(p_all: np.ndarray) -> np.ndarray:
    # MI = H[E[p]] - E[H[p]]
    p_mean = p_all.mean(axis=0)  # [N,C]
    return predictive_entropy(p_mean) - expected_entropy(p_all)

def chunk_list(xs: List[Any], bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i:i+bs]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step2: MC-dropout uncertainty + pruning for KG resampling pipeline")
    ap.add_argument("--in_ner", required=True, help="Input parquet (e.g., weighted_ner_data.parquet)")
    ap.add_argument("--model_path", required=True, help="Trained ClinicalBERT model dir (HF format)")
    ap.add_argument("--out_prefix", default="step2", help="Output prefix for files")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--T", type=int, default=10, help="MC-dropout passes")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prune_quantile", type=float, default=0.10,
                    help="Prune bottom fraction by score (0.10 means drop lowest 10%%)")
    ap.add_argument("--score_mode", choices=["mi", "entropy"], default="mi",
                    help="Uncertainty score: mutual information (epistemic-ish) or predictive entropy (total)")
    ap.add_argument("--keep_high_uncertainty", action="store_true",
                    help="If set, KEEP highest-uncertainty samples (drop lowest). Default: keep low-uncertainty.")
    ap.add_argument("--text_col", default="entity_text", help="Text column in NER parquet to score")
    ap.add_argument("--group_cols", default="note_id,entity_label,section_norm",
                    help="Columns to group by for one-score-per-group")
    ap.add_argument("--min_group_size", type=int, default=1)
    ap.add_argument("--no_token_type_ids", action="store_true",
                    help="Some tokenizers/models don't use token_type_ids; disable passing it")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    log(f"[INFO] device={device} cuda_available={torch.cuda.is_available()}")

    # Load data
    log(f"[INFO] reading: {args.in_ner}")
    df = pd.read_parquet(args.in_ner)
    for c in [args.text_col]:
        if c not in df.columns:
            raise ValueError(f"Missing text_col={c} in columns: {list(df.columns)}")
    df[args.text_col] = df[args.text_col].astype(str).fillna("")

    # Grouping
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"Missing group col={c} in columns: {list(df.columns)}")

    log(f"[INFO] grouping by: {group_cols}")
    g = df.groupby(group_cols, dropna=False)

    # Create group table to score: one representative text per group (concat small sample)
    # For stability: take up to 3 entity_text and join
    grp_rows = []
    for key, sub in g:
        if len(sub) < args.min_group_size:
            continue
        # concat a few strings to represent the group
        texts = sub[args.text_col].astype(str).head(3).tolist()
        rep_text = " ; ".join(texts)
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["_rep_text"] = rep_text
        row["_group_size"] = int(len(sub))
        grp_rows.append(row)

    grp = pd.DataFrame(grp_rows)
    log(f"[INFO] groups to score: {len(grp):,}")

    # Load model/tokenizer
    log(f"[INFO] loading model/tokenizer from: {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    mdl.to(device)

    # MC-dropout enable
    enable_mc_dropout(mdl)

    # Score groups
    texts = grp["_rep_text"].tolist()
    N = len(texts)
    C = int(mdl.config.num_labels)

    log(f"[INFO] num_labels={C} | MC T={args.T} | batch={args.batch_size} | max_len={args.max_len}")
    all_scores = np.zeros((N,), dtype=np.float32)

    # We will compute p_all in streaming to avoid RAM blowup:
    # For each batch, do T forward passes -> p_all [T, b, C] -> uncertainty score [b]
    mdl.eval()  # keep base eval; dropout modules are in train()

    with torch.no_grad():
        offset = 0
        for batch_texts in chunk_list(texts, args.batch_size):
            enc = tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            if args.no_token_type_ids and "token_type_ids" in enc:
                enc.pop("token_type_ids", None)

            # p_all: [T, b, C]
            p_list = []
            for _ in range(args.T):
                out = mdl(**enc)
                logits = out.logits.detach().float().cpu().numpy()  # [b,C]
                p = softmax_np(logits, axis=1)
                p_list.append(p)
            p_all = np.stack(p_list, axis=0)  # [T,b,C]

            if args.score_mode == "mi":
                sc = mutual_information(p_all)  # [b]
            else:
                sc = predictive_entropy(p_all.mean(axis=0))  # [b]

            bsz = len(batch_texts)
            all_scores[offset:offset+bsz] = sc.astype(np.float32)
            offset += bsz

            if offset % (args.batch_size * 50) == 0:
                log(f"[INFO] scored {offset:,}/{N:,} groups...")

    grp["uncertainty_score"] = all_scores

    # Decide pruning threshold
    q = float(args.prune_quantile)
    if not (0.0 <= q < 1.0):
        raise ValueError("--prune_quantile must be in [0,1)")
    if q == 0.0:
        thr = None
        keep_mask = np.ones((len(grp),), dtype=bool)
    else:
        if args.keep_high_uncertainty:
            # keep high -> drop lowest q
            thr = float(np.quantile(grp["uncertainty_score"].values, q))
            keep_mask = grp["uncertainty_score"].values >= thr
        else:
            # default: keep low -> drop highest q
            thr = float(np.quantile(grp["uncertainty_score"].values, 1.0 - q))
            keep_mask = grp["uncertainty_score"].values <= thr

    kept_groups = grp[keep_mask].copy()
    log(f"[INFO] prune_quantile={q} keep_high_uncertainty={args.keep_high_uncertainty}")
    log(f"[INFO] threshold={thr} kept_groups={len(kept_groups):,}/{len(grp):,} ({len(kept_groups)/max(1,len(grp)):.2%})")

    # Join back to original df to keep only groups
    # Build merge keys dataframe
    keep_keys = kept_groups[group_cols].copy()
    keep_keys["_KEEP"] = 1
    df2 = df.merge(keep_keys, on=group_cols, how="inner")
    df2 = df2.drop(columns=["_KEEP"])

    # Outputs
    out_scores = f"{args.out_prefix}_group_uncertainty.parquet"
    out_kept = f"{args.out_prefix}_ner_pruned.parquet"
    out_stats = f"{args.out_prefix}_stats.json"

    kept_groups.to_parquet(out_scores, index=False)
    df2.to_parquet(out_kept, index=False)

    stats = {
        "in_ner": args.in_ner,
        "model_path": args.model_path,
        "device": str(device),
        "T": args.T,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "score_mode": args.score_mode,
        "keep_high_uncertainty": bool(args.keep_high_uncertainty),
        "prune_quantile": q,
        "threshold": thr,
        "groups_total": int(len(grp)),
        "groups_kept": int(len(kept_groups)),
        "rows_in": int(len(df)),
        "rows_out": int(len(df2)),
        "score_desc": grp["uncertainty_score"].describe(percentiles=[0.01,0.05,0.1,0.5,0.9,0.95,0.99]).to_dict(),
    }
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log("[OK] wrote:")
    log(f"  - {out_scores}")
    log(f"  - {out_kept}")
    log(f"  - {out_stats}")

    # quick prints
    log("[INFO] score describe:")
    for k,v in stats["score_desc"].items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
