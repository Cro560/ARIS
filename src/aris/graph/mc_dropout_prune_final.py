import os, json, math, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p: [B, C], float64 preferred
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)  # [B]


@torch.no_grad()
def mc_scores(
    texts,
    tok,
    model,
    device,
    T: int = 10,
    max_len: int = 128,
    batch_size: int = 64,
):
    """
    Return dict of numpy arrays, length = len(texts)
    - entropy: predictive entropy H(p_bar)
    - exp_entropy: E[H(p_t)]
    - mi: mutual information = H(p_bar) - E[H(p_t)]
    - var_logits: mean over classes of Var_t(logits)
    """
    n = len(texts)
    ent = np.zeros(n, dtype=np.float64)
    exp_ent = np.zeros(n, dtype=np.float64)
    mi = np.zeros(n, dtype=np.float64)
    var_logits = np.zeros(n, dtype=np.float64)

    # IMPORTANT: enable dropout
    model.train()
    # reduce other randomness impact: make sure we don't update weights anyway
    # (still no_grad), so ok.

    for i0 in range(0, n, batch_size):
        batch_texts = texts[i0:i0+batch_size]
        enc = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits_T = []
        probs_T = []

        for _ in range(T):
            out = model(**enc)
            logits = out.logits  # [B, C] float32
            # move to float64 for stable entropy/MI accumulation
            logits64 = logits.to(torch.float64)
            probs64 = torch.softmax(logits64, dim=-1)
            logits_T.append(logits64)
            probs_T.append(probs64)

        # stack: [T, B, C]
        logits_stack = torch.stack(logits_T, dim=0)
        probs_stack = torch.stack(probs_T, dim=0)

        p_bar = probs_stack.mean(dim=0)               # [B, C]
        H_bar = entropy_from_probs(p_bar)             # [B]
        H_t = entropy_from_probs(probs_stack.view(-1, probs_stack.size(-1))).view(T, -1)  # [T,B]
        EH = H_t.mean(dim=0)                          # [B]
        MI = H_bar - EH                               # [B]

        # var over T of logits, then mean across classes
        V = logits_stack.var(dim=0, unbiased=False).mean(dim=-1)  # [B]

        bsz = len(batch_texts)
        ent[i0:i0+bsz] = H_bar.cpu().numpy()
        exp_ent[i0:i0+bsz] = EH.cpu().numpy()
        mi[i0:i0+bsz] = MI.cpu().numpy()
        var_logits[i0:i0+bsz] = V.cpu().numpy()

        if (i0 // batch_size) % 20 == 0:
            print(f"[INFO] scored {min(i0+bsz,n):,}/{n:,} groups...")

    return {
        "entropy": ent,
        "exp_entropy": exp_ent,
        "mi": mi,
        "var_logits": var_logits,
    }


def pick_score_col(scores: pd.DataFrame, mode: str) -> str:
    mode = mode.lower()
    if mode in scores.columns:
        return mode
    raise ValueError(f"Unknown score_mode={mode}. Available: {list(scores.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ner", required=True, help="Input parquet with columns: note_id, entity_label, section_norm, (text columns)")
    ap.add_argument("--model_path", required=True, help="HF path or local dir")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--score_mode", default="mi", choices=["mi","entropy","exp_entropy","var_logits"])
    ap.add_argument("--prune_quantile", type=float, default=0.10, help="fraction of groups to prune (exact by rank)")
    ap.add_argument("--keep_high_uncertainty", action="store_true", help="If set, keep highest-uncertainty groups and prune low; default keeps most and prunes low")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device} cuda_available={torch.cuda.is_available()}")

    df = pd.read_parquet(args.in_ner)
    key = ["note_id","entity_label","section_norm"]
    need = set(key)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # representative text per group
    # priority: _rep_text if exists else try a reasonable text column
    if "_rep_text" in df.columns:
        rep_col = "_rep_text"
    else:
        # try common candidates
        candidates = [c for c in ["text","span_text","entity_text","chunk_text","sentence","context"] if c in df.columns]
        if not candidates:
            raise ValueError("No _rep_text and no obvious text column found (text/span_text/entity_text/...)")
        rep_col = candidates[0]
    print(f"[INFO] using rep text column: {rep_col}")

    g = (
        df.groupby(key, as_index=False)
          .agg(_rep_text=(rep_col, "first"), _group_size=(rep_col, "size"))
    )
    print(f"[INFO] groups to score: {len(g):,}")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)

    print(f"[INFO] num_labels={model.config.num_labels} | MC T={args.T} | batch={args.batch_size} | max_len={args.max_len}")

    scores = mc_scores(
        texts=g["_rep_text"].tolist(),
        tok=tok,
        model=model,
        device=device,
        T=args.T,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    scores_df = pd.DataFrame(scores)
    out = pd.concat([g, scores_df], axis=1)

    score_col = pick_score_col(scores_df, args.score_mode)
    out["uncertainty_score"] = out[score_col].astype(np.float64)

    # exact pruning by rank (avoid quantile tie issues)
    n = len(out)
    k = int(math.floor(args.prune_quantile * n))
    k = max(0, min(k, n))

    # tie-break: stable order by (score, note_id, entity_label, section_norm)
    out = out.sort_values(
        by=["uncertainty_score","note_id","entity_label","section_norm"],
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    # By default we prune LOW uncertainty groups (keep most)
    # - if keep_high_uncertainty: we keep top (high), prune low
    # Either way, "pruned" means removed groups.
    prune_low = True  # prune low by default
    if args.keep_high_uncertainty:
        prune_low = True

    if k == 0:
        kept = out.copy()
    else:
        if prune_low:
            # remove bottom k
            kept = out.iloc[k:].copy()
        else:
            # remove top k
            kept = out.iloc[:-k].copy()

    print(f"[INFO] prune_quantile={args.prune_quantile} k={k} kept_groups={len(kept):,}/{n:,} ({len(kept)/n*100:.2f}%)")
    print(f"[INFO] score_mode={args.score_mode} | kept score range: [{kept['uncertainty_score'].min():.6g}, {kept['uncertainty_score'].max():.6g}]")

    # build pruned NER rows by inner-join on kept keys
    kept_keys = kept[key].copy()
    pruned_df = df.merge(kept_keys, on=key, how="inner")

    out_prefix = args.out_prefix
    out_unc = Path(f"{out_prefix}_group_uncertainty.parquet")
    out_prn = Path(f"{out_prefix}_ner_pruned.parquet")
    out_js  = Path(f"{out_prefix}_stats.json")

    out.to_parquet(out_unc, index=False)
    pruned_df.to_parquet(out_prn, index=False)

    stats = {
        "input_rows": int(len(df)),
        "input_groups": int(n),
        "kept_groups": int(len(kept)),
        "kept_rows": int(len(pruned_df)),
        "score_mode": args.score_mode,
        "T": int(args.T),
        "batch_size": int(args.batch_size),
        "max_len": int(args.max_len),
        "device": str(device),
        "score_summary": out["uncertainty_score"].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_dict(),
    }
    out_js.write_text(json.dumps(stats, indent=2))
    print("[OK] wrote:")
    print("  -", out_unc)
    print("  -", out_prn)
    print("  -", out_js)


if __name__ == "__main__":
    main()
