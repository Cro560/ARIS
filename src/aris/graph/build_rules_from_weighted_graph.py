#!/usr/bin/env python3
import argparse, os, glob, json
from collections import defaultdict
import pandas as pd

def pick_latest(patterns):
    cands = []
    for pat in patterns:
        cands += glob.glob(pat)
    if not cands:
        return None
    cands = sorted(cands, key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def first_existing_cols(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser(
        description="STEP3: Build rule lexicon from Step2 MC-dropout outputs (low-uncertainty subsets)."
    )
    ap.add_argument("--run_dir", default=".",
                    help="Working directory that contains step2 outputs and weighted_ner_data.parquet")
    ap.add_argument("--step2_prefix", default="step2_mc",
                    help="Prefix used in Step2 (--out_prefix). We'll auto-detect matching parquet files.")
    ap.add_argument("--weighted_ner", default="weighted_ner_data.parquet",
                    help="Step1 output parquet to pull surface forms from if Step2 output lacks text columns.")
    ap.add_argument("--out_dir", default="step3_out", help="Output directory")
    ap.add_argument("--prune_quantile", type=float, default=0.10,
                    help="For logging only; does not re-prune (Step2 already pruned).")
    ap.add_argument("--max_rules_per_label", type=int, default=3000,
                    help="Cap rules per label (top by frequency).")
    ap.add_argument("--min_count", type=int, default=3,
                    help="Minimum frequency to keep a phrase as a rule candidate.")
    ap.add_argument("--max_phrase_len", type=int, default=80,
                    help="Drop extremely long phrases (likely junk).")
    ap.add_argument("--lower", action="store_true", help="Lowercase phrases (recommended for AC matching).")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.chdir(run_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1) Locate Step2 outputs (best-effort, robust to file naming) ----
    # We try common patterns; if your Step2 writes different names, we still have a fallback.
    scored = pick_latest([
        f"{args.step2_prefix}*scor*.parquet",
        f"{args.step2_prefix}*uncert*.parquet",
        f"{args.step2_prefix}*score*.parquet",
        f"{args.step2_prefix}*.parquet",
    ])

    # pruned output (if exists)
    pruned = pick_latest([
        f"{args.step2_prefix}*prun*.parquet",
        f"{args.step2_prefix}*keep*.parquet",
        f"{args.step2_prefix}*filtered*.parquet",
    ])

    # If scored is ambiguous (the pruned file might be matched), keep both but treat pruned separately.
    if scored and pruned and os.path.samefile(scored, pruned):
        pruned = None

    print(f"[STEP3] run_dir={run_dir}")
    print(f"[STEP3] detected scored={scored}")
    print(f"[STEP3] detected pruned={pruned}")

    # ---- 2) Load inputs ----
    df_scored = None
    if scored and os.path.exists(scored):
        df_scored = pd.read_parquet(scored)
        print(f"[STEP3] scored shape: {df_scored.shape}")

    # Always load weighted_ner_data as the ultimate source of surface forms
    if not os.path.exists(args.weighted_ner):
        raise FileNotFoundError(f"Missing weighted ner parquet: {args.weighted_ner}")
    ner = pd.read_parquet(args.weighted_ner, columns=None)
    print(f"[STEP3] weighted_ner shape: {ner.shape}")

    # ---- 3) Determine key columns and build phrase table ----
    # We expect weighted_ner has at least: note_id, entity_label, section_norm
    for c in ["note_id", "entity_label"]:
        if c not in ner.columns:
            raise ValueError(f"[FATAL] weighted_ner missing required column: {c}")

    section_col = "section_norm" if "section_norm" in ner.columns else None

    # Surface-form column candidates (adapt to your schema)
    phrase_col = first_existing_cols(
        ner, ["entity_text", "span_text", "text", "entity", "matched_text", "mention"]
    )
    if phrase_col is None:
        raise ValueError(
            "[FATAL] Cannot find a surface-form column in weighted_ner_data.parquet. "
            "Expected one of: entity_text/span_text/text/entity/matched_text/mention"
        )

    # Basic phrase cleanup
    phrases = ner[["note_id", "entity_label", phrase_col] + ([section_col] if section_col else [])].copy()
    phrases.rename(columns={phrase_col: "phrase"}, inplace=True)
    phrases["phrase"] = phrases["phrase"].astype(str)

    if args.lower:
        phrases["phrase"] = phrases["phrase"].str.lower()

    phrases["phrase"] = phrases["phrase"].str.replace(r"\s+", " ", regex=True).str.strip()
    phrases = phrases[phrases["phrase"].str.len().between(1, args.max_phrase_len)]

    # ---- 4) If Step2 scored output exists, use it to keep only "good" low-uncertainty groups ----
    # We try to join on the same grouping keys Step2 used: note_id, entity_label, section_norm.
    # If we can't find uncertainty columns, we still produce frequency-based rules (fallback).
    if df_scored is not None:
        # Guess score column names
        unc_col = first_existing_cols(df_scored, ["uncertainty", "mi", "score_mi", "score", "unc"])
        keep_col = first_existing_cols(df_scored, ["keep", "is_kept", "kept", "pruned_keep"])
        # Also find keys
        key_cols = [c for c in ["note_id", "entity_label", "section_norm"] if c in df_scored.columns]
        print(f"[STEP3] scored keys={key_cols} unc_col={unc_col} keep_col={keep_col}")

        if key_cols and (keep_col or unc_col):
            scored_small = df_scored[key_cols + ([unc_col] if unc_col else []) + ([keep_col] if keep_col else [])].copy()

            # If keep flag exists, filter by it; else keep everything (user already pruned in Step2 anyway)
            if keep_col:
                scored_small = scored_small[scored_small[keep_col].astype(bool)]
                print(f"[STEP3] after keep-flag filter: {scored_small.shape}")

            # Merge: retain only phrases that belong to kept groups
            phrases_before = len(phrases)
            phrases = phrases.merge(scored_small.drop_duplicates(key_cols), on=key_cols, how="inner")
            print(f"[STEP3] phrases filtered by scored/keep: {phrases_before:,} -> {len(phrases):,}")

            # Attach uncertainty if available
            if unc_col and unc_col in phrases.columns:
                pass
            elif unc_col and unc_col in scored_small.columns:
                # Already merged, should exist
                pass
        else:
            print("[STEP3] scored output exists but cannot interpret keys/uncertainty; fallback to frequency-only rules.")
    else:
        print("[STEP3] no scored parquet detected; fallback to frequency-only rules.")

    # ---- 5) Build lexicon candidates (label + optional section + phrase) ----
    group_cols = ["entity_label", "phrase"] + (["section_norm"] if section_col else [])
    agg = phrases.groupby(group_cols).size().reset_index(name="count")

    # If uncertainty column attached, compute mean uncertainty per phrase group
    if "uncertainty" in phrases.columns:
        m = phrases.groupby(group_cols)["uncertainty"].mean().reset_index(name="mean_uncertainty")
        agg = agg.merge(m, on=group_cols, how="left")
    elif "mi" in phrases.columns:
        m = phrases.groupby(group_cols)["mi"].mean().reset_index(name="mean_uncertainty")
        agg = agg.merge(m, on=group_cols, how="left")
    else:
        agg["mean_uncertainty"] = None

    # Filter small counts
    agg = agg[agg["count"] >= args.min_count].copy()

    # For each label, keep top-N by count (ties stable)
    agg.sort_values(["entity_label", "count"], ascending=[True, False], inplace=True)
    agg = agg.groupby("entity_label", group_keys=False).head(args.max_rules_per_label)

    out_dir = args.out_dir
    lex_tsv = os.path.join(out_dir, "rules_lexicon.tsv")
    agg.to_csv(lex_tsv, sep="\t", index=False)
    print(f"[STEP3] wrote: {lex_tsv} (rows={len(agg):,})")

    # ---- 6) Build Aho–Corasick-ready JSON structure ----
    # We output:
    # {
    #   "LABEL": [{"phrase":"...", "section_norm":"...", "count":..., "mean_uncertainty":...}, ...],
    #   ...
    # }
    rules = defaultdict(list)
    for _, row in agg.iterrows():
        item = {
            "phrase": row["phrase"],
            "count": int(row["count"]),
        }
        if section_col:
            item["section_norm"] = row.get("section_norm", None)
        mu = row.get("mean_uncertainty", None)
        if pd.notna(mu):
            item["mean_uncertainty"] = float(mu)
        rules[row["entity_label"]].append(item)

    rules_json = os.path.join(out_dir, "rules_aho_corasick.json")
    with open(rules_json, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"[STEP3] wrote: {rules_json}")

    # ---- 7) Emit a “clean index” for next-round training/reweighting (note-level coverage) ----
    # This is intentionally light-weight: which notes/labels have rule evidence post-prune.
    idx = phrases[["note_id", "entity_label"] + (["section_norm"] if section_col else [])].drop_duplicates()
    idx_path = os.path.join(out_dir, "step3_clean_index.parquet")
    idx.to_parquet(idx_path, index=False)
    print(f"[STEP3] wrote: {idx_path} (rows={len(idx):,})")

    # ---- 8) Quick diagnostics ----
    diag = os.path.join(out_dir, "step3_summary.txt")
    with open(diag, "w") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"scored={scored}\n")
        f.write(f"pruned={pruned}\n")
        f.write(f"weighted_ner={args.weighted_ner}\n")
        f.write(f"phrase_col={phrase_col}\n")
        f.write(f"section_col={section_col}\n")
        f.write(f"min_count={args.min_count}\n")
        f.write(f"max_rules_per_label={args.max_rules_per_label}\n")
        f.write(f"rules_rows={len(agg)}\n")
        f.write(f"unique_labels={agg['entity_label'].nunique()}\n")
    print(f"[STEP3] wrote: {diag}")

    print("[STEP3] DONE.")

if __name__ == "__main__":
    main()
