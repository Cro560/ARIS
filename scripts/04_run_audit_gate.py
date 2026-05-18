import os, json, pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_automaton(pairs):
    import ahocorasick
    A = ahocorasick.Automaton()
    for tok, lab in pairs:
        A.add_word(tok, (lab, tok))
    A.make_automaton()
    return A

def match_with_index(text: str, engine: str, pairs, token2label):
    hits = {}
    low = (text or "").lower()

    if engine == "pyahocorasick":
        A = build_automaton(pairs)
        for _, (lab, tok) in A.iter(low):
            hits[lab] = hits.get(lab, 0) + 1
    else:
        for tok, lab in token2label.items():
            if tok in low:
                hits[lab] = hits.get(lab, 0) + low.count(tok)
    return hits

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_notes = os.environ.get("IN_NOTES_JSONL", "reports/notes_universal_smoke_3616512.jsonl")
    rules_dir = os.environ.get("RULES_DIR", "audit_rules/releases/_CURRENT")
    profile = os.environ.get("AUDIT_PROFILE", "strict")

    out_ok  = os.environ.get("OUT_OK_JSONL",  f"reports/audit/audited_ok_{ts}.jsonl")
    out_bad = os.environ.get("OUT_BAD_JSONL", f"reports/audit/quarantine_{ts}.jsonl")
    out_parq= os.environ.get("OUT_AUDIT_PARQ",f"reports/audit/audit_report_{ts}.parquet")

    thr_path = Path(rules_dir) / "thresholds.json"
    prof_path = Path(rules_dir) / "profiles.json"
    ac_pkl = Path(rules_dir) / "ac_index.pkl"

    if not thr_path.exists():  raise SystemExit(f"Missing thresholds.json in {rules_dir}")
    if not prof_path.exists(): raise SystemExit(f"Missing profiles.json in {rules_dir}")
    if not ac_pkl.exists():    raise SystemExit(f"Missing ac_index.pkl in {rules_dir} (run scripts/11_build_ac_index.py)")

    thr = load_json(str(thr_path))
    profiles = load_json(str(prof_path))
    if profile not in profiles:
        raise SystemExit(f"Profile '{profile}' not found; available={list(profiles.keys())}")
    prof_cfg = profiles[profile]

    payload = pickle.load(open(ac_pkl, "rb"))
    engine = payload.get("engine", "fallback_set")
    pairs = payload.get("pairs", [])
    token2label = payload.get("token2label", {})

    # If pyahocorasick not importable, force fallback
    if engine == "pyahocorasick":
        try:
            import ahocorasick  # noqa
        except Exception:
            engine = "fallback_set"

    min_total_hits = int(thr.get("min_total_hits", 1))
    required_labels = prof_cfg.get("required_labels", [])
    min_required_labels = int(prof_cfg.get("min_required_labels", 0))

    print("[INFO] IN:", in_notes)
    print("[INFO] RULES_DIR:", rules_dir, "PROFILE:", profile)
    print("[INFO] OUT_OK :", out_ok)
    print("[INFO] OUT_BAD:", out_bad)
    print("[INFO] OUT_PARQ:", out_parq)
    print("[INFO] enforce: min_total_hits=", min_total_hits,
          "required_labels=", required_labels,
          "min_required_labels=", min_required_labels)
    print("[INFO] engine:", engine, "n_tokens:", payload.get("n_tokens", len(token2label)))

    rows = read_jsonl(in_notes)
    ok, bad, report = [], [], []

    for r in rows:
        doc_id = r.get("doc_id")
        notes = r.get("notes") or r.get("text") or ""

        hits_by_label = match_with_index(notes, engine, pairs, token2label)
        n_hits_total = int(sum(hits_by_label.values()))
        req_hit = sum(1 for lab in required_labels if hits_by_label.get(lab, 0) > 0)

        reasons = []
        blocked = False

        if n_hits_total < min_total_hits:
            blocked = True
            reasons.append({"type":"min_total_hits", "need":min_total_hits, "got":n_hits_total})

        if min_required_labels > 0 and req_hit < min_required_labels:
            blocked = True
            reasons.append({"type":"min_required_labels", "need":min_required_labels, "got":req_hit,
                            "required_labels": required_labels})

        r["audit"] = {
            "rules_release": str(rules_dir),
            "profile": profile,
            "blocked": bool(blocked),
            "n_hits_total": n_hits_total,
            "hits_by_label": hits_by_label,
            "reasons": reasons,
        }

        rep = {"doc_id": doc_id, "blocked": bool(blocked), "n_hits_total": n_hits_total}
        for k,v in hits_by_label.items():
            rep[f"hits_{k}"] = int(v)
        report.append(rep)

        (bad if blocked else ok).append(r)

    Path(out_ok).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_ok, ok)
    write_jsonl(out_bad, bad)

    df = pd.DataFrame(report)
    df.to_parquet(out_parq, index=False)

    print(f"[INFO] ok={len(ok)} bad={len(bad)} total={len(rows)}")
    if len(df):
        print(df["blocked"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
