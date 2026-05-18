import os, json, argparse
from datetime import datetime

def pick_first(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default="reports/audit/final_ok_CURRENT.jsonl")
    ap.add_argument("--out_jsonl", default="reports/bnn/bnn_input_CURRENT.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    n_in = 0
    n_out = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line=line.strip()
            if not line: 
                continue
            n_in += 1
            obj = json.loads(line)

            doc_id = pick_first(obj, ["doc_id","id","note_id","row_id"], default=f"row_{n_in:06d}")
            text   = pick_first(obj, ["text","note_text","input_text","question"], default="")

            # 先给最稳妥的 features：直接用 text
            # 后续我们会把 spans/audit_summary/evidence 追加进去
            features_text = text

            audit_meta = {
                "rules_release": pick_first(obj, ["rules_release","rules_version","rules_tag","release"], default="CURRENT"),
                "profile": pick_first(obj, ["profile","audit_profile"], default=None),
                "has_spans": ("spans" in obj),
                "has_audit": ("audit" in obj),
            }

            out = {
                "doc_id": doc_id,
                "text": text,
                "features_text": features_text,
                "audit_meta": audit_meta
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print("[OK] wrote:", args.out_jsonl)
    print("[STAT] in:", n_in, "out:", n_out)

if __name__ == "__main__":
    main()
