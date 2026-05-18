import json, argparse
from typing import Dict, Any

def read_jsonl_map(path: str, key: str) -> Dict[str, Dict[str, Any]]:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "run_meta" in obj:
                continue
            k = obj.get(key)
            if k is None:
                continue
            m[str(k)] = obj
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_in", required=True, help="reports/audit/final_ok_CURRENT.jsonl")
    ap.add_argument("--bnn_gate_in", required=True, help="reports/bnn/bnn_gate_CURRENT.jsonl")
    ap.add_argument("--out_ok", required=True)
    ap.add_argument("--out_quarantine", required=True)
    ap.add_argument("--default_pass", action="store_true",
                    help="If doc_id missing in bnn_gate, treat as pass (NOT recommended).")
    args = ap.parse_args()

    gate_map = read_jsonl_map(args.bnn_gate_in, "doc_id")

    n_in = 0
    n_ok = 0
    n_q = 0
    n_missing = 0

    with open(args.audit_in, "r", encoding="utf-8") as fin, \
         open(args.out_ok, "w", encoding="utf-8") as fok, \
         open(args.out_quarantine, "w", encoding="utf-8") as fq:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_in += 1

            doc_id = str(obj.get("doc_id"))
            gate = gate_map.get(doc_id)

            if gate is None:
                n_missing += 1
                passed = True if args.default_pass else False
                gate_payload = {"missing": True, "passed": passed}
            else:
                gate_payload = gate.get("bnn_gate", {})
                passed = bool(gate_payload.get("passed", False))

            obj["bnn_gate"] = gate_payload

            if passed:
                fok.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                fq.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_q += 1

    print(f"[OK] wrote: {args.out_ok}")
    print(f"[OK] wrote: {args.out_quarantine}")
    print(f"[STAT] in={n_in} ok={n_ok} quarantine={n_q} missing_gate={n_missing}")

if __name__ == "__main__":
    main()
