import json, argparse
from typing import Dict, Any

def sigmoid(x: float) -> float:
    import math
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--conf_threshold", type=float, default=0.30,
                    help="Pass if confidence >= threshold.")
    ap.add_argument("--max_total_unc", type=float, default=10.0,
                    help="Fail if total_uncertainty > this.")
    ap.add_argument("--drop_activations", action="store_true",
                    help="Remove layer_activations from bnn_output (recommended).")
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    n_pass = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:

        # First line can be run_meta; keep it but also write a derived meta line
        first = fin.readline()
        if first:
            obj0 = json.loads(first)
            if "run_meta" in obj0:
                meta = obj0
                meta["postprocess"] = {
                    "conf_threshold": args.conf_threshold,
                    "max_total_unc": args.max_total_unc,
                    "drop_activations": bool(args.drop_activations),
                    "schema": "bnn_gate_v1",
                }
                fout.write(json.dumps(meta, ensure_ascii=False) + "\n")
            else:
                # not a meta line; process as normal record
                fin.seek(0)

        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)

            doc_id = obj.get("doc_id")
            bnn = obj.get("bnn_output", {}) or {}
            audit_meta = obj.get("audit_meta", {}) or {}

            mean = float(bnn.get("prediction_mean", 0.0))
            std = float(bnn.get("prediction_std", 0.0))
            epi = float(bnn.get("epistemic_uncertainty", std))
            ale = float(bnn.get("aleatoric_uncertainty", 0.0))
            tot = float(bnn.get("total_uncertainty", epi + ale))
            conf = float(bnn.get("confidence", 0.0))
            kl = float(bnn.get("kl_divergence", 0.0))

            # Optional: convert mean to probability if you want a bounded score
            prob = clamp01(sigmoid(mean))

            passed = (conf >= args.conf_threshold) and (tot <= args.max_total_unc)

            out: Dict[str, Any] = {
                "doc_id": doc_id,
                "audit_meta": audit_meta,
                "bnn_gate": {
                    "schema": "bnn_gate_v1",
                    "prediction_mean": mean,
                    "prediction_std": std,
                    "prob": prob,
                    "confidence": conf,
                    "epistemic_uncertainty": epi,
                    "aleatoric_uncertainty": ale,
                    "total_uncertainty": tot,
                    "kl_divergence": kl,
                    "passed": bool(passed),
                    "rules": {
                        "conf_threshold": args.conf_threshold,
                        "max_total_unc": args.max_total_unc,
                    },
                },
            }

            if not args.drop_activations:
                # keep only if explicitly requested
                if "layer_activations" in bnn:
                    out["bnn_gate"]["layer_activations"] = bnn["layer_activations"]

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            n_pass += int(passed)

    print(f"[OK] wrote: {args.out_jsonl}")
    print(f"[STAT] in={n_in} out={n_out} pass={n_pass} pass_rate={(n_pass/max(n_out,1)):.3f}")

if __name__ == "__main__":
    main()
