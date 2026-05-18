import argparse, os, subprocess, sys
from datetime import datetime

def sh(cmd: str):
    print(f"[CMD] {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_final_ok", default="reports/audit/final_ok_CURRENT.jsonl")
    ap.add_argument("--tag", default="CURRENT", help="output tag, e.g. CURRENT or run_YYYYMMDD_HHMMSS")
    ap.add_argument("--encoder_model", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--bnn_n_samples", type=int, default=10)
    ap.add_argument("--conf_threshold", type=float, default=0.30)
    ap.add_argument("--max_total_unc", type=float, default=10.0)
    args = ap.parse_args()

    tag = args.tag
    os.makedirs("reports/bnn", exist_ok=True)
    os.makedirs("reports/final", exist_ok=True)

    bnn_in  = f"reports/bnn/bnn_input_{tag}.jsonl"
    bnn_pred = f"reports/bnn/bnn_pred_{tag}.jsonl"
    bnn_gate = f"reports/bnn/bnn_gate_{tag}.jsonl"
    out_ok = f"reports/final/final_ok_{tag}.jsonl"
    out_q  = f"reports/final/final_quarantine_{tag}.jsonl"

    # 1) make bnn input
    sh(f"python scripts/12_make_bnn_input_from_final_ok.py --in_jsonl {args.audit_final_ok} --out_jsonl {bnn_in}")

    # 2) run bnn
    sh(
        "PYTHONPATH=${ARIS_REPO}:$PYTHONPATH "
        f"python scripts/13_bnn_gate_run.py "
        f"--encoder_model {args.encoder_model} "
        f"--in_jsonl {bnn_in} "
        f"--out_jsonl {bnn_pred} "
        f"--bnn_n_samples {args.bnn_n_samples}"
    )

    # 3) postprocess
    sh(
        f"python scripts/14_bnn_postprocess.py "
        f"--in_jsonl {bnn_pred} "
        f"--out_jsonl {bnn_gate} "
        f"--conf_threshold {args.conf_threshold} "
        f"--max_total_unc {args.max_total_unc} "
        f"--drop_activations"
    )

    # 4) merge
    sh(
        f"python scripts/15_merge_audit_and_bnn_gate.py "
        f"--audit_in {args.audit_final_ok} "
        f"--bnn_gate_in {bnn_gate} "
        f"--out_ok {out_ok} "
        f"--out_quarantine {out_q}"
    )

    print("[DONE]")
    print("OK:", out_ok)
    print("QUARANTINE:", out_q)

if __name__ == '__main__':
    main()
