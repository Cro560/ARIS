#!/bin/bash
#SBATCH --account=torch_pr_526_general
#SBATCH --job-name=full_bayes
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=${ARIS_REPO}/logs/20_full_bayes_pipe_%j.out
#SBATCH --error=${ARIS_REPO}/logs/20_full_bayes_pipe_%j.err

set -euo pipefail

PROJ=${ARIS_REPO}
cd $PROJ

# Use Python directly (like previous successful jobs)
PY="$PROJ/conda_envs/llm310/bin/python"
export PATH="$(dirname $PY):$PATH"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

# Verify Python
$PY -V
$PY -c "import torch; print('[OK] torch', torch.__version__)" || echo "[WARN] torch check failed"

export HF_HOME=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

export MODEL_DIR=${ARIS_REPO}/models/gemma-3-4b-it_full
export RULES_DIR=audit_rules/releases/_CURRENT
export AUDIT_PROFILE=strict

IN_STRUCTURED=${1:-data/structured_demo.jsonl}

echo "=== ENV ==="
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "IN_STRUCTURED=$IN_STRUCTURED"
echo "MODEL_DIR=$MODEL_DIR"
echo "RULES_DIR=$RULES_DIR"
echo "AUDIT_PROFILE=$AUDIT_PROFILE"
nvidia-smi || true
echo

# -----------------------------
# 1) Preflight checks
# -----------------------------
if [ ! -f "$IN_STRUCTURED" ]; then
  echo "[FAIL] missing input jsonl: $IN_STRUCTURED"
  exit 10
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "[FAIL] missing model dir: $MODEL_DIR"
  exit 11
fi
if [ ! -f "$RULES_DIR/ac_index.pkl" ]; then
  echo "[WARN] missing ac_index.pkl; building..."
  python scripts/11_build_ac_index.py
fi

# Ensure BNN init ckpt exists
mkdir -p checkpoints/bnn/CURRENT
if [ ! -f checkpoints/bnn/CURRENT/bnn_init.pt ]; then
  echo "[WARN] missing bnn_init.pt; creating init ckpt..."
  python - <<'PY'
import torch
from bayes_bnn.bnn_model import BayesianReasoningModule
bnn = BayesianReasoningModule(
    llm_embedding_dim=768,
    hidden_dims=[512,256,128],
    output_dim=1,
    prior_std=1.0,
    n_samples=10,
)
torch.save(bnn.state_dict(), "checkpoints/bnn/CURRENT/bnn_init.pt")
print("[OK] wrote checkpoints/bnn/CURRENT/bnn_init.pt")
PY
fi

# -----------------------------
# 2) Step09: Bottleneck supervise inputs (strict AC input gate)
# -----------------------------
echo "=== [STEP 09] bottleneck supervise inputs ==="
python scripts/09_bottleneck_supervise_inputs.py \
  --in_jsonl "$IN_STRUCTURED" \
  --out_dir reports/bottleneck \
  --llm_ckpt models/gemma-3-4b-it_full \
  --rules_version "$RULES_DIR"

SUP_OK=$(ls -1t reports/bottleneck/supervised_inputs_*.jsonl | head -n 1)
SUP_Q=$(ls -1t reports/bottleneck/quarantine_inputs_*.jsonl | head -n 1)
echo "SUP_OK=$SUP_OK"
echo "SUP_Q=$SUP_Q"
wc -l "$SUP_OK" "$SUP_Q" || true
echo

if [ "$(wc -l < "$SUP_OK")" -eq 0 ]; then
  echo "[FAIL] supervised_inputs is empty; nothing to generate."
  exit 12
fi

# -----------------------------
# 3) Step07: LLM generate notes
# -----------------------------
echo "=== [STEP 07] LLM universal infer (notes) ==="
OUT_NOTES="reports/notes_bn_${SLURM_JOB_ID}.jsonl"
python scripts/llm_universal_infer.py \
  --task notes \
  --in_jsonl "$SUP_OK" \
  --out_jsonl "$OUT_NOTES"

echo "WROTE_NOTES=$OUT_NOTES"
ls -lh "$OUT_NOTES"
head -n 1 "$OUT_NOTES" | cut -c1-260 || true
echo

# -----------------------------
# 4) Step12: AC audit notes
# -----------------------------
echo "=== [STEP 12] AC audit notes ==="
export IN_NOTES_JSONL="$OUT_NOTES"
python scripts/12_audit_notes_with_ac.py

AUD_OK=$(ls -1t reports/audit/audited_ok_*.jsonl | head -n 1)
AUD_BAD=$(ls -1t reports/audit/quarantine_*.jsonl | head -n 1)
ln -sf "$(basename "$AUD_OK")"  reports/audit/notes_ok_CURRENT.jsonl
ln -sf "$(basename "$AUD_BAD")" reports/audit/notes_quarantine_CURRENT.jsonl

echo "NOTES_OK_CURRENT=reports/audit/notes_ok_CURRENT.jsonl"
echo "NOTES_QUAR_CURRENT=reports/audit/notes_quarantine_CURRENT.jsonl"
echo

# -----------------------------
# 5) Step08: span align → bioclinicalbert_ready
# -----------------------------
echo "=== [STEP 08] span align notes -> bioclinicalbert_ready ==="
export IN_SOURCE_JSONL="$SUP_OK"
python scripts/08_span_align_notes.py

OUT_BERT=$(ls -1t reports/bioclinicalbert_ready_*.jsonl | head -n 1)
echo "OUT_BERT=$OUT_BERT"
echo

# -----------------------------
# 6) Step13: fail-closed gate on span mapping (AC fallback)
# -----------------------------
echo "=== [STEP 13] gate span mapping ==="
export IN_BERT_JSONL="$OUT_BERT"
python scripts/13_gate_span_mapping.py

BERT_OK=$(ls -1t reports/audit/bert_ok_*.jsonl | head -n 1)
BERT_BAD=$(ls -1t reports/audit/bert_quarantine_*.jsonl | head -n 1)

ln -sf "$(basename "$BERT_OK")"  reports/audit/final_ok_CURRENT.jsonl
ln -sf "$(basename "$BERT_BAD")" reports/audit/final_quarantine_CURRENT.jsonl

echo "AUDIT_FINAL_OK=reports/audit/final_ok_CURRENT.jsonl"
echo "AUDIT_FINAL_QUAR=reports/audit/final_quarantine_CURRENT.jsonl"
wc -l reports/audit/final_ok_CURRENT.jsonl reports/audit/final_quarantine_CURRENT.jsonl || true
echo

# -----------------------------
# 7) BNN chain: 12 -> 13 -> 14 -> 15
# -----------------------------
echo "=== [BNN-STEP 12] make bnn input from audit final_ok ==="
mkdir -p reports/bnn
python scripts/12_make_bnn_input_from_final_ok.py \
  --in_jsonl reports/audit/final_ok_CURRENT.jsonl \
  --out_jsonl reports/bnn/bnn_input_CURRENT.jsonl

echo "=== [BNN-STEP 13] encoder + bnn inference ==="
PYTHONPATH=${ARIS_REPO}:${PYTHONPATH:-} \
python scripts/13_bnn_gate_run.py \
  --encoder_model emilyalsentzer/Bio_ClinicalBERT \
  --in_jsonl reports/bnn/bnn_input_CURRENT.jsonl \
  --out_jsonl reports/bnn/bnn_pred_CURRENT.jsonl \
  --bnn_hidden_dims 512,256,128 \
  --bnn_output_dim 1 \
  --bnn_prior_std 1.0 \
  --bnn_n_samples 10 \
  --bnn_ckpt checkpoints/bnn/CURRENT/bnn_init.pt

echo "=== [BNN-STEP 14] postprocess gate ==="
python scripts/14_bnn_postprocess.py \
  --in_jsonl reports/bnn/bnn_pred_CURRENT.jsonl \
  --out_jsonl reports/bnn/bnn_gate_CURRENT.jsonl \
  --conf_threshold 0.30 \
  --max_total_unc 10.0 \
  --drop_activations

echo "=== [BNN-STEP 15] merge audit + bnn gate => final ==="
mkdir -p reports/final
python scripts/15_merge_audit_and_bnn_gate.py \
  --audit_in reports/audit/final_ok_CURRENT.jsonl \
  --bnn_gate_in reports/bnn/bnn_gate_CURRENT.jsonl \
  --out_ok reports/final/final_ok_CURRENT.jsonl \
  --out_quarantine reports/final/final_quarantine_CURRENT.jsonl

echo "=== DONE (BAYES+BNN) ==="
ls -lah reports/final/final_ok_CURRENT.jsonl reports/final/final_quarantine_CURRENT.jsonl
wc -l   reports/final/final_ok_CURRENT.jsonl reports/final/final_quarantine_CURRENT.jsonl || true
tail -n 2 reports/final/final_ok_CURRENT.jsonl || true
tail -n 2 reports/final/final_quarantine_CURRENT.jsonl || true
