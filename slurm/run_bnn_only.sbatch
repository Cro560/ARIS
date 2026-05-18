#!/bin/bash
#SBATCH --account=torch_pr_526_general
#SBATCH --job-name=bnn_only
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=${ARIS_REPO}/logs/23_bnn_only_%j.out
#SBATCH --error=${ARIS_REPO}/logs/23_bnn_only_%j.err

set -euo pipefail

PROJ=${ARIS_REPO}
cd $PROJ

PY="$PROJ/conda_envs/llm310/bin/python"
export PATH="$(dirname $PY):$PATH"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

$PY -V

export HF_HOME=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

export MODEL_DIR=${ARIS_REPO}/models/gemma-3-4b-it_full
export RULES_DIR=audit_rules/releases/_CURRENT

IN_STRUCTURED=${1:-data/structured_demo.jsonl}

echo "=== ENV (VARIANT 3: BNN-ONLY) ==="
echo "HOST=$(hostname)"
echo "IN_STRUCTURED=$IN_STRUCTURED"
nvidia-smi || true
echo

if [ ! -f "$IN_STRUCTURED" ]; then
  echo "[FAIL] missing input: $IN_STRUCTURED"
  exit 10
fi

# SKIP Step09: NO input audit - use raw inputs directly
SUP_OK="$IN_STRUCTURED"

# Step07: LLM generate
echo "=== [STEP 07] LLM generate ==="
OUT_NOTES="reports/bnn_only/notes_${SLURM_JOB_ID}.jsonl"
mkdir -p reports/bnn_only
$PY scripts/llm_universal_infer.py \
  --task notes \
  --in_jsonl "$SUP_OK" \
  --out_jsonl "$OUT_NOTES"

# Step12: Audit notes (for BNN input)
echo "=== [STEP 12] AC audit ==="
export IN_NOTES_JSONL="$OUT_NOTES"
$PY scripts/12_audit_notes_with_ac.py

# Step08: Span align
echo "=== [STEP 08] span align ==="
export IN_SOURCE_JSONL="$SUP_OK"
$PY scripts/08_span_align_notes.py
OUT_BERT=$(ls -1t reports/bioclinicalbert_ready_*.jsonl | head -n 1)

# 关键修复：不是从reports/audit/读，而是用刚生成的OUT_BERT
echo "=== [STEP 12-BNN] make BNN input ==="
mkdir -p reports/bnn
$PY scripts/12_make_bnn_input_from_final_ok.py \
  --in_jsonl "$OUT_BERT" \
  --out_jsonl reports/bnn/bnn_input_${SLURM_JOB_ID}.jsonl

BNN_INPUT="reports/bnn/bnn_input_${SLURM_JOB_ID}.jsonl"

# Step13-BNN: BNN gate run
echo "=== [STEP 13-BNN] BNN gate run ==="
export BNN_INPUT_JSONL="$BNN_INPUT"
$PY scripts/13_bnn_gate_run.py
BNN_OUT=$(ls -1t reports/bnn/bnn_pred_*.jsonl | head -n 1)

# Step14-BNN: Postprocess
echo "=== [STEP 14-BNN] BNN postprocess ==="
export BNN_PRED_JSONL="$BNN_OUT"
$PY scripts/14_bnn_postprocess.py
BNN_GATE=$(ls -1t reports/bnn/bnn_gate_*.jsonl | head -n 1)

# Step15: Merge
echo "=== [STEP 15] merge audit and BNN ==="
export IN_BERT_JSONL="$OUT_BERT"
export BNN_GATE_JSONL="$BNN_GATE"
$PY scripts/15_merge_audit_and_bnn_gate.py

FINAL_OK=$(ls -1t reports/final/final_ok_*.jsonl | head -n 1)
FINAL_Q=$(ls -1t reports/final/final_quarantine_*.jsonl | head -n 1)

# Save outputs
mkdir -p reports/bnn_only_final
cp -f "$FINAL_OK" reports/bnn_only_final/final_ok_${SLURM_JOB_ID}.jsonl
cp -f "$FINAL_Q" reports/bnn_only_final/final_quarantine_${SLURM_JOB_ID}.jsonl

echo "=== DONE (VARIANT 3) ==="
wc -l reports/bnn_only_final/final_ok_${SLURM_JOB_ID}.jsonl
