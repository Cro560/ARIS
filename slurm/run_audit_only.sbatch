#!/bin/bash
#SBATCH --account=torch_pr_526_general
#SBATCH --job-name=audit_only
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=${ARIS_REPO}/logs/22_audit_only_%j.out
#SBATCH --error=${ARIS_REPO}/logs/22_audit_only_%j.err

set -euo pipefail

PROJ=${ARIS_REPO}
cd $PROJ

# Use Python directly
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

echo "=== ENV (VARIANT 2: AUDIT-ONLY) ==="
echo "HOST=$(hostname)"
echo "IN_STRUCTURED=$IN_STRUCTURED"
echo "PYTHONPATH=$PYTHONPATH"
nvidia-smi || true
echo

if [ ! -f "$IN_STRUCTURED" ]; then
  echo "[FAIL] missing input: $IN_STRUCTURED"
  exit 10
fi

# Step09: Input audit gate
echo "=== [STEP 09] bottleneck supervise inputs ==="
$PY scripts/09_bottleneck_supervise_inputs.py \
  --in_jsonl "$IN_STRUCTURED" \
  --out_dir reports/audit_only \
  --llm_ckpt models/gemma-3-4b-it_full \
  --rules_version "$RULES_DIR"

SUP_OK=$(ls -1t reports/audit_only/supervised_inputs_*.jsonl | head -n 1)
echo "SUP_OK=$SUP_OK"

if [ ! -f "$SUP_OK" ] || [ "$(wc -l < "$SUP_OK")" -eq 0 ]; then
  echo "[FAIL] supervised_inputs empty or missing"
  exit 12
fi

# Step07: LLM generate
echo "=== [STEP 07] LLM generate ==="
OUT_NOTES="reports/audit_only/notes_${SLURM_JOB_ID}.jsonl"
$PY scripts/llm_universal_infer.py \
  --task notes \
  --in_jsonl "$SUP_OK" \
  --out_jsonl "$OUT_NOTES"

# Step12: Audit notes
echo "=== [STEP 12] AC audit ==="
export IN_NOTES_JSONL="$OUT_NOTES"
$PY scripts/12_audit_notes_with_ac.py

# Step08: Span align
echo "=== [STEP 08] span align ==="
export IN_SOURCE_JSONL="$SUP_OK"
$PY scripts/08_span_align_notes.py
OUT_BERT=$(ls -1t reports/bioclinicalbert_ready_*.jsonl | head -n 1)

# Step13: Gate mapping (NO BNN)
echo "=== [STEP 13] gate span mapping ==="
export IN_BERT_JSONL="$OUT_BERT"
$PY scripts/13_gate_span_mapping.py

BERT_OK=$(ls -1t reports/audit/bert_ok_*.jsonl | head -n 1)
BERT_BAD=$(ls -1t reports/audit/bert_quarantine_*.jsonl | head -n 1)

# Save outputs
mkdir -p reports/audit_only_final
cp -f "$BERT_OK" reports/audit_only_final/final_ok_${SLURM_JOB_ID}.jsonl
cp -f "$BERT_BAD" reports/audit_only_final/final_quarantine_${SLURM_JOB_ID}.jsonl

echo "=== DONE (VARIANT 2) ==="
wc -l reports/audit_only_final/final_ok_${SLURM_JOB_ID}.jsonl
