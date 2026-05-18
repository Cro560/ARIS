#!/bin/bash
#SBATCH --account=torch_pr_526_general
#SBATCH --job-name=baseline_free
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=${ARIS_REPO}/logs/21_baseline_free_llm_%j.out
#SBATCH --error=${ARIS_REPO}/logs/21_baseline_free_llm_%j.err

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

echo "=== ENV (BASELINE FREE) ==="
echo "HOST=$(hostname)"
echo "IN_STRUCTURED=$IN_STRUCTURED"
nvidia-smi || true
echo

if [ ! -f "$RULES_DIR/ac_index.pkl" ]; then
  echo "[WARN] missing ac_index.pkl; building..."
  python scripts/11_build_ac_index.py
fi

# -------------------------------------------
# Baseline definition:
#   - NO Step09 bottleneck supervise / input gate
#   - NO BNN
#   - Still run: Step07 -> Step12 -> Step08 -> Step13
# -------------------------------------------

echo "=== [STEP 07] LLM universal infer (notes) - FREE ==="
OUT_NOTES="reports/notes_free_${SLURM_JOB_ID}.jsonl"
python scripts/llm_universal_infer.py \
  --task notes \
  --in_jsonl "$IN_STRUCTURED" \
  --out_jsonl "$OUT_NOTES"

echo "WROTE_NOTES=$OUT_NOTES"
ls -lh "$OUT_NOTES"
echo

echo "=== [STEP 12] AC audit notes ==="
export IN_NOTES_JSONL="$OUT_NOTES"
python scripts/12_audit_notes_with_ac.py

echo "=== [STEP 08] span align notes -> bioclinicalbert_ready ==="
export IN_SOURCE_JSONL="$IN_STRUCTURED"
python scripts/08_span_align_notes.py
OUT_BERT=$(ls -1t reports/bioclinicalbert_ready_*.jsonl | head -n 1)
echo "OUT_BERT=$OUT_BERT"
echo

echo "=== [STEP 13] gate span mapping ==="
export IN_BERT_JSONL="$OUT_BERT"
python scripts/13_gate_span_mapping.py

BERT_OK=$(ls -1t reports/audit/bert_ok_*.jsonl | head -n 1)
BERT_BAD=$(ls -1t reports/audit/bert_quarantine_*.jsonl | head -n 1)

mkdir -p reports/baseline
cp -f "$BERT_OK"  reports/baseline/final_ok_${SLURM_JOB_ID}.jsonl
cp -f "$BERT_BAD" reports/baseline/final_quarantine_${SLURM_JOB_ID}.jsonl
ln -sf "final_ok_${SLURM_JOB_ID}.jsonl" reports/baseline/final_ok_CURRENT.jsonl
ln -sf "final_quarantine_${SLURM_JOB_ID}.jsonl" reports/baseline/final_quarantine_CURRENT.jsonl

echo "=== DONE (BASELINE FREE) ==="
ls -lah reports/baseline/final_ok_CURRENT.jsonl reports/baseline/final_quarantine_CURRENT.jsonl
wc -l   reports/baseline/final_ok_CURRENT.jsonl reports/baseline/final_quarantine_CURRENT.jsonl || true
